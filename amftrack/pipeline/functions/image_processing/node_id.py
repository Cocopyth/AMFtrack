import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from scipy import sparse
import cv2
import itertools
import json

from amftrack.pipeline.functions.image_processing.experiment_class_surf import orient
from amftrack.util.sys import get_dirname
import os
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    save_graphs,
    load_graphs,
)


def first_identification(nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, tolerance):
    #
    corresp = {}
    ambiguous = set()
    to_remove = set()
    degree_3sup_nodes_tm1 = [
        node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) >= 3
    ]
    degree_3sup_nodes_t = [
        node for node in nx_graph_t.nodes if nx_graph_t.degree(node) >= 3
    ]
    xs = [pos_tm1[node][0] for node in degree_3sup_nodes_tm1] + [
        pos_t[node][0] for node in degree_3sup_nodes_t
    ]+[0]
    ys = [pos_tm1[node][1] for node in degree_3sup_nodes_tm1] + [
        pos_t[node][1] for node in degree_3sup_nodes_t
    ]+[0]
    bounds = (np.max(xs) + 10000, np.max(ys) + 10000)
    Stm1 = sparse.csr_matrix(bounds, dtype=int)
    St = sparse.csr_matrix(bounds, dtype=int)
    for node in degree_3sup_nodes_tm1:
        Stm1[pos_tm1[node][0], pos_tm1[node][1]] = node
    for node in degree_3sup_nodes_t:
        St[pos_t[node][0], pos_t[node][1]] = node
    for node1 in degree_3sup_nodes_tm1:
        mini = np.inf
        posanchor = pos_tm1[node1]
        window = 150
        potential_surrounding_t = St[
            max(0, posanchor[0] - 2 * window) : posanchor[0] + 2 * window,
            max(0, posanchor[1] - 2 * window) : posanchor[1] + 2 * window,
        ]
        for node2 in potential_surrounding_t.data:
            distance = np.linalg.norm(pos_tm1[node1] - pos_t[node2])
            if distance < mini:
                mini = distance
                identifier = node2
        if mini < tolerance:
            if identifier in corresp.values():
                ambiguous.add(node1)
            corresp[node1] = identifier
        else:
            to_remove.add(node1)
    print("before ambiguities", len(corresp.keys()))
    while len(ambiguous) > 0:
        node = ambiguous.pop()
        identifier = corresp[node]
        candidates = [nod for nod in corresp.keys() if corresp[nod] == identifier]
        mini = np.inf
        for candidate in candidates:
            distance = node_dist(
                candidate,
                identifier,
                nx_graph_tm1,
                nx_graph_t,
                pos_tm1,
                pos_t,
                tolerance,
            )
            if distance < mini:
                right_candidate = candidate
                mini = distance
        for candidate in candidates:
            if candidate != right_candidate:
                corresp.pop(candidate)
                to_remove.add(candidate)
                ambiguous.discard(candidate)
    return (corresp, to_remove)


def find_closest_edge(tip, Sedge, nx_graph_t, pos_t, nx_graph_tm1, pos_tm1):
    posanchor = pos_tm1[tip]
    window = 1000
    potential_surrounding_t = Sedge[
        max(0, posanchor[0] - 2 * window) : posanchor[0] + 2 * window,
        max(0, posanchor[1] - 2 * window) : posanchor[1] + 2 * window,
    ]
    mini = np.inf
    for node_root in potential_surrounding_t.data:
        for edge in nx_graph_t.edges(int(node_root)):
            pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
            if np.linalg.norm(np.array(pixel_list[0]) - np.array(pos_tm1[tip])) <= 5000:
                distance = np.min(
                    np.linalg.norm(
                        np.array(pixel_list) - np.array(pos_tm1[tip]), axis=1
                    )
                )
                if distance < mini:
                    mini = distance
                    right_edge = edge
    if mini == np.inf:
        print(f"didnt find a tip to match tip in pos {posanchor}")
        return (None, None, None, None)
    origin = np.array(
        orient(
            nx_graph_tm1.get_edge_data(*list(nx_graph_tm1.edges(tip))[0])["pixel_list"],
            pos_tm1[tip],
        )
    )
    origin_vector = origin[0] - origin[-1]
    branch = np.array(
        orient(
            nx_graph_t.get_edge_data(*right_edge)["pixel_list"],
            pos_t[right_edge[0]],
        )
    )
    candidate_vector = branch[-1] - branch[0]
    dot_product = np.dot(origin_vector, candidate_vector)
    if dot_product >= 0:
        root = right_edge[0]
        next_node = right_edge[1]
    else:
        root = right_edge[1]
        next_node = right_edge[0]
    last_node = root
    current_node = next_node
    last_branch = np.array(
        orient(
            nx_graph_t.get_edge_data(root, next_node)["pixel_list"],
            pos_t[current_node],
        )
    )
    return (last_node, next_node, last_branch, current_node)


def track(
    tip,
    current_node,
    last_branch,
    last_node,
    length_id,
    nx_graph_t,
    pos_t,
    nx_graph_tm1,
    pos_tm1,
    corresp_tips,
    ambiguous,
    identified,
):
    i = 0
    loop = []
    while (
        nx_graph_t.degree(current_node) != 1 and not current_node in identified
    ):  # Careful : if there is a cycle with low angle this might loop indefinitely but unprobable
        i += 1
        if i >= 100:
            print(
                "identified infinite loop",
                i,
                tip,
                current_node,
                pos_t[current_node],
            )
            break
        mini = np.inf
        origin_vector = (
            last_branch[0] - last_branch[min(length_id, len(last_branch) - 1)]
        )
        unit_vector_origin = origin_vector / np.linalg.norm(origin_vector)
        candidate_vectors = []
        for neighbours_t in nx_graph_t.neighbors(current_node):
            if neighbours_t != last_node:
                branch_candidate = np.array(
                    orient(
                        nx_graph_t.get_edge_data(current_node, neighbours_t)[
                            "pixel_list"
                        ],
                        pos_t[current_node],
                    )
                )
                candidate_vector = (
                    branch_candidate[min(length_id, len(branch_candidate) - 1)]
                    - branch_candidate[0]
                )
                unit_vector_candidate = candidate_vector / np.linalg.norm(
                    candidate_vector
                )
                candidate_vectors.append(unit_vector_candidate)
                dot_product = np.dot(unit_vector_origin, unit_vector_candidate)
                angle = np.arccos(dot_product)
                if angle < mini:
                    mini = angle
                    next_node = neighbours_t
        #                     print('angle',dot_product,pos_t[last_node],pos_t[current_node],pos_t[neighbours_t],angle/(2*np.pi)*360)
        #!!!bug may happen here if two nodes are direct neighbours : I would nee to check further why it the case, optimal segmentation should avoid this issue.
        # This is especially a problem for degree 4 nodes. Maybe fuse nodes that are closer than 3 pixels.
        # Update on comment above, the fusing has been done. However the current tracking methodology may fail for short edges.
        if i >= 100:
            print(mini / (2 * np.pi) * 360, pos_t[next_node])
            if next_node in loop:
                break
            else:
                loop.append(next_node)
        assert len(candidate_vectors) >= 2, "candidate_vectors < 2"
        edge_couples = itertools.combinations(candidate_vectors, 2)
        competitor = np.max(
            [
                np.arccos(np.dot(candidate_vectors[0], -candidate_vectors[1]))
                for candidate_vectors in edge_couples
            ]
        )
        # Look if the continuation is indeed best, including the case of degree 4 or more. This is a difficult ambiguous case. Handling is what it is
        # Then candidate vectors is longer than 2.
        if mini < competitor:
            current_node, last_node = next_node, current_node
        else:
            break
    if current_node in identified:
        if last_node not in identified:
            if last_node in corresp_tips.values():
                ambiguous.add(tip)
            corresp_tips[int(tip)] = int(last_node)
    else:
        if current_node in corresp_tips.values():
            ambiguous.add(tip)
        corresp_tips[int(tip)] = int(current_node)
    return (ambiguous, corresp_tips)


def handle_ambiguous(ambiguous, corresp_tips, pos_tm1, pos_t):
    while len(ambiguous) > 0:  # improve ambiguity resolving!
        node = ambiguous.pop()
        identifier = corresp_tips[node]
        candidates = [
            nod for nod in corresp_tips.keys() if corresp_tips[nod] == identifier
        ]
        mini = np.inf
        for candidate in candidates:
            distance = np.linalg.norm(pos_tm1[candidate] - pos_t[identifier])
            if distance < mini:
                right_candidate = candidate
                mini = distance
        for candidate in candidates:
            if candidate != right_candidate:
                corresp_tips.pop(candidate)
                ambiguous.discard(candidate)
    return corresp_tips


def second_identification(exp, t, tp1, length_id=200, tolerance=50, save=False):
    nx_graph_tm1, nx_graph_t, pos_tm1, pos_t = (
        exp.nx_graph[t],
        exp.nx_graph[tp1],
        exp.positions[t],
        exp.positions[tp1],
    )
    reconnect_degree_2(nx_graph_t, pos_t)
    reconnect_degree_2(nx_graph_tm1, pos_tm1)
    corresp, to_remove = first_identification(
        nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, tolerance
    )
    identified = corresp.values()
    downstream_graphs = [nx_graph_t]
    downstream_pos = [pos_t]
    corresp_tips = {int(node): int(corresp[node]) for node in corresp.keys()}
    tips = [node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) == 1]
    ambiguous = set()
    Sedge = sparse.csr_matrix((50000, 60000), dtype=int)
    for edge in nx_graph_t.edges:
        pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
        pixela = pixel_list[0]
        pixelb = pixel_list[-1]
        Sedge[pixela[0], pixela[1]] = edge[0]
        Sedge[pixelb[0], pixelb[1]] = edge[1]
    for i, tip in enumerate(tips):
        last_node, next_node, last_branch, current_node = find_closest_edge(
            tip, Sedge, nx_graph_t, pos_t, nx_graph_tm1, pos_tm1
        )
        if last_node is None:
            continue
        ambiguous, corresp_tips = track(
            tip,
            current_node,
            last_branch,
            last_node,
            length_id,
            nx_graph_t,
            pos_t,
            nx_graph_tm1,
            pos_tm1,
            corresp_tips,
            ambiguous,
            identified,
        )
    corresp_tips = handle_ambiguous(ambiguous, corresp_tips, pos_tm1, pos_t)
    exp.corresps[t] = corresp_tips
    if save:
        target = get_corresp_path(exp, t, tp1)
        print(target)
        with open(target, "w") as jsonf:
            json.dump(corresp_tips, jsonf, indent=4)
    return corresp_tips


def relabel_pos(poss, mapping):
    return {mapping(node): poss[node] for node in poss.keys()}


def get_corresp_path(exp, t, tp1):
    date = exp.dates[t]
    directory_name = get_dirname(date, exp.folders)
    datep1 = exp.dates[tp1]
    directory_namep1 = get_dirname(datep1, exp.folders)
    target = os.path.join(exp.directory, directory_name)
    suffix = f"Analysis/corresp{directory_name}_{directory_namep1}.json"
    target = os.path.join(target, suffix)
    return target


def create_corresp(exp):
    for t in range(exp.ts - 1):
        tp1 = t + 1
        second_identification(exp, t, tp1, length_id=200, tolerance=50, save=True)


def create_labeled_graph(exp):
    print("create labeled graph")
    corresp_list = []
    for t in range(exp.ts - 1):
        print(t, "ae")
        tp1 = t + 1
        corresp_path = get_corresp_path(exp, t, tp1)
        assert os.path.exists(corresp_path), "Not all corresp exist"
        with open(corresp_path) as json_file:
            corresp = json.load(json_file)
        corresp_list.append({int(corresp[key]): int(key) for key in corresp.keys()})
    new_graph_list = []
    new_poss_list = []
    # mappings = []
    for t in range(exp.ts):
        print("relabel1 ", t)
        def mapping(node, t=t):
            return recursive_mapping(corresp_list, t, node)

        new_graph = nx.relabel_nodes(exp.nx_graph[t], mapping, copy=True)
        new_poss = relabel_pos(exp.positions[t], mapping)
        new_graph_list.append(new_graph)
        new_poss_list.append(new_poss)
        # mappings.append(mapping)
    exp.positions = new_poss_list
    exp.nx_graph = new_graph_list
    #     return(corresp_list,new_poss_list,mappings)

    all_nodes = set()
    for k,graph in enumerate(new_graph_list):
        print(k)
        all_nodes = all_nodes.union(set(graph.nodes))
    all_node_labels = sorted(all_nodes)
    dico = {node: j for j,node in enumerate(all_node_labels)}
    reduced_label_graph_list = []
    mapping_final = lambda node: dico[node] if node in dico.keys() else -1
    reduced_poss_list = []
    for t in range(exp.ts):
        print("relabel2 ", t)

        reduced_label_graph = nx.relabel_nodes(new_graph_list[t], mapping_final)
        reduced_label_graph_list.append(reduced_label_graph)
        reduced_poss = relabel_pos(new_poss_list[t], mapping_final)
        reduced_poss_list.append(reduced_poss)
    exp.nx_graph = reduced_label_graph_list
    exp.positions = reduced_poss_list
    exp.labeled = True
    return (reduced_label_graph_list, reduced_poss_list)

def create_labeled_graph_local(exp):
    all_nodes = set()

    print("create labeled graph")
    corresp_list = []
    for t in range(exp.ts - 1):
        print(t, "ae")
        tp1 = t + 1
        corresp_path = get_corresp_path(exp, t, tp1)
        assert os.path.exists(corresp_path), "Not all corresp exist"
        with open(corresp_path) as json_file:
            corresp = json.load(json_file)
        corresp_list.append({int(corresp[key]): int(key) for key in corresp.keys()})
    for t in range(exp.ts):
        exp2 = Experiment(exp.directory)

        exp2.load(exp.folders.iloc[t:t + 1], suffix="_width")
        def mapping(node, t=t):
            return recursive_mapping(corresp_list, t, node)

        new_graph = nx.relabel_nodes(exp2.nx_graph[0], mapping, copy=True)
        new_poss = relabel_pos(exp2.positions[0], mapping)
        exp2.positions[0] = new_poss
        exp2.nx_graph[0] = new_graph
        exp2.save_graphs(suffix="_labeled")

        all_nodes = all_nodes.union(set(new_graph.nodes))
    all_node_labels = sorted(all_nodes)
    dico = {node: j for j,node in enumerate(all_node_labels)}
    mapping_final = lambda node: dico[node] if node in dico.keys() else -1
    for t in range(exp.ts):
        exp2 = Experiment(exp.directory)

        exp2.load(exp.folders.iloc[t:t + 1], suffix="_labeled")
        new_graph = nx.relabel_nodes(exp2.nx_graph[0], mapping_final, copy=True)
        new_poss = relabel_pos(exp2.positions[0], mapping_final)
        exp2.positions[0] = new_poss
        exp2.nx_graph[0] = new_graph
        exp2.save_graphs(suffix="_labeled")
    exp.labeled = True

def recursive_mapping(corresp_list, t, node):
    if t == 0:
        return f"{t}_{node}"
    else:
        corresp = corresp_list[t - 1]
        if node not in corresp.keys():
            return f"{t}_{node}"
        else:
            return recursive_mapping(corresp_list, t - 1, corresp[node])


def reconnect_degree_2(nx_graph, pos, has_width=True):
    degree_2_nodes = [node for node in nx_graph.nodes if nx_graph.degree(node) == 2]
    while len(degree_2_nodes) > 0:
        node = degree_2_nodes.pop()
        neighbours = list(nx_graph.neighbors(node))
        right_n = neighbours[0]
        left_n = neighbours[1]
        right_edge = nx_graph.get_edge_data(node, right_n)["pixel_list"]
        left_edge = nx_graph.get_edge_data(node, left_n)["pixel_list"]
        if has_width:
            right_edge_width = nx_graph.get_edge_data(node, right_n)["width"]
            left_edge_width = nx_graph.get_edge_data(node, left_n)["width"]
        else:
            # Maybe change to Nan if it doesnt break the rest
            right_edge_width = 40
            left_edge_width = 40
        if np.any(right_edge[0] != pos[node]):
            right_edge = list(reversed(right_edge))
        if np.any(left_edge[-1] != pos[node]):
            left_edge = list(reversed(left_edge))
        pixel_list = left_edge + right_edge[1:]
        width_new = (
            right_edge_width * len(right_edge) + left_edge_width * len(left_edge)
        ) / (len(right_edge) + len(left_edge))
        info = {"weight": len(pixel_list), "pixel_list": pixel_list, "width": width_new}
        if right_n != left_n:
            connection_data = nx_graph.get_edge_data(right_n, left_n)
            if connection_data is None or connection_data["weight"] >= info["weight"]:
                if not connection_data is None:
                    nx_graph.remove_edge(right_n, left_n)
                nx_graph.add_edges_from([(right_n, left_n, info)])
        nx_graph.remove_node(node)
        degree_2_nodes = [node for node in nx_graph.nodes if nx_graph.degree(node) == 2]
    degree_0_nodes = [node for node in nx_graph.nodes if nx_graph.degree(node) == 2]
    for node in degree_0_nodes:
        nx_graph.remove_node(node)


def node_dist(
    node1,
    node2,
    nx_graph_tm1,
    nx_graph_t,
    pos_tm1,
    pos_t,
    tolerance,
    show=False,
):
    #!!! assumed shape == 3000,4096
    sparse_cross1 = sparse.dok_matrix((4 * tolerance, 4 * tolerance), dtype=bool)
    sparse_cross2 = sparse.dok_matrix((4 * tolerance, 4 * tolerance), dtype=bool)
    for edge in nx_graph_tm1.edges(node1):
        list_pixel = nx_graph_tm1.get_edge_data(*edge)["pixel_list"]
        if (pos_tm1[node1] != list_pixel[0]).any():
            list_pixel = list(reversed(list_pixel))
        #         print(list_pixel[0],pos_tm1[node1],list_pixel[-1])
        for pixel in list_pixel[:20]:
            sparse_cross1[
                np.array(pixel) - np.array(pos_tm1[node1]) + np.array((50, 50))
            ] = 1
    for edge in nx_graph_t.edges(node2):
        list_pixel = nx_graph_t.get_edge_data(*edge)["pixel_list"]
        if (pos_t[node2] != list_pixel[0]).any():
            list_pixel = list(reversed(list_pixel))
        #         print(list_pixel[0],pos_t[node2],list_pixel[-1])
        for pixel in list_pixel[:20]:
            #             if np.any(np.array(pixel)-np.array(pos_tm1[node1])+np.array((50,50))>=100):
            #                 print(list_pixel[0],pos_t[node2],list_pixel[-1])
            sparse_cross2[
                np.array(pixel) - np.array(pos_tm1[node1]) + np.array((50, 50))
            ] = 1
    kernel = np.ones((3, 3), np.uint8)
    dilation1 = cv2.dilate(
        sparse_cross1.todense().astype(np.uint8), kernel, iterations=3
    )
    dilation2 = cv2.dilate(
        sparse_cross2.todense().astype(np.uint8), kernel, iterations=3
    )
    if show:
        plt.imshow(dilation1)
        plt.imshow(dilation2, alpha=0.5)
        plt.show()
    return np.linalg.norm(dilation1 - dilation2)


def remove_spurs(nx_g, pos, threshold=100):
    found = True
    while found:
        spurs = []
        found = False
        for edge in nx_g.edges:
            edge_data = nx_g.get_edge_data(*edge)
            if (nx_g.degree(edge[0]) == 1 or nx_g.degree(edge[1]) == 1) and edge_data[
                "weight"
            ] < threshold:
                spurs.append(edge)
                found = True
        for spur in spurs:
            nx_g.remove_edge(spur[0], spur[1])
        reconnect_degree_2(nx_g, pos, has_width=False)
    return (nx_g, pos)


def reduce_labels(nx_graph_list, pos_list):
    new_poss = [{} for i in range(len(nx_graph_list))]
    new_graphs = []
    all_node_labels = set()
    for nx_graph in nx_graph_list:
        all_node_labels = all_node_labels.union(set(nx_graph.nodes))
    all_node_labels = sorted(all_node_labels)
    dico = {node: index for index, node in enumerate(sorted(all_node_labels))}

    def mapping(node):
        return dico[node]

    for i, nx_graph in enumerate(nx_graph_list):
        for node in nx_graph.nodes:
            pos = pos_list[i][node]
            new_poss[i][mapping(node)] = pos
        new_graphs.append(nx.relabel_nodes(nx_graph, mapping, copy=True))
    return (new_graphs, new_poss)
