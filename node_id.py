import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy import sparse
import cv2
from pymatreader import read_mat
from util import get_path
from plotutil import show_im, overlap, show_im_rgb
from extract_graph import (
    generate_graph_tab_from_skeleton,
    generate_nx_graph_from_skeleton,
    generate_skeleton,
)
import networkx as nx
from copy import deepcopy
from sparse_util import dilate
from scipy.optimize import minimize
from time import time


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


def first_identification(nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, tolerance):
    corresp = {}
    ambiguous = set()
    to_remove = set()
    degree_3sup_nodes_tm1 = [
        node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) >= 3
    ]
    degree_3sup_nodes_t = [
        node for node in nx_graph_t.nodes if nx_graph_t.degree(node) >= 3
    ]
    Stm1 = sparse.csr_matrix((30000, 60000))
    St = sparse.csr_matrix((30000, 60000))
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


def relabel_nodes(corresp, nx_graph_t, pos_t):
    invert_corresp = {}
    new_pos = {}
    maxi = max(nx_graph_t.nodes) + 1
    for key in corresp.keys():
        invert_corresp[corresp[key]] = key

    def mapping(node):
        if node in invert_corresp.keys():
            return invert_corresp[node]
        else:
            return maxi + node

    for node in nx_graph_t.nodes:
        pos = pos_t[node]
        if node in invert_corresp.keys():
            new_pos[invert_corresp[node]] = pos
        else:
            new_pos[maxi + node] = pos
    new_graph = nx.relabel_nodes(nx_graph_t, mapping)
    return (new_pos, new_graph)


def relabel_nodes_downstream(corresp, nx_graph_list, pos_list):
    invert_corresp = {}
    new_poss = [{} for i in range(len(nx_graph_list))]
    new_graphs = []
    all_nodes = set()
    for nx_graph in nx_graph_list:
        all_nodes = all_nodes.union(set(nx_graph.nodes))
    all_nodes = all_nodes.union(set(corresp.keys()))
    all_nodes = all_nodes.union(set(corresp.values()))
    maxi = max(all_nodes) + 1
    for key in corresp.keys():
        invert_corresp[corresp[key]] = key

    def mapping(node):
        if node in invert_corresp.keys():
            return invert_corresp[node]
        else:
            return maxi + node

    for i, nx_graph in enumerate(nx_graph_list):
        for node in nx_graph.nodes:
            pos = pos_list[i][node]
            new_poss[i][mapping(node)] = pos
        new_graphs.append(nx.relabel_nodes(nx_graph, mapping, copy=True))
    return (new_graphs, new_poss)


def reduce_labels(nx_graph_list, pos_list):
    new_poss = [{} for i in range(len(nx_graph_list))]
    new_graphs = []
    all_node_labels = set()
    node = [node for node in nx_graph_list[0].nodes if node in nx_graph_list[1]][0]
    for nx_graph in nx_graph_list:
        all_node_labels = all_node_labels.union(set(nx_graph.nodes))
    all_node_labels = sorted(all_node_labels)

    def mapping(node):
        return all_node_labels.index(node)

    for i, nx_graph in enumerate(nx_graph_list):
        for node in nx_graph.nodes:
            pos = pos_list[i][node]
            new_poss[i][mapping(node)] = pos
        new_graphs.append(nx.relabel_nodes(nx_graph, mapping, copy=True))
    #     node=choice([node for node in new_graphs[0].nodes if new_graphs[0].degree(node)==3])
    #     for i,nx_graph in enumerate(nx_graph_list):
    #         if node in new_poss[i].keys():
    #             print(node,i,new_poss[i][node])
    return (new_graphs, new_poss)


def reconnect_degree_2(nx_graph, pos):
    degree_2_nodes = [node for node in nx_graph.nodes if nx_graph.degree(node) == 2]
    while len(degree_2_nodes) > 0:
        node = degree_2_nodes.pop()
        neighbours = list(nx_graph.neighbors(node))
        right_n = neighbours[0]
        left_n = neighbours[1]
        right_edge = nx_graph.get_edge_data(node, right_n)["pixel_list"]
        left_edge = nx_graph.get_edge_data(node, left_n)["pixel_list"]
        if np.any(right_edge[0] != pos[node]):
            right_edge = list(reversed(right_edge))
        if np.any(left_edge[-1] != pos[node]):
            left_edge = list(reversed(left_edge))
        pixel_list = left_edge + right_edge[1:]
        info = {"weight": len(pixel_list), "pixel_list": pixel_list}
        if right_n != left_n:
            connection_data = nx_graph.get_edge_data(right_n, left_n)
            if connection_data is None or connection_data["weight"] >= info["weight"]:
                if not connection_data is None:
                    nx_graph.remove_edge(right_n, left_n)
                nx_graph.add_edges_from([(right_n, left_n, info)])
        nx_graph.remove_node(node)
        degree_2_nodes = [node for node in nx_graph.nodes if nx_graph.degree(node) == 2]


def clean_nodes(nx_graph, to_remove, pos):
    nx_graph = [deepcopy(nx_graph)]  # could be removed to speed up
    is_hair = True
    i = 0
    while is_hair:
        is_hair = False
        to_remove_possibly = list(to_remove)
        for node in to_remove_possibly:
            neighbours = nx_graph.neighbors(node)
            candidate_to_remove = []
            weight_candidate = []
            for neighbour in neighbours:
                if nx_graph.degree(neighbour) == 1:
                    is_hair = True
                    candidate_to_remove.append(neighbour)
                    weight_candidate.append(
                        len(nx_graph.get_edge_data(node, neighbour)["pixel_list"])
                    )
            if len(candidate_to_remove) > 0:
                node_to_remove = candidate_to_remove[np.argmin(weight_candidate)]
                nx_graph.remove_node(node_to_remove)
                i += 1
                if nx_graph.degree(node) == 2:
                    to_remove.discard(node)
    print("number removed first phase", i)
    reconnect_degree_2(nx_graph, pos)  # could possibly be done faster
    for node in to_remove:
        if node in nx_graph:
            neighbours = list(nx_graph.neighbors(node))
            candidate_to_fuse = []
            weight_candidate = []
            for neighbour in neighbours:
                candidate_to_fuse.append(neighbour)
                weight_candidate.append(
                    len(nx_graph.get_edge_data(node, neighbour)["pixel_list"])
                )
            node_to_fuse = candidate_to_fuse[np.argmin(weight_candidate)]
            if nx_graph.degree(node) == 1 and node_to_fuse not in to_remove:
                print(pos[node])
                continue
            for neighbour in neighbours:
                right_n = node_to_fuse
                left_n = neighbour
                right_edge = nx_graph.get_edge_data(node, right_n)["pixel_list"]
                left_edge = nx_graph.get_edge_data(node, left_n)["pixel_list"]
                if np.any(right_edge[0] != pos[node]):
                    right_edge = list(reversed(right_edge))
                if np.any(left_edge[-1] != pos[node]):
                    left_edge = list(reversed(left_edge))
                pixel_list = left_edge + right_edge[1:]
                info = {"weight": len(pixel_list), "pixel_list": pixel_list}
                if right_n != left_n:
                    connection_data = nx_graph.get_edge_data(right_n, left_n)
                    if (
                        connection_data is None
                        or connection_data["weight"] >= info["weight"]
                    ):
                        if not connection_data is None:
                            nx_graph.remove_edge(right_n, left_n)
                        nx_graph.add_edges_from([(right_n, left_n, info)])
            nx_graph.remove_node(node)
            i += 1
    print("number removed second phase", i)
    reconnect_degree_2(nx_graph, pos)
    return nx_graph


def orient(pixel_list, root_pos):
    if np.all(root_pos == pixel_list[0]):
        return pixel_list
    else:
        return list(reversed(pixel_list))


def second_identification(
    nx_graph_tm1,
    nx_graph_t,
    pos_tm1,
    pos_t,
    length_id=150,
    downstream_graphs=[],
    downstream_pos=[],
    tolerance=50,
):
    reconnect_degree_2(nx_graph_t, pos_t)
    t = time()
    corresp, to_remove = first_identification(
        nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, tolerance
    )
    print("first_id", time() - t)
    t = time()
    #     nx_graph_tm1=clean_nodes(nx_graph_tm1,to_remove,pos_tm1)
    #     print("clean_node",time()-t)
    #     t=time()
    downstream_graphs = [nx_graph_t] + downstream_graphs
    downstream_pos = [pos_t] + downstream_pos
    new_graphs, new_poss = relabel_nodes_downstream(
        corresp, downstream_graphs, downstream_pos
    )
    print("relabel", time() - t)
    t = time()
    pos_t = new_poss[0]
    nx_graph_t = new_graphs[0]
    downstream_pos = new_poss
    downstream_graphs = new_graphs
    corresp_tips = {node: node for node in corresp.keys()}
    tips = [node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) == 1]
    ambiguous = set()
    Sedge = sparse.csr_matrix((30000, 60000))
    for edge in nx_graph_t.edges:
        pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
        pixela = pixel_list[0]
        pixelb = pixel_list[-1]
        Sedge[pixela[0], pixela[1]] = edge[0]
        Sedge[pixelb[0], pixelb[1]] = edge[1]
    for i, tip in enumerate(tips):
        #         print(i/len(tips))
        mini1 = np.inf
        posanchor = pos_tm1[tip]
        window = 1000
        potential_surrounding_t = Sedge[
            max(0, posanchor[0] - 2 * window) : posanchor[0] + 2 * window,
            max(0, posanchor[1] - 2 * window) : posanchor[1] + 2 * window,
        ]
        #         potential_surrounding_t=Sedge
        #         for edge in nx_graph_t.edges:
        #             pixel_list=nx_graph_t.get_edge_data(*edge)['pixel_list']
        #             if np.linalg.norm(np.array(pixel_list[0])-np.array(pos_tm1[tip]))<=5000:
        #                 distance=np.min(np.linalg.norm(np.array(pixel_list)-np.array(pos_tm1[tip]),axis=1))
        #                 if distance<mini1:
        #                     mini1=distance
        #                     right_edge1 = edge
        #         print('t1 re',right_edge)
        mini = np.inf
        for node_root in potential_surrounding_t.data:
            for edge in nx_graph_t.edges(int(node_root)):
                pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
                if (
                    np.linalg.norm(np.array(pixel_list[0]) - np.array(pos_tm1[tip]))
                    <= 5000
                ):
                    distance = np.min(
                        np.linalg.norm(
                            np.array(pixel_list) - np.array(pos_tm1[tip]), axis=1
                        )
                    )
                    if distance < mini:
                        mini = distance
                        right_edge = edge
        #         print('t2 re',right_edge)
        #         if right_edge!=right_edge1:
        #             print('alaba',right_edge,right_edge1)
        #             print('len(surrounding)',len(potential_surrounding_t.data))
        if mini == np.inf:
            print(f"didnt find a tip to match tip in pos {posanchor}")
            continue
        origin = np.array(
            orient(
                nx_graph_tm1.get_edge_data(*list(nx_graph_tm1.edges(tip))[0])[
                    "pixel_list"
                ],
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
        i = 0
        loop = []
        while (
            nx_graph_t.degree(current_node) != 1
            and not current_node in nx_graph_tm1.nodes
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
            if i >= 100:
                print(mini / (2 * np.pi) * 360, pos_t[next_node])
                if next_node in loop:
                    break
                else:
                    loop.append(next_node)
            if len(candidate_vectors) < 2:
                print(
                    "candidate_vectors < 2",
                    nx_graph_t.degree(current_node),
                    pos_t[current_node],
                    [node for node in nx_graph_t.nodes if nx_graph_t.degree(node) == 2],
                )
            competitor = np.arccos(np.dot(candidate_vectors[0], -candidate_vectors[1]))
            if mini < competitor:
                current_node, last_node = next_node, current_node
            else:
                break
        if current_node in nx_graph_tm1.nodes:
            if last_node not in nx_graph_tm1.nodes:
                if last_node in corresp_tips.values():
                    ambiguous.add(tip)
                corresp_tips[tip] = last_node
        else:
            if current_node in corresp_tips.values():
                ambiguous.add(tip)
            corresp_tips[tip] = current_node
    print("tip_id", time() - t)
    t = time()
    while len(ambiguous) > 0:
        node = ambiguous.pop()
        identifier = corresp_tips[node]
        candidates = [
            nod for nod in corresp_tips.keys() if corresp_tips[nod] == identifier
        ]
        mini = np.inf
        for candidate in candidates:
            distance = np.linalg.norm(pos_tm1[candidate] - pos_t[identifier])
            #             print(identifier,distance)
            if distance < mini:
                right_candidate = candidate
                mini = distance
        for candidate in candidates:
            if candidate != right_candidate:
                corresp_tips.pop(candidate)
                ambiguous.discard(candidate)
    new_graphs, new_poss = relabel_nodes_downstream(
        corresp_tips, downstream_graphs, downstream_pos
    )
    downstream_pos = new_poss
    downstream_graphs = new_graphs
    #     print("second relabeling")
    #     print(len(nx_graph_tm1.nodes),len(new_graphs[0].nodes))
    new_graphs, new_poss = reduce_labels(
        [nx_graph_tm1] + downstream_graphs, [pos_tm1] + downstream_pos
    )
    #     print("third relabeling")
    #     print(len(new_graphs[0].nodes),len(new_graphs[1].nodes))
    return (new_graphs, new_poss)

def second_identification2(
    nx_graph_tm1,
    nx_graph_t,
    pos_tm1,
    pos_t,
    length_id=150,
    downstream_graphs=[],
    downstream_pos=[],
    tolerance=50,
):
    reconnect_degree_2(nx_graph_t, pos_t)
    t = time()
    corresp, to_remove = first_identification(
        nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, tolerance
    )
    print("first_id", time() - t)
    t = time()
    #     nx_graph_tm1=clean_nodes(nx_graph_tm1,to_remove,pos_tm1)
    #     print("clean_node",time()-t)
    #     t=time()
    downstream_graphs = [nx_graph_t] + downstream_graphs
    downstream_pos = [pos_t] + downstream_pos
    new_graphs, new_poss = relabel_nodes_downstream(
        corresp, downstream_graphs, downstream_pos
    )
    print("relabel", time() - t)
    t = time()
    pos_t = new_poss[0]
    nx_graph_t = new_graphs[0]
    downstream_pos = new_poss
    downstream_graphs = new_graphs
    corresp_tips = {node: node for node in corresp.keys()}
    tips = [node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) == 1]
    ambiguous = set()
    Sedge = sparse.csr_matrix((30000, 60000))
    for edge in nx_graph_t.edges:
        pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
        pixela = pixel_list[0]
        pixelb = pixel_list[-1]
        Sedge[pixela[0], pixela[1]] = edge[0]
        Sedge[pixelb[0], pixelb[1]] = edge[1]
    for i, tip in enumerate(tips):
        #         print(i/len(tips))
        mini1 = np.inf
        posanchor = pos_tm1[tip]
        window = 1000
        potential_surrounding_t = Sedge[
            max(0, posanchor[0] - 2 * window) : posanchor[0] + 2 * window,
            max(0, posanchor[1] - 2 * window) : posanchor[1] + 2 * window,
        ]
        #         potential_surrounding_t=Sedge
        #         for edge in nx_graph_t.edges:
        #             pixel_list=nx_graph_t.get_edge_data(*edge)['pixel_list']
        #             if np.linalg.norm(np.array(pixel_list[0])-np.array(pos_tm1[tip]))<=5000:
        #                 distance=np.min(np.linalg.norm(np.array(pixel_list)-np.array(pos_tm1[tip]),axis=1))
        #                 if distance<mini1:
        #                     mini1=distance
        #                     right_edge1 = edge
        #         print('t1 re',right_edge)
        mini = np.inf
        for node_root in potential_surrounding_t.data:
            for edge in nx_graph_t.edges(int(node_root)):
                pixel_list = nx_graph_t.get_edge_data(*edge)["pixel_list"]
                if (
                    np.linalg.norm(np.array(pixel_list[0]) - np.array(pos_tm1[tip]))
                    <= 5000
                ):
                    distance = np.min(
                        np.linalg.norm(
                            np.array(pixel_list) - np.array(pos_tm1[tip]), axis=1
                        )
                    )
                    if distance < mini:
                        mini = distance
                        right_edge = edge
        #         print('t2 re',right_edge)
        #         if right_edge!=right_edge1:
        #             print('alaba',right_edge,right_edge1)
        #             print('len(surrounding)',len(potential_surrounding_t.data))
        if mini == np.inf:
            print(f"didnt find a tip to match tip in pos {posanchor}")
            continue
        origin = np.array(
            orient(
                nx_graph_tm1.get_edge_data(*list(nx_graph_tm1.edges(tip))[0])[
                    "pixel_list"
                ],
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
        i = 0
        loop = []
        while (
            nx_graph_t.degree(current_node) != 1
            and not current_node in nx_graph_tm1.nodes
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
                    candidate_vectors.append((unit_vector_candidate,branch_candidate.shape[0]))
                    dot_product = np.dot(unit_vector_origin, unit_vector_candidate)
                    angle = np.arccos(dot_product)
                    score = angle/min(branch_candidate.shape[0],length_id)
                    if score < mini:
                        mini = score
                        next_node = neighbours_t
            #                     print('angle',dot_product,pos_t[last_node],pos_t[current_node],pos_t[neighbours_t],angle/(2*np.pi)*360)
            #!!!bug may happen here if two nodes are direct neighbours : I would nee to check further why it the case, optimal segmentation should avoid this issue.
            # This is especially a problem for degree 4 nodes. Maybe fuse nodes that are closer than 3 pixels.
            if i >= 100:
                print(mini / (2 * np.pi) * 360, pos_t[next_node])
                if next_node in loop:
                    break
                else:
                    loop.append(next_node)
            if len(candidate_vectors) < 2:
                print(
                    "candidate_vectors < 2",
                    nx_graph_t.degree(current_node),
                    pos_t[current_node],
                    [node for node in nx_graph_t.nodes if nx_graph_t.degree(node) == 2],
                )
            competitor_angle = np.arccos(np.dot(candidate_vectors[0][0], -candidate_vectors[1][0]))
            competitor_score = competitor_angle/min(candidate_vectors[0][1],candidate_vectors[1][1],length_id)
            if mini < competitor_score:
                current_node, last_node = next_node, current_node
            else:
                break
        if current_node in nx_graph_tm1.nodes:
            if last_node not in nx_graph_tm1.nodes:
                if last_node in corresp_tips.values():
                    ambiguous.add(tip)
                corresp_tips[tip] = last_node
        else:
            if current_node in corresp_tips.values():
                ambiguous.add(tip)
            corresp_tips[tip] = current_node
    print("tip_id", time() - t)
    t = time()
    while len(ambiguous) > 0:
        node = ambiguous.pop()
        identifier = corresp_tips[node]
        candidates = [
            nod for nod in corresp_tips.keys() if corresp_tips[nod] == identifier
        ]
        mini = np.inf
        for candidate in candidates:
            distance = np.linalg.norm(pos_tm1[candidate] - pos_t[identifier])
            #             print(identifier,distance)
            if distance < mini:
                right_candidate = candidate
                mini = distance
        for candidate in candidates:
            if candidate != right_candidate:
                corresp_tips.pop(candidate)
                ambiguous.discard(candidate)
    new_graphs, new_poss = relabel_nodes_downstream(
        corresp_tips, downstream_graphs, downstream_pos
    )
    downstream_pos = new_poss
    downstream_graphs = new_graphs
    #     print("second relabeling")
    #     print(len(nx_graph_tm1.nodes),len(new_graphs[0].nodes))
    new_graphs, new_poss = reduce_labels(
        [nx_graph_tm1] + downstream_graphs, [pos_tm1] + downstream_pos
    )
    #     print("third relabeling")
    #     print(len(new_graphs[0].nodes),len(new_graphs[1].nodes))
    return (new_graphs, new_poss)


def whole_movement_identification(
    nx_graph_tm1, nx_graph_t, pos_tm1, pos_t, length_id=50
):
    tips = [node for node in nx_graph_tm1.nodes if nx_graph_tm1.degree(node) == 1]
    tip_origin = {tip: [tip] for tip in tips}
    pixels_from_tip = {tip: [] for tip in tips}
    for number, tip in enumerate(tips):
        #         print('tip',pos_tm1[tip],tip)
        labeled = []
        if number % 100 == 0:
            print(number / len(tips))
        mini = np.inf
        for edge in nx_graph_t.edges:
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
        origin = np.array(
            orient(
                nx_graph_tm1.get_edge_data(*list(nx_graph_tm1.edges(tip))[0])[
                    "pixel_list"
                ],
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
        index_nearest_pixel = np.argmin(
            np.linalg.norm(branch - np.array(pos_tm1[tip]), axis=1)
        )
        candidate_vector = branch[-1] - branch[0]
        dot_product = np.dot(origin_vector, candidate_vector)
        #         if tip==5260:
        #             print(list(branch[index_nearest_pixel:]))
        #             print(list(branch[:index_nearest_pixel]))
        if dot_product >= 0:
            root = right_edge[0]
            next_node = right_edge[1]
            pixels_from_tip[tip].append(list(branch[index_nearest_pixel:]))
        else:
            root = right_edge[1]
            next_node = right_edge[0]
            pixels_from_tip[tip].append(
                list(reversed(list(branch[:index_nearest_pixel])))
            )
        # Could improve the candidate vector by chosing pixel around the forme tip but this identification should be rather unambiguous
        last_node = root
        current_node = next_node
        last_branch = np.array(
            orient(
                nx_graph_t.get_edge_data(root, next_node)["pixel_list"],
                pos_t[current_node],
            )
        )

        def label_node_recursive(last_node, current_node, corresp_label, labeled):
            if not current_node in labeled and not current_node in nx_graph_tm1.nodes:
                corresp_label[tip].append(current_node)
                labeled.append(current_node)
                pixel_list = nx_graph_t.get_edge_data(last_node, current_node)[
                    "pixel_list"
                ]
                pixels_from_tip[tip].append(pixel_list)
                if nx_graph_t.degree(current_node) >= 3:
                    for neighbour_t in nx_graph_t.neighbors(current_node):
                        if neighbour_t != last_node:
                            label_node_recursive(
                                current_node, neighbour_t, corresp_label, labeled
                            )

        label_node_recursive(last_node, current_node, tip_origin, labeled)
    return (tip_origin, pixels_from_tip)


def shift(skeleton1, skeleton2):
    skeleton1_dilated = dilate(dilate(skeleton1)).astype(np.float)
    skeleton2_dilated = dilate(dilate(skeleton2)).astype(np.float)

    def distance(shift):
        distance = 0
        #         print(shift)
        for pixel in skeleton1_dilated.keys():
            #             print(pixel[0]+shift[0],pixel[1]+shift[1])
            if (
                skeleton2_dilated.shape[0] > np.ceil(pixel[0] + shift[0]) >= 0
                and skeleton2_dilated.shape[1] > np.ceil(pixel[1] + shift[1]) >= 0
            ):
                shifted_pixel = (int(pixel[0] + shift[0]), int(pixel[1] + shift[1]))
                shifted_pixel_next = (
                    np.ceil(pixel[0] + shift[0]),
                    np.ceil(pixel[1] + shift[1]),
                )
                #                 print(shifted_pixel)
                prop = (
                    1
                    / 2
                    * (
                        pixel[0]
                        + shift[0]
                        - int(pixel[0] + shift[0])
                        + pixel[1]
                        + shift[1]
                        - int(pixel[1] + shift[1])
                    )
                )
                float_value = (1 - prop) * skeleton2_dilated[
                    shifted_pixel[0], shifted_pixel[1]
                ] + prop * (
                    skeleton2_dilated[shifted_pixel_next[0], shifted_pixel_next[1]]
                )
                distance += abs(skeleton1_dilated[pixel] - float_value)
            else:
                distance += 1
        #         for pixel in skeleton2_dilated.keys():
        #             if (skeleton2_dilated.shape[0]>pixel[0]-shift[0]>=0 and skeleton2_dilated.shape[1]>pixel[1]-shift[1]>=0):
        #                 shifted_pixel = (int(pixel[0]-shift[0]),int(pixel[1]-shift[1]))
        #                 distance+=abs(skeleton1_dilated[shifted_pixel[0],shifted_pixel[1]]^skeleton2_dilated[pixel])
        #             else:
        #                 distance+=1
        #         print(distance)
        return distance

    return minimize(
        distance,
        np.array([10, 10]),
        method="nelder-mead",
        options={"xatol": 1, "disp": True, "fatol": 0.1},
    )
