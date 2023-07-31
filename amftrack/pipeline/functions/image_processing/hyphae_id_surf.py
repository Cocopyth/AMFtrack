import pandas as pd
import networkx as nx
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    prune_graph,
)
from amftrack.pipeline.functions.image_processing.node_id import (
    reconnect_degree_2,
    reduce_labels,
)

import scipy.io as sio
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
    Edge,
    Hyphae,
)
from collections import Counter


def width_based_cleaning(exp):
    thresh = 1
    thresh_up = 11
    to_remove = [1]
    while len(to_remove) > 0:
        to_remove = []
        to_keep = []
        for t in range(exp.ts):
            for edge in exp.nx_graph[t].edges:
                edge_obj = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
                if (edge_obj.width(t) <= thresh or edge_obj.width(t) > thresh_up) and (
                    edge_obj.begin.degree(t) == 1 or edge_obj.end.degree(t) == 1
                ):
                    to_remove.append(edge)
                    exp.nx_graph[t].remove_edge(edge[0], edge[1])
                else:
                    to_keep.append(edge)
        print(f"Removing {len(to_remove)} edges based on 0 width")
    to_remove_node = []
    for t in range(exp.ts):
        pos = exp.positions[t]
        nx_graph = exp.nx_graph[t]
        reconnect_degree_2(nx_graph, pos)
        nodes = list(nx_graph.nodes)
        for node in nodes:
            if nx_graph.degree(node) == 0:
                nx_graph.remove_node(node)
                to_remove_node.append(node)
    print(f"Removing {len(to_remove_node)} nodes based on degree 0 ")


def relabel_nodes_after_amb(corresp, nx_graph_list, pos_list):
    new_poss = [{} for i in range(len(nx_graph_list))]
    new_graphs = []
    all_nodes = set()
    for nx_graph in nx_graph_list:
        all_nodes = all_nodes.union(set(nx_graph.nodes))
    all_nodes = all_nodes.union(set(corresp.keys()))
    all_nodes = all_nodes.union(set(corresp.values()))
    maxi = max(all_nodes) + 1

    def mapping(node):
        if node in corresp.keys():
            return int(corresp[node])
        else:
            return node

    for i, nx_graph in enumerate(nx_graph_list):
        for node in nx_graph.nodes:
            pos = pos_list[i][node]
            new_poss[i][mapping(node)] = pos
        new_graphs.append(nx.relabel_nodes(nx_graph, mapping, copy=True))
    return (new_graphs, new_poss)


def get_mother(hyphaes):
    nodes_within = {hyphae.end: {} for hyphae in hyphaes}
    for i, hyphae in enumerate(hyphaes):
        if i % 500 == 0:
            print(i / len(hyphaes))
        mothers = []
        t0 = hyphae.ts[0]
        for hyph in hyphaes:
            if t0 in hyph.ts and hyph.end != hyphae.end:
                if t0 in nodes_within[hyph.end].keys():
                    nodes_within_hyph = nodes_within[hyph.end][t0]
                else:
                    nodes_within_hyph = hyph.get_nodes_within(t0)[0]
                    nodes_within[hyph.end][t0] = nodes_within_hyph
                if hyphae.root.label in nodes_within_hyph:
                    mothers.append(hyph)
        hyphae.mother = mothers
    counter = 0
    for hyphae in hyphaes:
        if len(hyphae.mother) >= 2:
            counter += 1
    print(f"{counter} hyphae have multiple mothers")


def get_pixel_growth_and_new_children(hyphae, t1, t2):
    assert t1 < t2, "t1 should be strictly inferior to t2"
    edges = hyphae.get_nodes_within(t2)[1]
    mini = np.inf
    if t1 not in hyphae.ts:
        raise Exception("t1 not in hyphae.ts")
    else:
        if len(edges) == 0:
            #             print(hyphae.root, hyphae.end)
            return ([], [])
        for i, edge in enumerate(edges):
            distance = np.min(
                np.linalg.norm(
                    hyphae.end.pos(t1) - np.array(edge.pixel_list(t2)), axis=1
                )
            )
            if distance < mini:
                index = i
                mini = distance
                last_edge = edge
                index_nearest_pixel = np.argmin(
                    np.linalg.norm(
                        hyphae.end.pos(t1) - np.array(edge.pixel_list(t2)), axis=1
                    )
                )
        # if mini > 50:
        # print("failure in finding closest edge")
        pixels = [last_edge.pixel_list(t2)[index_nearest_pixel:]]
        nodes = [-1, last_edge.end]
        for edge in edges[index + 1 :]:
            pixels.append(edge.pixel_list(t2))
            nodes.append(edge.end)
        return (pixels, nodes)


def save_hyphaes(exp, path="Data/"):
    column_names_hyphaes = ["end", "root", "ts", "mother"]
    column_names_growth_info = [
        "hyphae",
        "t",
        "tp1",
        "nodes_in_hyphae",
        "segment_of_growth_t_tp1",
        "node_list_t_tp1",
    ]
    hyphaes = pd.DataFrame(columns=column_names_hyphaes)
    growth_info = pd.DataFrame(columns=column_names_growth_info)
    for hyph in exp.hyphaes:
        new_line_hyphae = pd.DataFrame(
            {
                "end": [hyph.end.label],
                "root": [hyph.root.label],
                "ts": [hyph.ts],
                "mother": [-1 if len(hyph.mother) == 0 else hyph.mother[0].end.label],
            }
        )  # index 0 for
        # mothers need to be modified to resolve multi mother issue
        hyphaes = hyphaes.append(new_line_hyphae, ignore_index=True)
        for index in range(len(hyph.ts[:-1])):
            t = hyph.ts[index]
            tp1 = hyph.ts[index + 1]
            pixels, nodes = get_pixel_growth_and_new_children(hyph, t, tp1)
            if len(nodes) >= 1 and nodes[0] == -1:
                nodes = [-1] + [node.label for node in nodes[1:]]
            else:
                nodes = [node.label for node in nodes]
            new_line_growth = pd.DataFrame(
                {
                    "hyphae": [hyph.end.label],
                    "t": [t],
                    "tp1": [tp1],
                    "nodes_in_hyphae": [hyph.get_nodes_within(t)[0]],
                    "segment_of_growth_t_tp1": [pixels],
                    "node_list_t_tp1": [nodes],
                }
            )
            growth_info = growth_info.append(new_line_growth, ignore_index=True)
    hyphaes.to_csv(
        path + f"hyphaes_{exp.prince_pos}_{exp.dates[0]}_{exp.dates[-1]}.csv"
    )
    growth_info.to_csv(
        path + f"growth_info_{exp.prince_pos}_{exp.dates[0]}_{exp.dates[-1]}.csv"
    )
    sio.savemat(
        path + f"hyphaes_{exp.prince_pos}_{exp.dates[0]}_{exp.dates[-1]}.mat",
        {name: col.values for name, col in hyphaes.items()},
    )
    sio.savemat(
        path + f"growth_info_{exp.prince_pos}_{exp.dates[0]}_{exp.dates[-1]}.mat",
        {name: col.values for name, col in growth_info.items()},
    )
    return (hyphaes, growth_info)


def get_anastomosing_hyphae(exp):
    anastomosing_hyphae = []
    for hyph in exp.hyphaes:
        hyph.ts = hyph.end.ts()
        for i, t in enumerate(hyph.ts[:-1]):
            tp1 = hyph.ts[i + 1]
            if (
                hyph.end.degree(t) == 1
                and hyph.end.degree(tp1) == 3
                and 1 not in [hyph.end.degree(k) for k in hyph.ts[i + 1 :]]
            ):
                anastomosing_hyphae.append((hyph, t, tp1))
    return anastomosing_hyphae


def resolve_anastomosis_crossing_by_root(exp, lim_considered=1):
    hyphaes, problems = get_hyphae(exp, lim_considered=lim_considered)
    exp.hyphaes = hyphaes
    # print("getting anastomosing", len(hyphaes))
    anastomosing_hyphae = get_anastomosing_hyphae(exp)
    # print("relabeling")
    to_relabel = []
    corresp_hyph = {}
    i = 0
    poss_root_hypha = {}
    for hypha in exp.hyphaes:
        pos_root_hypha = np.mean([hypha.root.pos(t) for t in hypha.root.ts()], axis=0)
        poss_root_hypha[hypha] = pos_root_hypha
    for hyph, t0, tp1 in anastomosing_hyphae:
        # if i % 200 == 0:
        # print(i / len(anastomosing_hyphae))
        i += 1
        corresp_hyph[hyph.end.label] = []
        pos_root_hyph = np.mean([hyph.root.pos(t) for t in hyph.root.ts()], axis=0)

        for hypha in exp.hyphaes:
            pos_root_hypha = poss_root_hypha[hypha]
            if np.linalg.norm(pos_root_hyph - pos_root_hypha) <= 100:
                t00 = hypha.ts[0]
                if (
                    t00 in hyph.ts
                    and hypha.get_root(t00) == hyph.get_root(t00)
                    and t00 > t0
                    and hypha not in to_relabel
                ):
                    corresp_hyph[hyph.end.label].append(hypha)
                    to_relabel.append(hypha)
    node_after_cross = [hypha.end.label for hypha in to_relabel]
    node_false_anastomose = [
        hyph.end.label
        for hyph, _, _ in anastomosing_hyphae
        if len(corresp_hyph[hyph.end.label]) != 0
    ]
    considered_nodes = node_after_cross + node_false_anastomose
    all_nodes = set()
    for nx_graph in exp.nx_graph:
        all_nodes = all_nodes.union(set(nx_graph.nodes))
    maxi = max(all_nodes) + 1
    corresp_node = {}
    for end_label in corresp_hyph.keys():
        corresp_node[end_label] = [hypha.end.label for hypha in corresp_hyph[end_label]]
    for t in range(exp.ts):
        # print(t)
        nx_graph = exp.nx_graph[t]
        new_poss = {}
        poss = exp.positions[t]

        def mapping(node):
            if node in considered_nodes:
                if node in node_false_anastomose:
                    equ_hyph = corresp_hyph[node]
                    time_appearance_equ = min([hyph.ts[0] for hyph in equ_hyph])
                    if t >= time_appearance_equ:
                        return maxi + node
                    else:
                        return node
                else:
                    hyph_equs = [
                        nodo
                        for nodo in node_false_anastomose
                        if node in corresp_node[nodo]
                    ]
                    assert len(hyph_equs) == 1
                    equ_hyph = hyph_equs[0]
                    return equ_hyph
            else:
                return node

        for node in nx_graph.nodes:
            pos = poss[node]
            new_poss[mapping(node)] = pos
        new_graph = nx.relabel_nodes(nx_graph, mapping, copy=True)
        exp.nx_graph[t] = new_graph
        exp.positions[t] = new_poss
    new_graph_list, new_poss_list = reduce_labels(exp.nx_graph, exp.positions)
    exp.nx_graph, exp.positions = new_graph_list, new_poss_list
    labels = {node for g in exp.nx_graph for node in g}
    exp.nodes = []
    for label in labels:
        exp.nodes.append(Node(label, exp))
    # print("getting hyphae again")
    hyphaes, problems = get_hyphae(exp, lim_considered=lim_considered)
    exp.hyphaes = hyphaes


def get_hyphae(experiment, lim_considered=1,rh_only=True):
    tips = [
        node
        for node in experiment.nodes
        if node.degree(node.ts()[0]) == 1 and len(node.ts()) >= lim_considered
    ]
    if rh_only:
        tips = [
            node
            for node in tips
            if np.linalg.norm(node.pos(node.ts()[0]) - node.pos(node.ts()[-1])) >= 1500
        ]
    problems = []
    hyphaes = []
    for i, tip in enumerate(tips):
        # if i % 200 == 0:
        # print(i / len(tips))
        #         tip = choice(tips)
        hypha = Hyphae(tip)
        occurence_count, roots = get_occurence_count(tip)
        if (
            len(occurence_count.values()) >= 2
            and occurence_count.most_common(2)[0][0] != roots[0]
            and occurence_count.most_common(2)[1][1]
            / occurence_count.most_common(2)[0][1]
            >= 0.75
        ):
            problems.append(tip)
        else:
            hypha.root = occurence_count.most_common(2)[0][0]
            hypha.ts = hypha.end.ts()
            hyphaes.append(hypha)
    # print(
    #     f"Detected problems during hyphae detection, {len(problems)} hyphaes have inconsistent root over time"
    # )
    experiment.inconsistent_root = problems
    return (hyphaes, problems)


def get_occurence_count(tip):
    hypha = Hyphae(tip)
    roots = []
    for t in tip.ts():
        if tip.degree(t) == 1:
            root, edges, nodes = hypha.get_edges(t, 200)
            roots.append(root)
    occurence_count = Counter(roots)
    return occurence_count, roots
