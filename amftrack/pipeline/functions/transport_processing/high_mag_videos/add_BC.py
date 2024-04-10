import networkx as nx

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Node,
    Edge,
)
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_all_edges,
    get_all_nodes,
)
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.pipeline.functions.post_processing.util import (
    is_in_study_zone,
    is_in_ROI_node,
)
import numpy as np
import pandas as pd

from amftrack.pipeline.functions.transport_processing.high_mag_videos.register_videos import (
    add_attribute,
)

hyphae = pd.read_excel(
    "/home/cbisot/pycode/AMFtrack/amftrack/notebooks/transport/hyphae.xlsx"
)


def add_hyphal_attributes(edge_data_csv, edge, mapping, t, exp, unique_id):
    hyphae_plate = hyphae[hyphae["plate"] == unique_id]
    for index, row in hyphae_plate.iterrows():
        begin, end = row["begin"], row["end"]
        fun = lambda edge: get_abcisse(edge, begin, end, t, exp)
        add_attribute(edge_data_csv, edge, fun, f"abcisse_{begin}_{end}", mapping)


def find_lowest_nodes(nodes, t):
    positions = [node.pos(t) for node in nodes]

    # 1. Find the range of x positions
    min_x = min(positions, key=lambda p: p[1])[1]
    max_x = max(positions, key=lambda p: p[1])[1]

    # 2. Calculate the size of each part
    part_size = (max_x - min_x) / 10

    # 3. Group nodes by x range and select the node with the lowest y in each group
    selected_nodes = []
    for i in range(10):
        start_x = min_x + i * part_size
        end_x = start_x + part_size

        # Find nodes within the current x range
        nodes_in_range = [node for node in nodes if start_x <= node.pos(t)[1] < end_x]

        # 4. Select the node with the lowest y in the current range
        if nodes_in_range:
            lowest_y_node = max(nodes_in_range, key=lambda p: p.pos(t)[0])
            selected_nodes.append(lowest_y_node)
    return selected_nodes


Vmax = 1


def get_weight(node, t):
    weight = 0
    for edge in node.edges(t):
        weight += Vmax * np.pi * edge.width(t) / 2
    return weight


def get_shortest_path_edges(node2, shortest):
    exp = node2.experiment
    nodes = shortest[node2.label]
    edges = []
    for i in range(len(nodes) - 1):
        nodea = Node(nodes[i], exp)
        nodeb = Node(nodes[i + 1], exp)
        edges.append(Edge(nodea, nodeb, exp))
    return edges


def get_quantitative_BC_dic(exp, t, nodes_sink, nodes_source):
    edges = get_all_edges(exp, t)
    edge_flux = {edge: 0 for edge in edges}
    for node1 in nodes_sink:
        shortest = nx.single_source_dijkstra_path(
            exp.nx_graph[t], node1.label, weight="weight"
        )
        for node2 in nodes_source:
            w = get_weight(node2)
            path = get_shortest_path_edges(node2, shortest)
            for edge in path:
                edge_flux[edge] += w / len(nodes_sink)


def add_betweenness_QP(exp, t):
    exp.save_location = ""

    load_study_zone(exp)
    edges = get_all_edges(exp, t)
    nodes = get_all_nodes(exp, t)
    nodes_source = [node for node in nodes if is_in_ROI_node(node, t)]
    nodes_sink = [node for node in nodes if not is_in_ROI_node(node, t)]
    nodes_sink = find_lowest_nodes(nodes_sink, t)
    fluxes = get_quantitative_BC_dic(exp, t, nodes_sink, nodes_source)
    for edge in exp.nx_graph[t].edges:
        # if (
        #     edge not in final_current_flow_betweeness.keys()
        #     and (edge[1], edge[0]) not in final_current_flow_betweeness.keys()
        # ):
        #     final_current_flow_betweeness[edge] = 0
        if edge not in fluxes.keys() and (edge[1], edge[0]) not in fluxes.keys():
            fluxes[edge] = 0
    nx.set_edge_attributes(exp.nx_graph[t], fluxes, "betweenness_QP")


def add_betweenness(exp, t):
    exp.save_location = ""

    load_study_zone(exp)
    edges = get_all_edges(exp, t)
    nodes = get_all_nodes(exp, t)
    nodes_source = [
        node
        for node in nodes
        if not is_in_study_zone(node, t, 1000, 100)[1]
        and is_in_study_zone(node, t, 1000, 100)[0]
    ]
    nodes_source = nodes_source
    # nodes_sink = [
    #     node
    #     for node in nodes
    #     if is_in_study_zone(node, t, 1000, 150)[1]
    # ]
    nodes_sink = [
        node
        for node in nodes
        if is_in_study_zone(node, t, 1000, 150)[1]
        if node.degree(t) == 1
    ]
    weights = {(edge.begin.label, edge.end.label): edge.length_um(t) for edge in edges}
    nx.set_edge_attributes(exp.nx_graph[t], weights, "length")
    weights = {
        (edge.begin.label, edge.end.label): 1 / edge.length_um(t) for edge in edges
    }
    nx.set_edge_attributes(exp.nx_graph[t], weights, "1/length")
    t = 0
    G = exp.nx_graph[t]
    S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
    len_connected = [len(nx_graph.nodes) for nx_graph in S]
    final_current_flow_betweeness = {}
    final_betweeness = {}

    for g in S:
        source = [node.label for node in nodes_source if node.label in g]
        sink = [node.label for node in nodes_sink if node.label in g]
        # current_flow_betweeness = nx.edge_current_flow_betweenness_centrality_subset(
        #     g, source, sink, weight="1/length"
        # )
        # betweeness = nx.edge_current_flow_betweenness_centrality_subset(
        #     g, sink, source, weight="length"
        # )

        betweeness = nx.edge_betweenness_centrality_subset(
            g, source, sink, normalized=True, weight="length"
        )
        # for edge in current_flow_betweeness.keys():
        #     final_current_flow_betweeness[edge] = current_flow_betweeness[edge]
        for edge in betweeness.keys():
            final_betweeness[edge] = betweeness[edge]

    for edge in exp.nx_graph[t].edges:
        # if (
        #     edge not in final_current_flow_betweeness.keys()
        #     and (edge[1], edge[0]) not in final_current_flow_betweeness.keys()
        # ):
        #     final_current_flow_betweeness[edge] = 0
        if (
            edge not in final_betweeness.keys()
            and (edge[1], edge[0]) not in final_betweeness.keys()
        ):
            final_betweeness[edge] = 0
    nx.set_edge_attributes(exp.nx_graph[t], final_betweeness, "betweenness")


def get_abcisse(edge, begin, end, t, exp):
    begin = Node(begin, exp).get_pseudo_identity(t).label
    try:
        if Node(end, exp).is_in(t):
            nodes = nx.shortest_path(exp.nx_graph[t], begin, end, weight="weight")
            nodes = [Node(node, exp) for node in nodes]
            edges = [Edge(nodes[i], nodes[i + 1], exp) for i in range(len(nodes) - 1)]
            lengths = [edge.length_um(t) for edge in edges]
            abciss = np.cumsum(lengths)
            if edge in edges:
                i = edges.index(edge)
                return abciss[i]
    except nx.NetworkXNoPath:
        return -1
    return -1


def get_derivative(edge_d, t, fun):
    edges_begin = edge_d.begin.edges(t)
    edges_begin.remove(edge_d)
    edges_end = edge_d.end.edges(t)
    edges_end.remove(edge_d)
    weight_begin = np.sum([fun(edge) for edge in edges_begin])
    weight_end = np.sum([fun(edge) for edge in edges_end])
    return weight_end - weight_begin
