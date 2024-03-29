import networkx as nx

from amftrack.pipeline.functions.image_processing.experiment_util import get_all_edges, get_all_nodes
from amftrack.pipeline.functions.post_processing.extract_study_zone import load_study_zone
from amftrack.pipeline.functions.post_processing.util import is_in_study_zone


def add_betweenness(exp,t):
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
        if is_in_study_zone(node, t, 1000, 150)[1] if node.degree(t)==1
    ]
    weights = {(edge.begin.label, edge.end.label): edge.length_um(t) for edge in edges}
    nx.set_edge_attributes(exp.nx_graph[t], weights, "length")
    weights = {(edge.begin.label, edge.end.label): 1 / edge.length_um(t) for edge in edges}
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