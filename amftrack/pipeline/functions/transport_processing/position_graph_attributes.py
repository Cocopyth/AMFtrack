import networkx as nx
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
import numpy as np
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
    distance_point_edge,
    plot_edge,
    plot_edge_cropped,
    find_nearest_edge,
    get_edge_from_node_labels,
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,
    find_neighboring_edges,
    reconstruct_image,
    reconstruct_skeletton_from_edges,
    reconstruct_skeletton_unicolor,
    plot_edge_color_value,
    reconstruct_image_from_general,
    plot_full,
    find_nearest_edge,
)


def find_edges(kymo_id, dictionary, exp, t):
    if int(kymo_id) in dictionary.keys():
        nodes_limit = dictionary[int(kymo_id)].split(",")
        nodes = nx.shortest_path(
            exp.nx_graph[t],
            source=int(nodes_limit[0]),
            target=int(nodes_limit[1]),
            weight="weight",
        )
        edges = [
            Edge(
                exp.get_node(nodes[i]),
                exp.get_node(nodes[i + 1]),
                exp,
            )
            for i in range(len(nodes) - 1)
        ]
        return edges
    else:
        print(kymo_id)
        return []


def find_tip(kymo_id, dictionary, exp):
    if int(kymo_id) in dictionary.keys():
        tip = dictionary[int(kymo_id)].split(",")[0]
        return Node(tip, exp)
    else:
        return None


def is_anastomosed(kymo_id, anastomosed_dict):
    if int(kymo_id) in anastomosed_dict.keys():
        return anastomosed_dict[int(kymo_id)]
    else:
        return False


def get_random_betweenness(exp, pos, t, edges):
    edge = find_nearest_edge(pos, exp, t, edges)
    try:
        result = edge.current_flow_betweeness(t)
    except:
        result = None
    return result


def get_betweenness(exp, pos, t, edges):
    edge = find_nearest_edge(pos, exp, t, edges)
    try:
        result = edge.betweeness(t)
    except:
        result = None
    return result


def find_closest_end(tip, edge, t):
    """finds which end of the edge is closer to the tip"""
    dist1 = nx.shortest_path_length(
        edge.exp.nx_graph[t],
        source=tip.label,
        target=edge.begin.label,
        weight="weight",
    )
    dist2 = nx.shortest_path_length(
        edge.exp.nx_graph[t],
        source=tip.label,
        target=edge.end.label,
        weight="weight",
    )
    if dist1 < dist2:
        return edge.begin
    else:
        return edge.end


def get_dist_tip_loc(exp, pos, t, edges_list, tip):
    """Find the distance between the closest edge of the position
    pos and the tip along the edges of the
    graph of the experiment at timestep t"""
    edge = find_nearest_edge(pos, exp, t, edges_list) if len(edges_list) > 0 else None
    node = find_closest_end(tip, edge, t)
    if edge is not None:
        dist = np.linalg.norm(pos - node.pos(t))
        if tip is not None:
            dist += nx.shortest_path_length(
                exp.nx_graph[t],
                source=tip.label,
                target=node.label,
                weight="weight",
            )
        return dist
    else:
        return None


def get_num_nodes_tip_loc(exp, pos, t, edges_list, tip):
    """Find the number of nodes between the closest edge of the position
    pos and the tip along the edges of the
    graph of the experiment at timestep t"""
    edge = find_nearest_edge(pos, exp, t, edges_list) if len(edges_list) > 0 else None
    node = find_closest_end(tip, edge, t)
    if edge is not None:
        if tip is not None:
            nodes = nx.shortest_path(
                exp.nx_graph[t],
                source=tip.label,
                target=node.label,
                weight="weight",
            )
        return len(nodes)
    else:
        return None


def get_poss_edges_lists(table, exp, t):
    poss = [
        exp.timestep_to_general((5 * row["posy"], 5 * row["posx"]), t)
        for index, row in table.iterrows()
    ]
    edges_lists = [find_edges(row["kymo_id"]) for index, row in table.iterrows()]
    return (poss, edges_lists)


def get_tip_lists(table, exp, t):
    tip_lists = [find_tip(row["kymo_id"]) for index, row in table.iterrows()]
    return tip_lists


def get_betweenness_loc(table, exp, t=0):
    poss, edges_lists = get_poss_edges_lists(table, exp, t)
    table[f"betweenness"] = [
        get_betweenness(exp, pos, t, edges_lists[i])
        if len(edges_lists[i]) > 0
        else np.nan
        for i, pos in enumerate(poss)
    ]
    return table


def get_random_betweenness_loc(table, exp, t):
    poss, edges_lists = get_poss_edges_lists(table, exp, t)
    table[f"random_betweenness"] = [
        get_random_betweenness(exp, pos, t, edges_lists[i])
        if len(edges_lists[i]) > 0
        else np.nan
        for i, pos in enumerate(poss)
    ]
    return table


def get_dist_tip(table, exp, t):
    poss, edges_lists = get_poss_edges_lists(table, exp, t)
    tip_lists = get_tip_lists(table, exp, t)
    table[f"dist_tip"] = [
        get_dist_tip_loc(exp, pos, t, edges_lists[i], tip_lists[i])
        if len(edges_lists[i]) > 0
        else np.nan
        for i, pos in enumerate(poss)
    ]
    return table


def get_num_nodes_tip(table, exp, t):
    poss, edges_lists = get_poss_edges_lists(table, exp, t)
    tip_lists = get_tip_lists(table, exp, t)
    table[f"num_nodes_tip"] = [
        get_num_nodes_tip_loc(exp, pos, t, edges_lists[i], tip_lists[i])
        if len(edges_lists[i]) > 0
        else np.nan
        for i, pos in enumerate(poss)
    ]
    return table


def get_betweenness_max_min(exp, t):
    edges = get_all_edges(exp, t)
    betweenness = [edge.betweeness(t) for edge in edges if edge.betweeness(t) > 0]
    return (np.min(betweenness), np.max(betweenness))


def get_random_betweenness_max_min(exp, t=0):
    edges = get_all_edges(exp, t)
    betweenness = [
        edge.current_flow_betweeness(t)
        for edge in edges
        if edge.current_flow_betweeness(t) > 0
    ]
    return (np.min(betweenness), np.max(betweenness))
