from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
from amftrack.pipeline.functions.post_processing.util import (
    measure_length_um_edge,
    is_in_study_zone,
)
import numpy as np
from scipy import spatial
from shapely.geometry import Polygon, shape
import networkx as nx
import scipy.io as sio
import os
from amftrack.pipeline.functions.image_processing.experiment_util import get_all_edges

is_circle = False


def get_is_out_study(exp, t, args=None):
    return ("out_study", int(t > exp.reach_out))


def get_length_study_zone(exp, t, args=None):
    """Return the length of the edges in the study zone.
    Parameters:
    ----------
    exp: Experiment
    t: int
    """
    length = 0
    for edge in exp.nx_graph[t].edges:
        edge_obj = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
        is_in_end = np.all(is_in_study_zone(edge_obj.end, t, 1000, 150, is_circle))
        is_in_begin = np.all(is_in_study_zone(edge_obj.begin, t, 1000, 150, is_circle))
        if is_in_end and is_in_begin:
            length += measure_length_um_edge(edge_obj, t)
    return ("tot_length_study", length)


def get_length_tot(exp, t, args=None):
    length = 0
    for edge in exp.nx_graph[t].edges:
        edge_obj = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
        length += measure_length_um_edge(edge_obj, t)
    return ("tot_length", length)


def get_tot_biovolume(exp, t, args=None):
    nodes = [node for node in exp.nodes if node.is_in(t)]
    edges = {edge for node in nodes for edge in node.edges(t)}
    tot_biovolume = np.sum(
        [
            np.pi
            * (edge.width(t) / 2) ** 2
            * np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t))
            * 1.725
            for edge in edges
        ]
    )
    return ("tot_biovolume", tot_biovolume)


def get_tot_biovolume_study(exp, t, args=None):
    nodes = [
        node
        for node in exp.nodes
        if node.is_in(t) and np.all(is_in_study_zone(node, t, 1000, 200, is_circle))
    ]
    edges = {edge for node in nodes for edge in node.edges(t)}
    tot_biovolume = np.sum(
        [
            np.pi
            * (edge.width(t) / 2) ** 2
            * np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t))
            * 1.725
            for edge in edges
        ]
    )
    return ("tot_biovolume_study", tot_biovolume)


# def get_length_in_ring_rough(exp, t, args=None):
#     length = 0
#     for edge in exp.nx_graph[t].edges:
#         edge_obj = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
#         is_in_end = np.all(is_in_study_zone(edge_obj.end, t, 1000, 150))
#         is_in_begin = np.all(is_in_study_zone(edge_obj.begin, t, 1000, 150))
#         if is_in_end and is_in_begin:
#             length += np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t)) * 1.725
#     return ("tot_length_study_rough", length)


def get_area(exp, t, args=None):
    nodes = np.array([node.pos(t) for node in exp.nodes if node.is_in(t)])
    if len(nodes) > 0:
        hull = spatial.ConvexHull(nodes)
        poly = Polygon([nodes[vertice] for vertice in hull.vertices])
        area = poly.area * 1.725**2 / (1000**2)
    else:
        area = 0
    return ("area", area)


def get_area_study_zone(exp, t, args=None):
    nodes = np.array(
        [
            node.pos(t)
            for node in exp.nodes
            if node.is_in(t) and np.all(is_in_study_zone(node, t, 1000, 150, is_circle))
        ]
    )
    if len(nodes) > 3:
        hull = spatial.ConvexHull(nodes)
        poly = Polygon([nodes[vertice] for vertice in hull.vertices])
        area = poly.area * 1.725**2 / (1000**2)
    else:
        area = 0
    return ("area_study", area)


def get_area_separate_connected_components(exp, t, args=None):
    nx_graph = exp.nx_graph[t]
    threshold = 0.1
    S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
    selected = [
        g for g in S if g.size(weight="weight") * len(g.nodes) / 10**6 >= threshold
    ]
    area = 0
    for g in selected:
        nodes = np.array(
            [
                node.pos(t)
                for node in exp.nodes
                if node.is_in(t)
                and np.all(is_in_study_zone(node, t, 1000, 150, is_circle))
                and (node.label in g.nodes)
            ]
        )
        if len(nodes) > 3:
            hull = spatial.ConvexHull(nodes)
            poly = Polygon([nodes[vertice] for vertice in hull.vertices])
            area += poly.area * 1.725**2 / (1000**2)
    return ("area_sep_comp", area)


def get_num_tips(exp, t, args=None):
    return (
        "num_tips",
        len([node for node in exp.nodes if node.is_in(t) and node.degree(t) == 1]),
    )


def get_num_tips_study_zone(exp, t, args=None):
    return (
        "num_tips_study",
        len(
            [
                node
                for node in exp.nodes
                if node.is_in(t)
                and node.degree(t) == 1
                and np.all(is_in_study_zone(node, t, 1000, 150, is_circle))
            ]
        ),
    )


def get_num_BAS_tips(exp, t, args=None):
    def h(edge, t):
        boolean = (
            edge.end.degree(t) == 1 or edge.begin.degree(t) == 1
        ) and edge.length_um(t) < 1000
        # boolean +=((edge.width(t)*edge.length_um(t))<3000)*edge.width(t)<7
        return boolean

    edges = get_all_edges(exp, t)
    edges = [
        edge
        for edge in edges
        if np.all(is_in_study_zone(edge.begin, t, 1000, 150, is_circle))
    ]
    edges = [
        edge
        for edge in edges
        if np.all(is_in_study_zone(edge.end, t, 1000, 150, is_circle))
    ]
    edge_tip = [edge for edge in edges if h(edge, t)]
    return ("num_tips_BAS_study", len(edge_tip))


def get_num_nodes(exp, t, args=None):
    return ("num_nodes", len([node for node in exp.nodes if node.is_in(t)]))


def get_num_nodes_study_zone(exp, t, args=None):
    return (
        "num_nodes_study",
        len(
            [
                node
                for node in exp.nodes
                if node.is_in(t)
                and node.degree(t) > 0
                and np.all(is_in_study_zone(node, t, 1000, 150, is_circle))
            ]
        ),
    )


def get_num_edges(exp, t, args=None):
    return (
        "num_edges_study",
        np.sum(
            [
                node.degree(t)
                for node in exp.nodes
                if node.is_in(t)
                and node.degree(t) > 0
                and np.all(is_in_study_zone(node, t, 1000, 150, is_circle))
            ]
        )
        / 2,
    )


def get_length(exp, t, args=None):
    length = 0
    for edge in exp.nx_graph[t].edges:
        edge_obj = Edge(Node(edge[0], exp), Node(edge[1], exp), exp)
        length += measure_length_um_edge(edge_obj, t)
    return ("tot_length", length)


def get_num_trunks(exp, t, args=None):
    return ("num_trunks", int(exp.num_trunk))


def get_num_spores(exp, t, args=None):
    table = exp.folders
    directory = exp.directory
    folder = table["folder"].iloc[t]
    path_file = os.path.join(directory, folder, "Analysis", "spores.mat")
    if os.path.exists(path_file):
        spore_data = sio.loadmat(path_file)["spores"]
        num_spore = len(spore_data)
        return ("num_spores", num_spore)
    else:
        return ("num_spores", None)


def get_spore_volume(exp, t, args=None):
    table = exp.folders
    directory = exp.directory
    folder = table["folder"].iloc[t]
    path_file = os.path.join(directory, folder, "Analysis", "spores.mat")
    if os.path.exists(path_file):
        spore_data = sio.loadmat(path_file)["spores"]
        num_spore = len(spore_data)
        if num_spore > 0:
            spore_volume = np.sum(4 / 3 * np.pi * (spore_data[:, 2] * 1.725) ** 3)
            return ("spore_volume", spore_volume)
        else:
            return ("spore_volume", 0)
    else:
        return ("spore_volume", None)


def get_mean_edge_straight(exp, t, args=None):
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edges = [
        Edge(Node(edge_coord[0], exp), Node(edge_coord[1], exp), exp)
        for edge_coord in list(G.edges)
    ]
    straightnesses = []
    for edge in edges:
        is_in_end = np.all(is_in_study_zone(edge.end, t, 1000, 150, is_circle))
        is_in_begin = np.all(is_in_study_zone(edge.begin, t, 1000, 150, is_circle))
        if is_in_end and is_in_begin:
            length = measure_length_um_edge(edge, t)
            straight_distance = (
                np.linalg.norm(edge.end.pos(t) - edge.begin.pos(t)) * 1.725
            )
            if straight_distance > 40:
                straightnesses.append(straight_distance / length)
    if len(straightnesses) > 0:
        return ("mean_straightness", np.mean(straightnesses))
    else:
        return ("mean_straightness", None)


# def get_L_RH(exp, t, args=None):
#     plate = exp.folders["Plate"].unique()[0]
#     time_plate_info, global_hypha_info, time_hypha_info = get_data_tables()
#     table = global_hypha_info.loc[global_hypha_info["Plate"] == plate].copy()
#     table["log_length"] = np.log10((table["tot_length_C"] + 1).astype(float))
#     table["is_rh"] = (table["log_length"] >= 3.36).astype(int)
#     table = table.set_index("hypha")
#     hyphaes = table.loc[
#         (table["strop_track"] >= t)
#         & (table["timestep_init_growth"] <= t)
#         & ((table["out_of_ROI"].isnull()) | (table["out_of_ROI"] > t))
#     ].index
#     rh = hyphaes.loc[(hyphaes["is_rh"] == 1)].index
#     select_time = time_hypha_info.loc[time_hypha_info["Plate"] == plate]
#     L_rh = np.sum(
#         select_time.loc[(select_time["end"].isin(rh)) & (select_time["timestep"] == t)][
#             "tot_length_C"
#         ]
#     )
#     return ("L_rh", L_rh)
#
#
# def get_L_BAS(exp, t, args=None):
#     plate = exp.folders["Plate"].unique()[0]
#     time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(
#         redownload=True
#     )
#     table = global_hypha_info.loc[global_hypha_info["Plate"] == plate].copy()
#     table["log_length"] = np.log10((table["tot_length_C"] + 1).astype(float))
#     table["is_rh"] = (table["log_length"] >= 3.36).astype(int)
#     table = table.set_index("hypha")
#     hyphaes = table.loc[
#         (table["strop_track"] >= t)
#         & (table["timestep_init_growth"] <= t)
#         & ((table["out_of_ROI"].isnull()) | (table["out_of_ROI"] > t))
#     ]
#     bas = hyphaes.loc[(hyphaes["is_rh"] == 0)].index
#     select_time = time_hypha_info.loc[time_hypha_info["Plate"] == plate]
#     L_bas = np.sum(
#         select_time.loc[
#             (select_time["end"].isin(bas)) & (select_time["timestep"] == t)
#         ]["tot_length_C"]
#     )
#     return ("L_BAS", L_bas)
