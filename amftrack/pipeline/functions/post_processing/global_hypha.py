from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_pixel_growth_and_new_children,
)
from amftrack.pipeline.functions.post_processing.util import (
    measure_length_um_edge,
    is_in_study_zone,
)
import numpy as np
from amftrack.notebooks.analysis.util import *

pixel_conversion_factor = 1.725
import networkx as nx


def get_width_f(hyph, args):
    t = hyph.end.ts()[-1]
    try:
        edges = hyph.get_nodes_within(t)[1]
        widths = np.array([edge.width(t) for edge in edges])
        lengths = np.array([measure_length_um_edge(edge, t) for edge in edges])
        av_width = np.sum(widths * lengths) / np.sum(lengths)
        return ("av_width_final", av_width)
    except nx.exception.NetworkXNoPath:
        return ("av_width_final", None)


def get_tot_length_C_f(hyph, args):
    t = hyph.end.ts()[-1]
    try:
        edges = hyph.get_nodes_within(t)[1]
        lengths = np.array([measure_length_um_edge(edge, t) for edge in edges])
        tot_length_C = np.sum(lengths)
        return ("tot_length_C", tot_length_C)
    except nx.exception.NetworkXNoPath:
        return ("tot_length_C", None)


def get_tot_growth_C_f(hyph, args):
    t0 = hyph.end.ts()[0]
    tf = hyph.end.ts()[-1]
    try:
        pixels, nodes = get_pixel_growth_and_new_children(hyph, t0, tf)
        tot_growth_C = np.sum([get_length_um(seg) for seg in pixels])
    except AssertionError:
        return ("tot_growth_C", 0)
    except nx.exception.NetworkXNoPath:
        return ("tot_growth_C", None)
    return ("tot_growth_C", tot_growth_C)


def get_tot_length_pp_f(hyph, args):
    t = hyph.end.ts()[-1]
    tip_pos = hyph.end.pos(t)
    root_pos = hyph.get_root(t).pos(t)
    tot_length_pp = np.linalg.norm(tip_pos - root_pos) * pixel_conversion_factor
    return ("tot_length_pp", tot_length_pp)


def get_tot_growth_pp_f(hyph, args):
    t0 = hyph.end.ts()[0]
    tf = hyph.end.ts()[-1]
    tip_pos_init = hyph.end.pos(t0)
    tip_pos_finish = hyph.end.pos(tf)
    tot_growth_pp = (
        np.linalg.norm(tip_pos_init - tip_pos_finish) * pixel_conversion_factor
    )
    return ("tot_growth_pp", tot_growth_pp)


def get_timestep_stop_growth(hyph, args):
    tf = hyph.end.ts()[-1]
    pos_end = hyph.end.pos(tf)
    thresh = 40
    for t in hyph.ts:
        if np.linalg.norm(hyph.end.pos(t) - pos_end) < thresh:
            return ("timestep_stop_growth", t)


def get_time_stop_growth(hyph, args):
    tf = get_timestep_stop_growth(hyph, args)[1]
    return ("time_stop_growth", get_time(hyph.experiment, 0, tf))


def get_time_init_growth(hyph, args):
    t0 = hyph.end.ts()[0]
    return ("time_init_growth", get_time(hyph.experiment, 0, t0))


def get_timestep_init_growth(hyph, args):
    t0 = hyph.end.ts()[0]
    return ("timestep_init_growth", t0)


def get_mean_speed_growth(hyph, args):
    tf = get_timestep_stop_growth(hyph, args)[1]
    t0 = hyph.end.ts()[0]
    try:
        pixels, nodes = get_pixel_growth_and_new_children(hyph, t0, tf)
        tot_growth_C = np.sum([get_length_um(seg) for seg in pixels])
        mean_speed = tot_growth_C / get_time(hyph.experiment, t0, tf)
        return ("mean_speed", mean_speed)
    except AssertionError:
        return ("mean_speed", 0)
    except nx.exception.NetworkXNoPath:
        return ("mean_speed", None)


def get_stop_track(hyph, args):
    return ("strop_track", hyph.ts[-1])


def gets_out_of_ROI(hyph, args):
    for t in hyph.end.ts():
        if not np.all(is_in_study_zone(hyph.end, t, 1000, 150)):
            return ("out_of_ROI", t)
    return ("out_of_ROI", None)


def get_timestep_anastomosis(hyph, args):
    for i, t in enumerate(hyph.end.ts()[:-1]):
        tp1 = hyph.end.ts()[i + 1]
        if (
            hyph.end.degree(t) == 1
            and hyph.end.degree(tp1) == 3
            and 1 not in [hyph.end.degree(k) for k in hyph.end.ts()[i + 1 :]]
        ):
            return ("timestep_anastomosis", t)
    return ("timestep_anastomosis", None)


def get_timestep_biological_stop_growth(hyph, args):
    for i, t in enumerate(hyph.end.ts()[:-1]):
        tp1 = hyph.end.ts()[i + 1]
        if (
            hyph.end.degree(t) == 1
            and hyph.end.degree(tp1) == 3
            and 1 not in [hyph.end.degree(t) for t in hyph.end.ts()[i + 1 :]]
        ):
            return ("timestep_biological_stop_growth", None)
    t = get_timestep_stop_growth(hyph, args)[1]
    if t != hyph.end.ts()[-1]:
        return ("timestep_biological_stop_growth", t)
    else:
        return ("timestep_biological_stop_growth", None)


def get_num_branch(hyph, args):
    t0 = hyph.ts[0]
    exp = hyph.experiment
    thresh = 1600
    junctions_found = [hyph.end.neighbours(t0)[0]]
    try:
        for t in hyph.end.ts()[1:]:
            G = exp.nx_graph[t]
            G = G.subgraph(nx.node_connected_component(G, hyph.end.label))
            if hyph.end.degree(t) == 1:
                nodes, edges = hyph.get_nodes_within(t)
                potentials = []
                nodes = [Node(node, exp) for node in nodes]
                try:
                    last_junction_index = nodes.index(junctions_found[-1])
                except:
                    last_junction_index = 0
                for node in nodes[last_junction_index + 1 : -1]:
                    dist = np.linalg.norm(node.pos(t) - hyph.end.pos(t))
                    # To avoid detecting two times the same  node with different labels
                    dists_junction_found = [np.inf] + [
                        np.linalg.norm(node.pos(t) - nodo.pos(t))
                        for nodo in junctions_found
                        if nodo.is_in(t)
                    ]
                    if (
                        dist < thresh
                        and min(dists_junction_found) > 40
                        and (node not in junctions_found)
                    ):
                        extra_hypha_neighbours = [
                            nodo for nodo in node.neighbours(t) if nodo not in nodes
                        ]
                        tips = [
                            nodo
                            for nodo in extra_hypha_neighbours
                            if nx.edge_connectivity(G, nodo.label, node.label) == 1
                        ]
                        if len(tips) == node.degree(t) - 2:
                            # print(t)
                            junctions_found.append(node)
        return ("num_branch", len(junctions_found) - 1)
    except nx.exception.NetworkXNoPath:
        return ("num_branch", None)
