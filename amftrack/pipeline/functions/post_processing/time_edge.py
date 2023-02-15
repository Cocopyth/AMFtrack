import numpy as np
from amftrack.pipeline.functions.post_processing.util import (
    measure_length_um_edge,
    is_in_study_zone,
)
import networkx as nx

is_circle = False


def get_time_since_start(edge, t, args=None):
    exp = edge.experiment
    seconds = (exp.dates[t] - exp.dates[edge.ts()[0]]).total_seconds()
    return ("time_since_emergence", float(seconds / 3600))


def get_time_since_begin_exp(edge, t, args=None):
    exp = edge.experiment
    seconds = (exp.dates[t] - exp.dates[0]).total_seconds()
    return ("time_since_begin_exp", float(seconds / 3600))


def get_tot_length_C(edge, t, args=None):
    length = measure_length_um_edge(edge, t)
    return ("tot_length_C", float(length))


def get_tot_length_straight(edge, t, args=None):
    pos_end = edge.end.pos(t)
    pos_begin = edge.begin.pos(t)
    return ("tot_length_straight", float(np.linalg.norm(pos_begin - pos_end)))


def get_width_edge(edge, t, args=None):
    width = edge.width(t)
    return ("width_edge", float(width))


def get_pos_x(edge, t, args=None):
    pos_end = edge.end.pos(t)
    return ("pos_x", int(pos_end[0]))


def get_pos_y(edge, t, args=None):
    pos_end = edge.end.pos(t)
    return ("pos_y", int(pos_end[1]))


def get_in_ROI(edge, t, args=None):
    return (
        "in_ROI",
        str(
            is_in_study_zone(edge.end, t, 1000, 150, is_circle)
            or is_in_study_zone(edge.begin, t, 1000, 150, is_circle)
        ),
    )


def get_connected_component_id(edge, t, args=None):
    nx_graph = edge.experiment.nx_graph[t]
    S = [nx_graph.subgraph(c).copy() for c in nx.connected_components(nx_graph)]
    for j, subgraph in enumerate(S):
        nodes = subgraph.nodes
        if edge.begin.label in nodes and edge.end.label in nodes:
            return ("component_id", j)
