from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_pixel_growth_and_new_children,
)
import numpy as np
import networkx as nx
from amftrack.pipeline.functions.post_processing.util import (
    measure_length_um_edge,
    is_in_study_zone,
    measure_length_um,
)
from scipy import sparse

is_circle = False


def get_time_since_start(hypha, t, tp1, args=None):
    exp = hypha.experiment
    seconds = (exp.dates[t] - exp.dates[hypha.ts[0]]).total_seconds()
    return ("time_since_emergence", float(seconds / 3600))


def get_time_since_begin_exp(hypha, t, tp1, args=None):
    exp = hypha.experiment
    seconds = (exp.dates[t] - exp.dates[0]).total_seconds()
    return ("time_since_begin_exp", float(seconds / 3600))


def get_timedelta(hypha, t, tp1, args=None):
    exp = hypha.experiment
    seconds = (exp.dates[tp1] - exp.dates[t]).total_seconds()
    return ("timedelta", float(seconds / 3600))


def get_speed(hypha, t, tp1, args=None):
    try:
        pixels, nodes = get_pixel_growth_and_new_children(hypha, t, tp1)
        if len(pixels) > 0:
            speed = (
                np.sum([measure_length_um(seg) for seg in pixels])
                / get_timedelta(hypha, t, tp1, None)[1]
            )
            return ("speed", float(speed))
        else:
            return ("speed", -1)
    except nx.exception.NetworkXNoPath:
        print("not_connected", hypha.end.label, hypha.get_root(tp1).label)
        return ("speed", -1)


def find_angle(veca, vecb):
    begin = veca / np.linalg.norm(veca)
    end = vecb / np.linalg.norm(vecb)
    dot_product = min(np.dot(begin, end), 1)
    if begin[1] * end[0] - end[1] * begin[0] >= 0:  # determinant
        angle = -np.arccos(dot_product) / (2 * np.pi) * 360
    else:
        angle = np.arccos(dot_product) / (2 * np.pi) * 360
    return angle


def get_absolute_angle_growth(hypha, t, tp1, args=None):
    try:
        segs, nodes = get_pixel_growth_and_new_children(hypha, t, tp1)
        curve = [pixel for seg in segs for pixel in seg]
        curve_array = np.array(curve)
        vec1 = curve_array[-1] - curve_array[0]
        vec2 = [0, 1]
        angle = find_angle(vec1, vec2)
        return ("absolute_angle", float(angle))
    except nx.exception.NetworkXNoPath:
        print("not_connected", hypha.end.label, hypha.get_root(tp1).label)
        return ("absolute_angle", None)


def get_tot_length_C(hyph, t, tp1, args=None):
    try:

        edges = hyph.get_nodes_within(t)[1]
        if len(edges) > 0:

            lengths = np.array([measure_length_um_edge(edge, t) for edge in edges])
            tot_length_C = np.sum(lengths)
            return ("tot_length_C", float(tot_length_C))
        else:
            return ("tot_length_C", None)
    except nx.exception.NetworkXNoPath:
        return ("tot_length_C", None)


def get_timestep(hypha, t, tp1, args=None):
    return ("timestep", t)


def get_timestep_init(hypha, t, tp1, args=None):
    return ("timestep_init", hypha.ts[0])


def get_time_init(hypha, t, tp1, args=None):
    return ("time_init", get_timedelta(hypha, 0, hypha.ts[0], None)[1])


def get_degree(hypha, t, tp1, args=None):
    return ("degree", hypha.end.degree(t))


def get_width_tip_edge(hypha, t, tp1, args=None):
    try:
        edges = hypha.get_nodes_within(t)[1]
        if len(edges) > 0:
            edge = edges[-1]
            width = edge.width(t)
            return ("width_tip_edge", float(width))
        else:
            return ("width_tip_edge", None)
    except nx.exception.NetworkXNoPath:
        return ("width_tip_edge", None)


def get_width_root_edge(hypha, t, tp1, args=None):
    try:
        edges = hypha.get_nodes_within(t)[1]
        if len(edges) > 0:
            edge = edges[0]
            width = edge.width(t)
            return ("width_tip_edge", float(width))
        else:
            return ("width_tip_edge", None)
    except nx.exception.NetworkXNoPath:
        print("no_path")
        return ("width_root_edge", None)


def get_width_average(hypha, t, tp1, args=None):
    try:
        edges = hypha.get_nodes_within(t)[1]
        widths = np.array([edge.width(t) for edge in edges])
        lengths = np.array([measure_length_um_edge(edge, t) for edge in edges])
        if len(edges) > 0:
            av_width = np.sum(widths * lengths) / np.sum(lengths)
            return ("av_width", float(av_width))
        else:
            return ("av_width", None)
    except nx.exception.NetworkXNoPath:
        return ("av_width", None)


def get_has_reached_final_pos(hypha, t, tp1, args=None):
    tf = hypha.end.ts()[-1]
    pos_end = hypha.end.pos(tf)
    thresh = 40
    return (
        "has_reached_final_pos",
        str(np.linalg.norm(hypha.end.pos(t) - pos_end) < thresh),
    )


def get_distance_final_pos(hypha, t, tp1, args=None):
    tf = hypha.end.ts()[-1]
    pos_end = hypha.end.pos(tf)
    return ("distance_final_pos", float(np.linalg.norm(hypha.end.pos(t) - pos_end)))


def get_pos_x(hypha, t, tp1, args=None):
    pos_end = hypha.end.pos(t)
    return ("pos_x", int(pos_end[0]))


def get_pos_y(hypha, t, tp1, args=None):
    pos_end = hypha.end.pos(t)
    return ("pos_y", int(pos_end[1]))


# def get_local_density(hypha, t, tp1, args=[200]):
#     window = args[0]
#     exp = hypha.experiment
#     pos = hypha.end.pos(t)
#     x, y = pos[0], pos[1]
#     skeletons = [sparse.csr_matrix(skel) for skel in exp.skeletons]
#     skeleton = skeletons[t][x - window : x + window, y - window : y + window]
#     density = skeleton.count_nonzero() / (window**2 * 1.725) * 10**6
#     return (f"density_window{window}", density)


def get_in_ROI(hypha, t, tp1, args=None):
    return ("in_ROI", str(np.all(is_in_study_zone(hypha.end, t, 1000, 150, is_circle))))
