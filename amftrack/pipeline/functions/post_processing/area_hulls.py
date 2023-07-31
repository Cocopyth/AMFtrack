import os.path

from shapely.geometry import Polygon, Point
from scipy import spatial
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from amftrack.pipeline.functions.post_processing.area_hulls_util import *
from amftrack.pipeline.functions.post_processing.util import (
    is_in_study_zone,
)
from amftrack.notebooks.analysis.util import get_time, splitPolygon
from amftrack.util.sys import temp_path
import numpy as np
import pandas as pd
import networkx as nx
import pickle
import shapely.speedups
import scipy

shapely.speedups.enable()


from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)




# def get_speed_in_ring(hull1, hull2, t, exp, rh_only, op_id):
#     hyphae_ring = get_hyphae_in_ring(hull1, hull2, t, exp)
#     hyphae_ring = [hyph.end.label for hyph in hyphae_ring]
#     plate = exp.folders["Plate"].unique()[0]
#     time_plate_info, global_hypha_info, time_hypha_info = get_data_tables(
#         op_id, redownload=False
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
#     if rh_only:
#         selection_hypha = hyphaes.loc[(hyphaes["is_rh"] == 1)].index
#
#     else:
#         selection_hypha = hyphaes
#     nodes = get_nodes_in_ring(hull1, hull2, t, exp)
#     tips = [
#         node
#         for node in nodes
#         if node.degree(t) == 1 and node.is_in(t + 1) and len(node.ts()) > 2
#     ]
#     growing_tips = [
#         node.label
#         for node in tips
#         if np.linalg.norm(node.pos(t) - node.pos(node.ts()[-1])) >= 40
#     ]
#     select_time = time_hypha_info.loc[time_hypha_info["Plate"] == plate]
#     rh_ring = select_time.loc[
#         (select_time["end"].isin(selection_hypha))
#         & (select_time["end"].isin(hyphae_ring))
#         & (select_time["end"].isin(growing_tips))
#         & (select_time["timestep"] == t)
#         & (select_time["speed"] >= 50)
#     ]
#     speed_ring = np.mean(rh_ring["speed"])
#     return speed_ring


def get_density_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr-{incr}_index-{i}", length / area)
    else:
        return (f"ring_density_incr-{incr}_index-{i}", None)


def get_density_in_ring_new(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring_new(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr-{incr}_index-{i}-new", length / area)
    else:
        return (f"ring_density_incr-{incr}_index-{i}-new", None)


def get_density_in_ring_new_fixed(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_length_in_ring_new(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_density_incr_fixex-{incr}_index-{i}-new", length / area)
    else:
        return (f"ring_density_incr_fixex-{incr}_index-{i}-new", None)


# def get_density_in_ring_new_bootstrap(exp, t, args):
#     incr = args["incr"]
#     i = args["i"]
#     regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
#     if i + 2 <= len(regular_hulls):
#         hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
#         res = get_density_in_ring_bootstrap(hull1, hull2, t, exp, n_resamples=1)
#         if res is None:
#             return (f"ring_density_incr-{incr}_index-{i}-boot", None)
#         else:
#             return (
#                 f"ring_density_incr-{incr}_index-{i}-boot",
#                 np.median(res.bootstrap_distribution),
#             )
#     else:
#         return (f"ring_density_incr-{incr}_index-{i}_boot", None)


def get_std_density_in_ring_new_bootstrap(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        res = get_density_in_ring_bootstrap(hull1, hull2, t, exp, n_resamples=100)
        if res is None:
            return (f"ring_density_incr-{incr}_index-{i}-bootstd", None)
        else:
            return (f"ring_density_incr-{incr}_index-{i}-bootstd", res.standard_error)
    else:
        return (f"ring_density_incr-{incr}_index-{i}-bootstd", None)


def get_std_tip_in_ring_new_bootstrap(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        res = get_tip_in_ring_bootstrap(
            hull1, hull2, t, exp, rh_only, max_t, n_resamples=100
        )
        if res is None:
            return (f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd", None)
        else:
            return (
                f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd",
                res.standard_error,
            )
    else:
        return (f"ring_active_tips_density_incr-{incr}_index-{i}-bootstd", None)


def get_biovolume_density_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls):
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        biomass = get_biovolume_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_biovolume_density_incr-{incr}_index-{i}", biomass / area)
    else:
        return (f"ring_biovolume_density_incr-{incr}_index-{i}", None)


def get_density_BAS_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        length = get_BAS_length_in_ring(hull1, hull2, t, exp)
        area = ring_area(hull1, hull2)
        return (f"ring_bas_density_incr-{incr}_index-{i}", length / area)
    else:
        return (f"ring_bas_density_incr-{incr}_index-{i}", None)


# def get_mean_speed_in_ring(exp, t, args):
#     incr = args["incr"]
#     i = args["i"]
#     rh_only = args["rh_only"]
#     op_id = args["op_id"]
#
#     regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
#     if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
#         hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
#         speed = get_speed_in_ring(hull1, hull2, t, exp, rh_only, op_id)
#         return (f"mean_speed_incr-{incr}_index-{i}", speed)
#     else:
#         return (f"mean_speed_incr-{incr}_index-{i}", None)


def get_density_anastomose_in_ring(exp: Experiment, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_anas_in_ring(hull1, hull2, t, exp, rh_only)
        area = ring_area(hull1, hull2)
        return (f"ring_anas_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_anas_density_incr-{incr}_index-{i}", None)


def get_density_branch_rate_in_ring(exp, t, args):
    """A function to get the density of branch rate in a ring.
    :param exp: an experiment object
    :param t: the time point
    :param args: a dictionary of arguments
    """
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_branch_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_branch_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_branch_density_incr-{incr}_index-{i}", None)


def get_density_stop_rate_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_stop_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_stop_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_stop_density_incr-{incr}_index-{i}", None)

def get_density_lost_track_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_rate_stop_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_stop_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_stop_density_incr-{incr}_index-{i}", None)

def get_density_active_tips_in_ring(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    max_t = args["max_t"] if "max_t" in args.keys() else np.inf
    regular_hulls, indexes = get_regular_hulls_area_fixed(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_num_active_tips_in_ring(hull1, hull2, t, exp, rh_only, max_t)
        area = ring_area(hull1, hull2)
        return (f"ring_active_tips_density_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_active_tips_density_incr-{incr}_index-{i}", None)


def get_density_active_tips_in_ring_fixed(exp, t, args):
    incr = args["incr"]
    i = args["i"]
    rh_only = args["rh_only"]
    regular_hulls, indexes = get_regular_hulls(exp, range(exp.ts), incr)
    if i + 2 <= len(regular_hulls) and t <= exp.ts - 2:
        hull1, hull2 = regular_hulls[i], regular_hulls[i + 1]
        rate = get_num_active_tips_in_ring(hull1, hull2, t, exp, rh_only)
        area = ring_area(hull1, hull2)
        return (f"ring_active_tips_density_reg_incr-{incr}_index-{i}", rate / area)
    else:
        return (f"ring_active_tips_density_reg_incr-{incr}_index-{i}", None)
