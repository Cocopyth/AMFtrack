from util import get_path, get_dates_datetime, get_dirname
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from extract_graph import (
    generate_nx_graph,
    transform_list,
    generate_skeleton,
    generate_nx_graph_from_skeleton,
    from_connection_tab,
)
from node_id import whole_movement_identification, second_identification
import ast
from plotutil import plot_t_tp1, compress_skeleton
from scipy import sparse
from sparse_util import dilate, zhangSuen
from realign import realign
from datetime import datetime, timedelta
from node_id import orient
import pickle
from matplotlib.widgets import CheckButtons
import scipy.io as sio
import imageio
from pymatreader import read_mat
import os
from matplotlib import colors
from random import choice
from experiment_class_surf import Experiment, clean_exp_with_hyphaes
from hyphae_id_surf import (
    clean_and_relabel,
    get_mother,
    save_hyphaes,
    resolve_ambiguity_two_ends,
    solve_degree4,
    clean_obvious_fake_tips,
)
import sys

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
from directory import directory

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen
exp = Experiment(plate)
exp.load(dates)
exp_clean = clean_and_relabel(exp)
to_remove = []
for hyph in exp_clean.hyphaes:
    hyph.update_ts()
    if len(hyph.ts) == 0:
        to_remove.append(hyph)
for hyph in to_remove:
    exp_clean.hyphaes.remove(hyph)
get_mother(exp_clean.hyphaes)
solve_two_ends = resolve_ambiguity_two_ends(exp_clean.hyphaes)
# solved = solve_degree4(exp_clean)
clean_obvious_fake_tips(exp_clean)
dirName = f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}"
try:
    os.mkdir(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}")
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")
hyphs, gr_inf = save_hyphaes(
    exp_clean, f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/"
)
exp_clean.save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
exp_clean.pickle_save(f"{directory}Analysis_Plate{plate}_{dates[0]}_{dates[-1]}/")
