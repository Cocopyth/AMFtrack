import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy import sparse
import cv2
from pymatreader import read_mat

# from extract_graph import dic_to_sparse
from util import get_path, get_dates_datetime, get_dirname
from plotutil import (
    show_im,
    overlap,
    show_im_rgb,
    plot_nodes,
    plot_nodes_from_list,
    plot_t_tp1,
)
from extract_graph import (
    generate_graph_tab_from_skeleton,
    generate_nx_graph_from_skeleton,
    generate_skeleton,
    clean,
)
import networkx as nx
from node_id import (
    second_identification,
    whole_movement_identification,
    first_identification,
    relabel_nodes,
    clean_nodes,
    orient,
    second_identification2,
)
from extract_graph import (
    dic_to_sparse,
    from_sparse_to_graph,
    generate_nx_graph,
    prune_graph,
    from_nx_to_tab,
    from_nx_to_tab_matlab,
    sparse_to_doc,
    connections_pixel_list_to_tab,
    transform_list,
    clean_degree_4,
)
from sparse_util import dilate, zhangSuen
from realign import realign, reconnect, realign2
from util import get_path
import pandas as pd
from datetime import datetime, timedelta
import ast
import os
import scipy.sparse
import scipy.io as sio
import pickle
import sys

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
directory = str(sys.argv[4])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen

nx_graph_pos = []
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned.p"
    nx_graph_pos.append(pickle.load(open(path_save, "rb")))
nx_graph_pruned = [c[0] for c in nx_graph_pos]
poss_aligned = [c[1] for c in nx_graph_pos]
downstream_graphs = []
downstream_pos = []
begin = len(dates) - 1
downstream_graphs = [nx_graph_pruned[begin]]
downstream_poss = [poss_aligned[begin]]
for i in range(begin - 1, -1, -1):
    print("i=", i)
    new_graphs, new_poss = second_identification(
        nx_graph_pruned[i],
        downstream_graphs[0],
        poss_aligned[i],
        downstream_poss[0],
        50,
        downstream_graphs[1:],
        downstream_poss[1:],
        tolerance=30,
    )
    downstream_graphs = new_graphs
    downstream_poss = new_poss

nx_graph_pruned = downstream_graphs
poss_aligned = downstream_poss
for i, g in enumerate(nx_graph_pruned):
    date = dates[i]
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_labeled.p"
    pos = poss_aligned[i]
    pickle.dump((g, pos), open(path_save, "wb"))

for i, date in enumerate(dates):
    tab = from_nx_to_tab(nx_graph_pruned[i], poss_aligned[i])
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/graph_full_labeled.mat"
    sio.savemat(path_save, {name: col.values for name, col in tab.items()})
