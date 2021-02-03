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
directory = str(sys.argv[2])

i = int(sys.argv[-1])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
date_datetime = dates_datetime[i]
date = date_datetime
directory_name = get_dirname(date, plate)
path_snap = directory + directory_name
skel = read_mat(path_snap + "/Analysis/skeleton_pruned_realigned.mat")["skeleton"]
skeleton = scipy.sparse.dok_matrix(skel)

# nx_graph_poss=[generate_nx_graph(from_sparse_to_graph(skeleton)) for skeleton in skels_aligned]
# nx_graphs_aligned=[nx_graph_pos[0] for nx_graph_pos in nx_graph_poss]
# poss_aligned=[nx_graph_pos[1] for nx_graph_pos in nx_graph_poss]
# nx_graph_pruned=[clean_degree_4(prune_graph(nx_graph),poss_aligned[i])[0] for i,nx_graph in enumerate(nx_graphs_aligned)]
nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
nx_graph_pruned = clean_degree_4(nx_graph, pos)[0]
path_save = path_snap + "/Analysis/nx_graph_pruned.p"
print(path_save)
pickle.dump((nx_graph_pruned, pos), open(path_save, "wb"))
