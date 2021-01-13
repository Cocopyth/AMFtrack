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
    from_nx_to_tab,
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
from matplotlib import colors
from copy import deepcopy, copy
from collections import Counter
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage.morphology import thin
from skimage import data, filters
from random import choice
import scipy.sparse
import os
from time import time
from skimage.feature import hessian_matrix_det
import sys
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
import scipy.sparse
from realign import transform_skeleton_final

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
j = int(sys.argv[4])
from directory import directory

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()

dates_datetime_chosen = dates_datetime[begin : end + 1]
dates = dates_datetime_chosen

dilateds = []
# skels = []
skel_docs = []
directory_name = get_dirname(dates[0], plate)
path_snap = directory + directory_name
skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
skel = skel_info["skeleton"]
# skels.append(skel)
skel_doc = sparse_to_doc(skel)
skel_docs.append(skel_doc)
Rs = [np.array([[1, 0], [0, 1]])]
ts = [np.array([0, 0])]
for date in dates[1:]:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
    skel = skel_info["skeleton"]
    #     skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)
    transform = sio.loadmat(path_snap + "/Analysis/transform.mat")
    R, t = transform["R"], transform["t"]
    Rs.append(R)
    ts.append(t)

# skel_doc = skel_docs[0]
# skel_aligned_t = skels[0]
# skel_sparse = scipy.sparse.csc_matrix(skels[0])
# directory_name=f'2020{dates[0]}_Plate{0 if plate<10 else ""}{plate}'
# path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
# sio.savemat(path_snap+'/Analysis/skeleton_realigned.mat',{'skeleton' : skel_sparse,'R' : np.array([[1,0],[0,1]]),'t' : np.array([0,0])})
R0 = np.array([[1, 0], [0, 1]])
t0 = np.array([0, 0])
for i, skel in enumerate(skel_docs):
    R0 = np.dot(np.transpose(Rs[i]), R0)
    t0 = -np.dot(ts[i], np.transpose(Rs[i])) + np.dot(t0, np.transpose(Rs[i]))
    print("treatin", i)
    directory_name = f'2020{dates[i]}_Plate{0 if plate<10 else ""}{plate}'
    path_snap = directory + directory_name
    if i == j:
        skel_aligned = transform_skeleton_final(skel, R0, t0).astype(np.uint8)
        skel_sparse = scipy.sparse.csc_matrix(skel_aligned)
        sio.savemat(
            path_snap + "/Analysis/skeleton_pruned_realigned.mat",
            {"skeleton": skel_sparse, "R": R0, "t": t0},
        )
        dim = skel_sparse.shape
        kernel = np.ones((5, 5), np.uint8)
        itera = 1
        skel
        compressed = cv2.resize(
            cv2.dilate(skel_sparse.todense(), kernel, iterations=itera),
            (dim[1] // 5, dim[0] // 5),
        )
        sio.savemat(
            path_snap + "/Analysis/skeleton_realigned_compressed.mat",
            {"skeleton": compressed},
        )
