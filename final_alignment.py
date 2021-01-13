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
from realign import realign, realign_final
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
from time import sleep
from pycpd import RigidRegistration, DeformableRegistration
import open3d as o3d
from cycpd import rigid_registration
import sys


i = int(sys.argv[2])
plate = int(sys.argv[1])

from directory import directory

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime[i : i + 2]
print("========")
print(f"Matching plate {plate} at dates {dates_datetime_chosen}")
print("========")
dates = dates_datetime_chosen

dilateds = []
skels = []
skel_docs = []
for date in dates:
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
    skel = skel_info["skeleton"]
    skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)
isnan = True
while isnan:
    skeleton1, skeleton2 = skel_docs[0], skel_docs[1]
    skelet_pos = np.array(list(skeleton1.keys()))
    samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
    X = np.transpose(skelet_pos[samples, :])
    skelet_pos = np.array(list(skeleton2.keys()))
    samples = np.random.choice(skelet_pos.shape[0], len(skeleton2.keys()) // 100)
    Y = np.transpose(skelet_pos[samples, :])
    reg = rigid_registration(
        **{
            "X": np.transpose(X.astype(float)),
            "Y": np.transpose(Y.astype(float)),
            "scale": False,
        }
    )
    out = reg.register()
    Rfound = reg.R[0:2, 0:2]
    tfound = np.dot(Rfound, reg.t[0:2])
    nx_graph1, pos1 = generate_nx_graph(from_sparse_to_graph(skeleton1))
    nx_graph2, pos2 = generate_nx_graph(from_sparse_to_graph(skeleton2))
    pruned1 = prune_graph(nx_graph1)
    pruned2 = prune_graph(nx_graph2)
    #     pruned1 = nx_graph1
    #     pruned2 = nx_graph2
    t_init = -tfound
    Rot_init = Rfound
    X = np.transpose(
        np.array([pos1[node] for node in pruned1 if pruned1.degree(node) == 3])
    )
    Y = np.transpose(
        np.array([pos2[node] for node in pruned2 if pruned2.degree(node) == 3])
    )
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(X[0,:],X[1,:])
    # ax.scatter(Y[0,:],Y[1,:])
    Xex = np.transpose(np.transpose(np.dot(Rot_init, X)) + t_init)
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(Xex[0,:],Xex[1,:])
    # ax.scatter(Y[0,:],Y[1,:])
    X = np.insert(X, 2, values=0, axis=0)
    Y = np.insert(Y, 2, values=0, axis=0)
    print(X.shape, Y.shape)
    vectorX = o3d.utility.Vector3dVector(np.transpose(X))
    vectorY = o3d.utility.Vector3dVector(np.transpose(Y))
    source = o3d.geometry.PointCloud(vectorX)
    target = o3d.geometry.PointCloud(vectorY)
    threshold = 200
    trans_init = np.asarray(
        [
            [Rot_init[0, 0], Rot_init[0, 1], 0, t_init[0]],
            [Rot_init[1, 0], Rot_init[1, 1], 0, t_init[1]],
            [0, 0, 1, 0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )
    reg_p2p = o3d.registration.registration_icp(
        source,
        target,
        threshold,
        trans_init,
        o3d.registration.TransformationEstimationPointToPoint(),
    )
    print(reg_p2p)
    Rfound = reg_p2p.transformation[0:2, 0:2]
    tfound = reg_p2p.transformation[0:2, 3]
    print(Rfound, tfound)
    X, Y = X[0:2, :], Y[0:2, :]
    Yrep = np.transpose(np.transpose(np.dot(Rfound, X)) + tfound)
    # fig=plt.figure(figsize=(10,9))
    # ax = fig.add_subplot(111)
    # ax.scatter(np.transpose(Yrep)[:,0],np.transpose(Yrep)[:,1])
    # ax.scatter(np.transpose(Y)[:,0],np.transpose(Y)[:,1])
    isnan = np.isnan(tfound[0])
sio.savemat(path_snap + "/Analysis/transform.mat", {"R": Rfound, "t": tfound})
