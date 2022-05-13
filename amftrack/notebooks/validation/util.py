import sys

sys.path.insert(0, "/home/cbisot/pycode/MscThesis/")
import pandas as pd
from amftrack.util.sys import get_dates_datetime, get_dirname, temp_path
import ast
from amftrack.plotutil import plot_t_tp1
from scipy import sparse
from datetime import datetime
from amftrack.pipeline.functions.image_processing.node_id import orient
import pickle
import scipy.io as sio
from pymatreader import read_mat
from matplotlib import colors
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage import filters
from random import choice
import scipy.sparse
import os
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    sparse_to_doc,
)
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Edge,
    Node,
    Hyphae,
    plot_raw_plus,
)
from amftrack.pipeline.paths.directory import (
    run_parallel,
    find_state,
    directory_scratch,
    directory_project,
)
from amftrack.notebooks.analysis.util import *
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
from amftrack.pipeline.functions.image_processing.hyphae_id_surf import (
    get_pixel_growth_and_new_children,
    get_hyphae,
)
from collections import Counter
from IPython.display import clear_output
from amftrack.notebooks.analysis.data_info import *
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
plt.rcParams.update(
    {"font.family": "verdana", "font.weight": "normal", "font.size": 20}
)
from amftrack.plotutil import plot_node_skel


def transform_skeleton_final_for_show(skeleton_doc, Rot, trans):
    skeleton_transformed = {}
    transformed_keys = np.round(
        np.transpose(np.dot(Rot, np.transpose(np.array(list(skeleton_doc.keys())))))
        + trans
    ).astype(np.int)
    i = 0
    for pixel in list(transformed_keys):
        i += 1
        skeleton_transformed[(pixel[0], pixel[1])] = 1
    skeleton_transformed_sparse = sparse.lil_matrix((27000, 60000))
    for pixel in list(skeleton_transformed.keys()):
        i += 1
        skeleton_transformed_sparse[(pixel[0], pixel[1])] = 1
    return skeleton_transformed_sparse


def get_skeleton_non_aligned(exp, boundaries, t, directory):
    i = t
    plate = exp.plate
    listdir = os.listdir(directory)
    dates = exp.dates
    date = dates[i]
    directory_name = get_dirname(date, plate)
    path_snap = directory + directory_name
    skel = read_mat(path_snap + "/Analysis/skeleton.mat")
    skelet = skel["skeleton"]
    skelet = sparse_to_doc(skelet)
    #     Rot= skel['R']
    #     trans = skel['t']
    skel_aligned = transform_skeleton_final_for_show(
        skelet, np.array([[1, 0], [0, 1]]), np.array([0, 0])
    )
    output = skel_aligned[
        boundaries[2] : boundaries[3], boundaries[0] : boundaries[1]
    ].todense()
    kernel = np.ones((5, 5), np.uint8)
    output = cv2.dilate(output.astype(np.uint8), kernel, iterations=1)
    return output
