import sys

sys.path.insert(0, "/home/cbisot/pycode/MscThesis/")
from amftrack.util.sys import get_dirname
from pymatreader import read_mat
import cv2
import matplotlib.pyplot as plt
import os
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
)
from amftrack.notebooks.analysis.util import *
from IPython.core.interactiveshell import InteractiveShell

InteractiveShell.ast_node_interactivity = "all"
plt.rcParams.update(
    {"font.family": "verdana", "font.weight": "normal", "font.size": 20}
)


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
    plate = exp.prince_pos
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
