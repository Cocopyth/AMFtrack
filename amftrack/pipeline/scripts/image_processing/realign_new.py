import pickle
import sys

import networkx as nx

from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.functions.image_processing.experiment_util import get_all_edges
from amftrack.util.sys import temp_path
from scipy import sparse
import scipy.io as sio
from pymatreader import read_mat
import cv2
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
)
import scipy.sparse

import pandas as pd

def transform_pixel_list(pixel_list,R,trans):
    return([R@np.array(pixel)+trans for pixel in pixel_list])

def process(args):
    j = int(args[-1])
    op_id = int(args[-2])

    directory = str(args[1])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    folders = run_info.sort_values("datetime")

    select = folders.iloc[j+1 : j + 2]
    directory_name = folder_list[j+1]
    path_snap = directory + directory_name
    exp = Experiment(directory)
    exp.load(select, suffix="_width")
    Rs = [np.array([[1, 0], [0, 1]])]
    ts = [np.array([0, 0])]
    for i, directory_name in enumerate(folder_list[1 : j + 1]):
        transform = sio.loadmat(path_snap + "/Analysis/transform_new.mat")
        R, t = transform["R"], transform["t"]
        Rs.append(R)
        ts.append(t)

    R0 = np.array([[1, 0], [0, 1]])
    t0 = np.array([0, 0])
    for i in range(len(Rs)):
        index = len(Rs)-1-i
        R0 = np.dot(Rs[index], R0)
        t0 = np.dot(ts[index], Rs[index]) + np.dot(t0, Rs[index])
    time = 0
    edges = get_all_edges(exp, time)

    new_pixel_list = {(edge.begin.label, edge.end.label): transform_pixel_list(edge.pixel_list(time),R0,t0) for edge in edges}

    nx.set_edge_attributes(exp.nx_graph[time], new_pixel_list, "pixel_list")
    pos = exp.positions[time]
    pos = {node: R0 @ pos[node] + t0 for node in pos.keys()}
    directory_name = exp.folders['folder'].iloc[time]
    path_snap = directory + directory_name
    path_save = path_snap + "/Analysis/nx_graph_pruned_realigned.p"
    pickle.dump((exp.nx_graph[time], pos), open(path_save, "wb"))

if __name__ == "__main__":
    process(sys.argv)
