from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util.sys import get_dates_datetime, get_dirname, temp_path
from scipy import sparse
import scipy.io as sio
from pymatreader import read_mat
import cv2
import numpy as np
from amftrack.pipeline.functions.image_processing.extract_graph import (
    sparse_to_doc,
)
import scipy.sparse
from amftrack.pipeline.functions.image_processing.realign import (
    transform_skeleton_final,
)
from amftrack.pipeline.paths.directory import directory_scratch
import pandas as pd


j = int(sys.argv[-1])
op_id = int(sys.argv[-2])

directory = str(sys.argv[1])

run_info = pd.read_json(f"{temp_path}/{op_id}.json")
folder_list = list(run_info["folder"])
folder_list.sort()

dilateds = []
# skels = []
skel_docs = []
directory_name = folder_list[0]
path_snap = directory + directory_name
skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
skel = skel_info["skeleton"]
# skels.append(skel)
skel_doc = sparse_to_doc(skel)
skel_docs.append(skel_doc)
Rs = [np.array([[1, 0], [0, 1]])]
ts = [np.array([0, 0])]
for i, directory_name in enumerate(folder_list[1:]):
    path_snap = directory + directory_name
    skel_info = read_mat(path_snap + "/Analysis/skeleton_pruned.mat")
    skel = skel_info["skeleton"]
    #     skels.append(skel)
    if i + 1 == j:
        skel_doc = sparse_to_doc(skel)
        skel_docs.append(skel_doc)
    else:
        skel_docs.append(0)
    try:
        transform = sio.loadmat(path_snap + "/Analysis/transform.mat")
    except:
        transform = sio.loadmat(path_snap + "/Analysis/transform_corrupt.mat")
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
    #     print(i+begin,j)
    R0 = np.dot(np.transpose(Rs[i]), R0)
    t0 = -np.dot(ts[i], np.transpose(Rs[i])) + np.dot(t0, np.transpose(Rs[i]))
    directory_name = folder_list[i]
    path_snap = directory + directory_name
    if i == j:
        print(f"saving {i} {path_snap}")
        skel_aligned = transform_skeleton_final(skel, R0, t0).astype(np.uint8)
        skel_sparse = scipy.sparse.csc_matrix(skel_aligned)
        sio.savemat(
            path_snap + "/Analysis/skeleton_pruned_realigned.mat",
            {"skeleton": skel_sparse, "R": R0, "t": t0},
        )
        dim = skel_sparse.shape
        kernel = np.ones((5, 5), np.uint8)
        itera = 1
        compressed = cv2.resize(
            cv2.dilate(skel_sparse.todense(), kernel, iterations=itera),
            (dim[1] // 5, dim[0] // 5),
        )
        sio.savemat(
            path_snap + "/Analysis/skeleton_realigned_compressed.mat",
            {"skeleton": compressed},
        )
