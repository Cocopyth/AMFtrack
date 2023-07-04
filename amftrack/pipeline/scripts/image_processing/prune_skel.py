import os

import numpy as np
from scipy import sparse
import cv2
from pymatreader import read_mat
import sys

# from extract_graph import dic_to_sparse
from amftrack.pipeline.functions.image_processing.extract_graph import generate_skeleton
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    prune_graph,
    clean_degree_4,
)
from amftrack.util.sys import temp_path
import scipy.sparse
import scipy.io as sio
import pandas as pd
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
threshold = float(sys.argv[1])
skip = bool(sys.argv[2])

directory = str(sys.argv[3])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
path_snap = directory + directory_name
skel = read_mat(path_snap + "/Analysis/skeleton_masked.mat")["skeleton"]
if not skip:
    skeleton = scipy.sparse.dok_matrix(skel)

# nx_graph_poss=[generate_nx_graph(from_sparse_to_graph(skeleton)) for skeleton in skels_aligned]
# nx_graphs_aligned=[nx_graph_pos[0] for nx_graph_pos in nx_graph_poss]
# poss_aligned=[nx_graph_pos[1] for nx_graph_pos in nx_graph_poss]
# nx_graph_pruned=[clean_degree_4(prune_graph(nx_graph),poss_aligned[i])[0] for i,nx_graph in enumerate(nx_graphs_aligned)]
    nx_graph, pos = generate_nx_graph(from_sparse_to_graph(skeleton))
    nx_graph, pos = remove_spurs(nx_graph, pos)

    nx_graph_pruned = clean_degree_4(prune_graph(nx_graph, threshold), pos)[0]
    shape_skel = skel.shape
    skeleton = generate_skeleton(
        nx_graph_pruned, (max(30000, shape_skel[0]), max(60000, shape_skel[1]))
    )
    skel = scipy.sparse.csc_matrix(skeleton, dtype=np.uint8)
else:
    skel = scipy.sparse.csc_matrix(skel, dtype=np.uint8)
sio.savemat(path_snap + "/Analysis/skeleton_pruned.mat", {"skeleton": skel})
dim = skel.shape
kernel = np.ones((5, 5), np.uint8)
itera = 1
sio.savemat(
    path_snap + "/Analysis/skeleton_pruned_compressed.mat",
    {
        "skeleton": cv2.resize(
            cv2.dilate(skel.todense(), kernel, iterations=itera),
            (dim[1] // 5, dim[0] // 5),
        )
    },
)
