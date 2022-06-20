import sys
from amftrack.util.sys import temp_path
import pandas as pd
from pymatreader import read_mat
import cv2
from amftrack.pipeline.functions.image_processing.extract_skel import bowler_hat
import numpy as np
from scipy import sparse
from time import time_ns
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload
import imageio
import os
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
    distance_point_edge,
    plot_edge,
    plot_edge_cropped,
    find_nearest_edge,
    get_edge_from_node_labels,
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,
    find_neighboring_edges,
    reconstruct_image,
    reconstruct_skeletton_from_edges,
    reconstruct_skeletton_unicolor,
    plot_edge_width,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from random import choice
from amftrack.util.video_util import make_video,make_video_tile

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
directory = str(sys.argv[1])

run_info = pd.read_json(f"{temp_path}/{op_id}.json")
unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
id_unique = unique_ids[i]
select = select.sort_values('datetime')
select = select.iloc[:10]
exp = Experiment(directory)
exp.load(select)
nodes = get_all_nodes(exp, 0)
nodes = [node for node in nodes if node.is_in(0) and
         np.linalg.norm(node.pos(0)-node.pos(node.ts()[-1]))>1000 and
         len(node.ts())>3]

paths_list = []

for k in range(2):
    paths = []
    node_select = choice(nodes)
    pos = node_select.pos(0)
    for t in range(exp.ts):
        exp.load_tile_information(t)
        window = 1500
        region = [[pos[1] - window, pos[1] + window], [pos[0] - window, pos[0] + window]]
        path = f"plot_nodes_{time_ns()}"
        paths.append(path)
        plot_full_image_with_features(
            exp,
            t,
            region=region,
            downsizing=5,
            nodes=[node for node in get_all_nodes(exp, 0) if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window],
            edges=get_all_edges(exp, t),
            dilation=20,
            prettify=True,
            save_path=os.path.join(temp_path, path),
        )
    paths_list.append(path)

dir_drop = "DATA/PRINCE"
upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_tracked.mp4"
texts = [(folder,'') for folder in list(select['folder'])]
resize = (2048,1504)
make_video_tile(paths_list,texts,resize,save_path=None,upload_path=upload_path,fontScale=3)