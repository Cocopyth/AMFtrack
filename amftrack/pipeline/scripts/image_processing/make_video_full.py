import sys
import pandas as pd
import numpy as np
from time import time_ns
from amftrack.util.sys import temp_path

import os
from amftrack.pipeline.functions.image_processing.experiment_util import (
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,

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
exp = Experiment(directory)
exp.load(select)
nodes = get_all_nodes(exp, 0)
nodes = [node for node in nodes if node.is_in(0) and
         np.linalg.norm(node.pos(0)-node.pos(node.ts()[-1]))>1000 and
         len(node.ts())>3]

paths_list = []

node_select_list = [choice(nodes) for k in range(2)]
for t in range(exp.ts):
    exp.load_tile_information(t)
    paths = []
    for k in range(2):
        node_select= node_select_list[k]
        pos = node_select.pos(0)
        window = 1500
        region = [[pos[0] - window, pos[1] - window], [pos[0] + window, pos[1] + window]]
        path = f"plot_nodes_{time_ns()}"
        path = os.path.join(temp_path, path)
        paths.append(path+'.png')
        plot_full_image_with_features(
            exp,
            t,
            region=region,
            downsizing=5,
            nodes=[node for node in get_all_nodes(exp, 0) if node.is_in(t) and np.linalg.norm(node.pos(t) - pos) <= window],
            edges=get_all_edges(exp, t),
            dilation=5,
            prettify=False,
            save_path=path,
        )
    paths_list.append(paths)

dir_drop = "DATA/PRINCE"
upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_tracked.mp4"
texts = [(folder,'') for folder in list(select['folder'])]
resize = (2048,2048)
make_video_tile(paths_list,texts,resize,save_path=None,upload_path=upload_path,fontScale=3)
for paths in paths_list:
    for path in paths:
        os.remove(path)