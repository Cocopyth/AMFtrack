from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    load_graphs,
)
from amftrack.util.sys import temp_path
import pickle
import pandas as pd
import os
import json
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
import sys
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
import numpy as np
import networkx as nx
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    get_width_info_new,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    save_graphs,
)

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
load_graphs_bool = eval(sys.argv[3])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
resolution = 50
skip = False
run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})

folder_list = list(run_info["t"])
t = folder_list[i]
select = run_info.loc[run_info["t"] == t]
row = [row for index, row in select.iterrows()][0]
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
exp.save_location = "/".join(path_exp.split("/")[:-1])
t = row["t"]
try:
    exp.labeled
except AttributeError:
    exp.labeled = True  # For older versions of experiments, to be removed later
load_study_zone(exp)
if load_graphs_bool:
    load_graphs(exp, directory, indexes=[t], post_process=True)
# load_skel(exp,[t])
# print('size after loading',get_size(exp)/10**6)

exp.load_tile_information(t)
(G, pos) = exp.nx_graph[t], exp.positions[t]
edge_test = get_width_info_new(exp, t, resolution=resolution, skip=skip)
nx.set_edge_attributes(G, edge_test, "width")
exp.nx_graph[t] = G
save_graphs(exp, ts=[t])
