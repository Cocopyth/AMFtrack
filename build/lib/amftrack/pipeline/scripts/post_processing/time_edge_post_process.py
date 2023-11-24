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


directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
load_graphs_bool = eval(sys.argv[3])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
list_f, list_args = pickle.load(open(f"{temp_path}/{op_id}.pick", "rb"))

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

folder_analysis = row["folder_analysis"]
path_edge_info_t = f"{directory}{folder_analysis}/time_edge_info/edge_info_{t}.json"
if not os.path.isfile(path_edge_info_t) or overwrite:
    time_edge_info_t = {}
else:
    time_edge_info_t = json.load(open(path_edge_info_t, "r"))

for edge in exp.nx_graph[t].edges:
    edge = Edge(Node(np.min(edge), exp), Node(np.max(edge), exp), exp)
    if t in edge.ts():
        data_edge = (
            time_edge_info_t[str((edge.end.label, edge.begin.label))]
            if str((edge.end.label, edge.begin.label)) in time_edge_info_t.keys()
            else {}
        )
        data_edge["end"] = int(edge.end.label)
        data_edge["begin"] = int(edge.begin.label)
        for index, f in enumerate(list_f):
            column, result = f(edge, t, list_args[index])
            data_edge[column] = result
        time_edge_info_t[str((edge.end.label, edge.begin.label))] = data_edge
print(time_edge_info_t)
path = f"{directory}{folder_analysis}/time_edge_info"
print(path)
try:
    os.mkdir(path)
except OSError as error:
    pass
with open(path_edge_info_t, "w") as jsonf:
    json.dump(time_edge_info_t, jsonf, indent=4)
