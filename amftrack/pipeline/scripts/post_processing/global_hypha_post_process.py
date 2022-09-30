from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
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
from time import time

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
list_f, list_args = pickle.load(open(f"{temp_path}/{op_id}.pick", "rb"))
folder_list = list(run_info["folder_analysis"])
directory_name = folder_list[i]
select = run_info.loc[run_info["folder_analysis"] == directory_name]
row = [row for index, row in select.iterrows()][0]
path_global_hypha_info = row["path_global_hypha_info"]

if not os.path.isfile(f"{directory}{path_global_hypha_info}") or overwrite:
    global_hypha_info = {}
else:
    global_hypha_info = json.load(open(f"{directory}{path_global_hypha_info}", "r"))
# print(global_hypha_info)
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
exp.save_location = "/".join(path_exp.split("/")[:-1])
try:
    exp.labeled
except AttributeError:
    exp.labeled = True
load_graphs(exp, directory, post_process=True)

load_study_zone(exp)
folder = row["folder_analysis"]
path = f'{directory}{row["folder_analysis"]}'
t = time()
for j, hypha in enumerate(exp.hyphaes):
    # if j%100==0:
    #     print(time()-t)
    #     print(j/len(exp.hyphaes))
    #     t = time()
    data_hypha = (
        global_hypha_info[str(hypha.end.label)]
        if str(hypha.end.label) in global_hypha_info.keys()
        else {}
    )
    # print(data_hypha)
    for index, f in enumerate(list_f):
        column, result = f(hypha, list_args[index])
        # print(column,result)
        data_hypha[column] = result
    data_hypha["Plate"] = row["Plate"]
    data_hypha["path_exp"] = row["path_exp"]
    data_hypha["folder_analysis"] = row["folder_analysis"]
    global_hypha_info[str(hypha.end.label)] = data_hypha
with open(f"{directory}{path_global_hypha_info}", "w") as jsonf:
    json.dump(global_hypha_info, jsonf, indent=4)
