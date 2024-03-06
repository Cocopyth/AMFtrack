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
tp1 = t + 1
try:
    exp.labeled
except AttributeError:
    exp.labeled = True  # For older versions of experiments, to be removed later
load_study_zone(exp)
if load_graphs_bool:
    load_graphs(exp, directory, indexes=[t, tp1], post_process=True)
# load_skel(exp,[t])
# print('size after loading',get_size(exp)/10**6)

folder_analysis = row["folder_analysis"]
path_hyph_info_t = f"{directory}{folder_analysis}/time_hypha_info/hyph_info_{t}.json"
if not os.path.isfile(path_hyph_info_t) or overwrite:
    time_hypha_info_t = {}
else:
    time_hypha_info_t = json.load(open(path_hyph_info_t, "r"))

if tp1 < exp.ts:
    for hypha in exp.hyphaes:
        if t in hypha.ts and tp1 in hypha.ts:
            data_hypha = (
                time_hypha_info_t[str(hypha.end.label)]
                if str(hypha.end.label) in time_hypha_info_t.keys()
                else {}
            )
            data_hypha["end"] = hypha.end.label
            for index, f in enumerate(list_f):
                column, result = f(hypha, t, tp1, list_args[index])
                data_hypha[column] = result
            time_hypha_info_t[str(hypha.end.label)] = data_hypha
path = f"{directory}{folder_analysis}/time_hypha_info"
print(path)
try:
    os.mkdir(path)
except OSError as error:
    pass
with open(path_hyph_info_t, "w") as jsonf:
    json.dump(time_hypha_info_t, jsonf, indent=4)
