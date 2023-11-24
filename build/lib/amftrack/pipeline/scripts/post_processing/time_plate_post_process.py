from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    load_graphs,
)

import pickle
from amftrack.pipeline.functions.post_processing.extract_study_zone import (
    load_study_zone,
)
from amftrack.util.sys import *
from time import time_ns

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
path_time_plate_info = row["path_time_plate_info"]

if not os.path.isfile(f"{directory}{path_time_plate_info}") or overwrite:
    time_plate_info = {}
else:
    time_plate_info = json.load(open(f"{directory}{path_time_plate_info}", "r"))
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
exp.save_location = "/".join(path_exp.split("/")[:-1])
try:
    exp.labeled
except AttributeError:
    exp.labeled = True
load_graphs(exp, directory, post_process=True)
op_id2 = time_ns()
# get_data_tables(op_id2, redownload=True)
folder = row["folder_analysis"]
path = f'{directory}{row["folder_analysis"]}'
load_study_zone(exp)
for t in range(exp.ts):
    data_t = time_plate_info[str(t)] if str(t) in time_plate_info.keys() else {}
    date = exp.dates[t]
    date_str = datetime.strftime(date, "%d.%m.%Y, %H:%M:")
    for index, f in enumerate(list_f):
        list_args[index]["op_id"] = op_id2
        column, result = f(exp, t, list_args[index])
        print(column, result)
        data_t[column] = result
    data_t["date"] = date_str
    data_t["Plate"] = row["Plate"]
    data_t["path_exp"] = row["path_exp"]
    data_t["folder_analysis"] = row["folder_analysis"]
    time_plate_info[str(t)] = data_t
with open(f"{directory}{path_time_plate_info}", "w") as jsonf:
    print("saving")
    json.dump(time_plate_info, jsonf, indent=4)
