import sys

from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    get_width_info,
    get_width_info_new,
)


from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.sys import temp_path
import pickle
import networkx as nx
import pandas as pd
from amftrack.pipeline.functions.spore_processing.spore_id import make_spore_data
from amftrack.util.sys import (
    get_dates_datetime,
    get_dirname,
    temp_path,
    get_data_info,
    update_plate_info,
    get_current_folders,
    get_folders_by_plate_id,
)
import json
import os

directory = str(sys.argv[1])
overwrite = eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder_analysis"])
run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
directory_name = folder_list[i]
select = run_info.loc[run_info["folder_analysis"] == directory_name]
row = [row for index, row in select.iterrows()][0]
path_time_plate_info = row["path_time_plate_info"]

unique_id = row["unique_id"]
update_plate_info(directory, local=True)
all_folders = get_current_folders(directory, local=True)
folders = all_folders.loc[all_folders["unique_id"] == unique_id]
folders = folders.sort_values(by="datetime", ascending=True)
folders = folders.loc[folders["/Analysis/nx_graph_pruned.p"]]
exp = Experiment(directory)
exp.load(folders, suffix="")
exp.dates.sort()
spore_datatable = make_spore_data(exp)
path_save = path_time_plate_info.split("/")[0]
print(path_save)
path_save = os.path.join(directory, path_save, "spore_data.json")
spore_datatable.reset_index(inplace=True)
spore_datatable.to_json(path_save)
