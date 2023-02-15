import sys
from amftrack.util.sys import (
    update_analysis_info,
    get_analysis_info,
)
from amftrack.pipeline.launching.run_super import run_parallel_post
from amftrack.pipeline.functions.post_processing.time_edge import *
from amftrack.pipeline.launching.run_super import run_parallel, run_launcher
import pandas as pd
import os

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
update_analysis_info(directory_targ)
analysis_info = get_analysis_info(directory_targ)
analysis_folders = analysis_info.loc[analysis_info["unique_id"].isin(plates)]
list_f = [
    get_tot_length_C,
    get_tot_length_straight,
    get_time_since_start,
    get_time_since_begin_exp,
    get_width_edge,
    get_pos_x,
    get_pos_y,
    get_in_ROI,
    get_connected_component_id,
]
list_args = [{}] * len(list_f)
# list_args= [[500],[1000],[2000]]+[[]]
# list_args= [[500]]
overwrite = True
load_graphs = True
num_parallel = 6
time = "5:00:00"
for index, row in analysis_folders.iterrows():
    folder = row["folder_analysis"]
    path_time_plate_info = row["path_time_plate_info"]
    plate = row["Plate"]
    num_cpus = 32
    if os.path.isfile(f"{directory_targ}{path_time_plate_info}"):
        whole_plate_info = pd.read_json(
            f"{directory_targ}{path_time_plate_info}", convert_dates=True
        ).transpose()
        whole_plate_info.index.name = "t"
        whole_plate_info.reset_index(inplace=True)
        run_parallel_post(
            "time_edge_post_process.py",
            list_f,
            list_args,
            [directory_targ, overwrite, load_graphs],
            whole_plate_info,
            num_parallel,
            time,
            "edge_post_process",
            cpus=num_cpus,
            name_job=name_job,
            node="fat",
        )
