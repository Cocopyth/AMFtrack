import sys
from amftrack.util.sys import (
    update_analysis_info,
    get_analysis_info,
)
from amftrack.pipeline.launching.run_super import run_parallel_post
from amftrack.pipeline.functions.post_processing.time_hypha import *
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
    get_pos_x,
    get_pos_y,
    get_time_since_begin_exp,
    get_distance_final_pos,
    get_timedelta,
    get_time_since_start,
    get_speed,
    get_timestep,
    get_timestep_init,
    get_time_init,
    get_degree,
    get_width_tip_edge,
    get_width_root_edge,
    get_width_average,
    get_has_reached_final_pos,
    get_in_ROI,
]
# list_f = [local_density,local_density,local_density]
# list_f = [get_time_since_begin_exp]
# list_f = [get_width_tip_edge, get_width_root_edge]
list_args = [{}] * len(list_f)
# list_args= [[500],[1000],[2000]]+[[]]
# list_args= [[500]]
overwrite = True
load_graphs = True
num_parallel = 32
time = "12:00:00"
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
            "time_hypha_post_process.py",
            list_f,
            list_args,
            [directory_targ, overwrite, load_graphs],
            whole_plate_info,
            num_parallel,
            time,
            "time_hypha_post_process",
            cpus=num_cpus,
            name_job=name_job,
            node="fat",
        )

if stage >= 0:
    run_launcher(
        "analysis_uploader.py",
        [directory_targ, name_job],
        plates,
        "12:00:00",
        name="upload_analysis",
        dependency=True,
        name_job=name_job,
    )
elif stage == 0:
    run_launcher(
        "dropbox_uploader.py",
        [directory_targ, name_job],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
