import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel_all_time, run_launcher

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
dir_drop = "DATA/PRINCE"
from time import time_ns

suffix_data_info = time_ns()
update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info)
all_folders = get_current_folders(
    directory_targ, local=True, suffix_data_info=suffix_data_info
)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Img/TileConfiguration.txt.registered"] == True]
num_parallel = 50
time = "10:00"
args = []
run_parallel_all_time(
    "make_video_single.py",
    args,
    folders,
    num_parallel,
    time,
    "make_video",
    cpus=32,
    node="fat_rome",
    dependency=False,
    name_job=name_job,
)

run_parallel_all_time(
    "make_video_stitched.py",
    args,
    folders,
    num_parallel,
    time,
    "make_video",
    cpus=32,
    node="fat_rome",
    dependency=False,
    name_job=name_job,
)

if stage > 0:
    run_launcher(
        "skeletonizer_agg.py",
        [directory_targ, name_job, stage - 1],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
elif stage == 0:
    run_launcher(
        "dropbox_uploader.py",
        [directory_targ, name_job],
        plates,
        "3:00:00",
        dependency=True,
        name_job=name_job,
    )
