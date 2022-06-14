import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel_all_time

directory_targ = str(sys.argv[1])
plates = sys.argv[2:]
dir_drop = "DATA/PRINCE"
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
num_parallel = 50
time = '10:00'
args = []
run_parallel_all_time(
    "make_video_single.py",
    args,
    folders,
    num_parallel,
    time,
    "make_video",
    cpus=32,
    node="fat",
    dependency=False,
    name_job="video.sh",
)
run_parallel_all_time(
    "make_video_stitched.py",
    args,
    folders,
    num_parallel,
    time,
    "make_video",
    cpus=32,
    node="fat",
    dependency=False,
    name_job="one_shot.sh"
)