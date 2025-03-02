import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel, run_launcher

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
from time import time_ns

suffix_data_info = time_ns()
update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info)
all_folders = get_current_folders(
    directory_targ, local=True, suffix_data_info=suffix_data_info
)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Img/TileConfiguration.txt.registered"] == True]
num_parallel = 100
time = "12:00:00"

args = [directory_targ]
run_parallel(
    "extract_skel_ML.py",
    args,
    folders,
    num_parallel,
    time,
    "skeletonization",
    cpus=128,
    node="fat_rome",
    name_job=name_job,
)
num_parallel = 100
time = "4:00"
args = [directory_targ]
run_parallel(
    "compress_image.py",
    args,
    folders,
    num_parallel,
    time,
    "compress",
    cpus=128,
    node="fat_rome",
    name_job=name_job,
)
num_parallel = 100
time = "10:00"
args = [directory_targ]
run_parallel(
    "detect_blob.py",
    args,
    folders,
    num_parallel,
    time,
    "detect_blob",
    cpus=128,
    node="fat_rome",
    name_job=name_job,
)

if stage > 0:
    run_launcher(
        "skelet_video_maker.py",
        [directory_targ, name_job, stage - 1],
        plates,
        "3:00:00",
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
