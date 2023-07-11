import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel_stitch, run_launcher
from time import time_ns

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
dir_drop = "DATA/PRINCE"
suffix_data_info = time_ns()
update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info)
all_folders = get_current_folders(
    directory_targ, local=True, suffix_data_info=suffix_data_info
)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
# folders = folders.loc[folders["/Img/TileConfiguration.txt.registered"] == False]

num_parallel = 50
time = "40:00"
if len(folders) > 0:
    run_parallel_stitch(
        directory_targ,
        folders,
        num_parallel,
        time,
        cpus=128,
        node="fat",
        name_job=name_job,
    )
if stage > 0:
    run_launcher(
        "video_maker.py",
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
