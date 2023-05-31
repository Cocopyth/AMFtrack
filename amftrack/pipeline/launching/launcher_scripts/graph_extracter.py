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
folders = folders.loc[folders["/Analysis/skeleton_realigned_compressed.mat"] == True]
num_parallel = 128
time = "24:00:00"
args = [directory_targ]
run_parallel(
    "extract_nx_graph.py",
    args,
    folders,
    num_parallel,
    time,
    "extract_nx",
    cpus=128,
    node="fat",
    name_job=name_job,
)
if stage > 0:
    run_launcher(
        "width_extracter.py",
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
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
