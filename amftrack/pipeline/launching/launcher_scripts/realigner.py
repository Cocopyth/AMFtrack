import sys

from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel, run_launcher
from time import time_ns

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]

suffix_data_info = time_ns()
update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info)
all_folders = get_current_folders(
    directory_targ, local=True, suffix_data_info=suffix_data_info
)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Analysis/skeleton_pruned.mat"] == True]
for plate in plates:
    select = folders.loc[folders["unique_id"] == plate]
    num_parallel = 128
    time = "12:00:00"
    # thresh = 10000  # For R. irregularis, thresh 10000 is good. For Aggregatum, higher may be necessary
    # args = [thresh, directory_targ]
    args = [directory_targ]

    run_parallel(
        "final_alignment_new.py",
        args,
        select,
        num_parallel,
        time,
        "realign",
        cpus=128,
        node="fat_rome",
        name_job=name_job,
    )

if stage > 0:
    run_launcher(
        "skelet_realigner.py",
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
