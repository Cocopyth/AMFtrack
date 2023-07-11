import sys
from amftrack.util.dbx import read_saved_dropbox_state
from amftrack.pipeline.launching.run_super import run_parallel_transfer, run_launcher
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from time import sleep

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
next = str(sys.argv[4])
plates = sys.argv[5:]

dir_drop = "DATA/PRINCE"
all_folders_drop = read_saved_dropbox_state("/DATA/PRINCE")
folders_drop = all_folders_drop.loc[all_folders_drop["unique_id"].isin(plates)]
update_plate_info(directory_targ, local=True, strong_constraint=False)
all_folders = get_current_folders(directory_targ, local=True)
if len(all_folders) > 0:
    folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
    folders_drop2 = folders_drop.loc[~folders_drop["folder"].isin(folders["folder"])]
    folders_drop3 = folders_drop2.loc[~folders_drop2["folder"].str.contains("Analysis")]
else:
    folders_drop2 = folders_drop
    folders_drop3 = folders_drop
while len(folders_drop3) > 0:
    run_parallel_transfer(
        "from_drop.py",
        [directory_targ],
        folders_drop2,
        50,
        "4:00:00",
        "staging",
        cpus=1,
        node="staging",
        name_job=name_job,
    )
    sleep(3600)
    update_plate_info(directory_targ, local=True, strong_constraint=False)
    all_folders = get_current_folders(directory_targ, local=True)
    if len(all_folders) > 0:
        folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
        folders_drop2 = folders_drop.loc[
            ~folders_drop["folder"].isin(folders["folder"])
        ]
        folders_drop3 = folders_drop2.loc[
            ~folders_drop2["folder"].str.contains("Analysis")
        ]
    else:
        folders_drop2 = folders_drop
        folders_drop3 = folders_drop

if stage > 0:
    run_launcher(
        next,
        [directory_targ, name_job, stage - 1],
        plates,
        "12:00:00",
        dependency=True,
        name_job=name_job,
    )
