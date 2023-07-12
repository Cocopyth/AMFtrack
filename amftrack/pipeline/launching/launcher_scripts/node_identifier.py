import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import (
    run_parallel,
    run_launcher,
    run_parallel_all_time,
)

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
folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"] == True]
args = [directory_targ]
num_parallel = 128

for unique_id in plates:
    select = folders.loc[folders["unique_id"] == unique_id]
    time = "12:00:00"
    run_parallel(
        "track_nodes.py",
        args,
        select,
        num_parallel,
        time,
        "track_node",
        cpus=128,
        node="fat",
        name_job=name_job,
    )

time = "6:00:00"
run_parallel_all_time(
    "make_labeled_graphs.py",
    args,
    folders,
    32,
    time,
    "make_graphs",
    cpus=32,
    node="thin",
    dependency=True,
    name_job=name_job,
)
if stage > 0:
    run_launcher(
        "full_video_maker.py",
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
