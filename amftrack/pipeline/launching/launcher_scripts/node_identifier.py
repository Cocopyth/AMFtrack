import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel,run_launcher, run_parallel_all_time

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_width.p"]==True]
plates = list(set(list(folders["unique_id"].values)))
args = [directory_targ]
num_parallel = 128

for unique_id in plates:
    select = folders.loc[folders["unique_id"] == unique_id]
    time = "2:00:00"
    run_parallel(
        "track_nodes.py",
        args,
        select,
        num_parallel,
        time,
        "track_node",
        cpus=128,
        node="fat",
        name_job=name_job
    )

time = "12:00:00"
run_parallel_all_time(
    "make_labeled_graphs.py",
    args,
    folders,
    num_parallel,
    time,
    "make_graphs",
    cpus=128,
    node="fat",
    dependency=True,
    name_job=name_job
)
if stage>1000:
    run_launcher('node_identifier.py',[directory_targ,name_job,stage-1],plates,'20:00',dependency=True,name_job = name_job)
else:
    run_launcher('dropbox_uploader.py',[directory_targ,name_job],plates,'20:00',dependency=True,name_job = name_job)