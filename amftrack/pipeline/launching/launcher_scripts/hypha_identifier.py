import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel,run_launcher, run_parallel_all_time

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
from time import time_ns
suffix_data_info=time_ns()
update_plate_info(directory_targ, local=True,suffix_data_info=suffix_data_info)
all_folders = get_current_folders(directory_targ, local=True,suffix_data_info=suffix_data_info)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Analysis/nx_graph_pruned_labeled.p"]==True]
args = [directory_targ]
num_parallel = 32
time = "8:00:00"
limit = 1000
version = 1
labeled = True
args = [directory_targ, limit, version, labeled]
run_parallel_all_time(
    "hyphae_extraction.py",
    args,
    folders,
    num_parallel,
    time,
    "hyphae",
    cpus=32,
    node="fat",
    name_job=name_job
)