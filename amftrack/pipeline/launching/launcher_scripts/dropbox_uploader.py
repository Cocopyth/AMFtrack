import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel_transfer

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])

plates = sys.argv[3:]
dir_drop = "DATA/PRINCE"
from time import time_ns

suffix_data_info = time_ns()
update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info)
all_folders = get_current_folders(
    directory_targ, local=True, suffix_data_info=suffix_data_info
)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
dir_drop = "DATA/PRINCE"
delete = True
run_parallel_transfer(
    "toward_drop.py",
    [dir_drop, delete],
    folders,
    50,
    "4:00:00",
    "staging",
    cpus=1,
    node="staging",
    name_job=name_job,
)
