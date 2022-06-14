import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel_stitch

directory_targ = str(sys.argv[1])
plates = sys.argv[2:]
dir_drop = "DATA/PRINCE"
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
num_parallel = 50
time = "40:00"
run_parallel_stitch(directory_targ, folders, num_parallel, time, cpus=128,
                    node="fat",    name_job="one_shot.sh")