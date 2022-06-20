import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel_all_time,run_launcher

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders = folders.loc[folders["/Analysis/skeleton_compressed.mat"]==True]
num_parallel = 30
time = '1:00:00'
args = []
run_parallel_all_time(
    "make_video_skelet.py",
    args,
    folders,
    num_parallel,
    time,
    "make_video",
    cpus=32,
    node="fat",
    dependency=False,
    name_job=name_job,
)
if stage>0:
    run_launcher('masker.py',[directory_targ,name_job,stage-1],plates,'20:00',dependency=True,name_job = name_job)
elif stage==0:
    run_launcher('dropbox_uploader.py',[directory_targ,name_job],plates,'20:00',dependency=True,name_job = name_job)