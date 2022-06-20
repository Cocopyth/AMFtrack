import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,)
from amftrack.pipeline.launching.run_super import run_parallel,run_launcher

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]
update_plate_info(directory_targ, local=True)
all_folders = get_current_folders(directory_targ, local=True)
folders = all_folders.loc[all_folders["unique_id"].isin(plates)]
folders.loc[folders['/Img/TileConfiguration.txt.registered']==True]
num_parallel = 100
time = "6:00:00"
hyph_width = 30
perc_low = 93
perc_high = 99.5
args = [hyph_width, perc_low, perc_high, directory_targ]
run_parallel(
    "extract_skel_2.py",
    args,
    folders,
    num_parallel,
    time,
    "skeletonization",
    cpus=128,
    node="fat",
    name_job = name_job

)
num_parallel = 100
time = "4:00"
args = [directory_targ]
run_parallel(
    "compress_image.py",
    args,
    folders,
    num_parallel,
    time,
    "compress",
    cpus=128,
    node="fat",
    name_job=name_job

)

if stage>0:
    run_launcher('skelet_video_maker.py',[directory_targ,name_job,stage-1],plates,'20:00',dependency=True,name_job = name_job)
elif stage==0:
    run_launcher('dropbox_uploader.py',[directory_targ,name_job],plates,'20:00',dependency=True,name_job = name_job)