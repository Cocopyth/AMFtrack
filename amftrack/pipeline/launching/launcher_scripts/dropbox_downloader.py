import sys
from amftrack.util.dbx import get_dropbox_folders
from amftrack.pipeline.launching.run_super import run_parallel_transfer,run_launcher

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
stage = int(sys.argv[3])
plates = sys.argv[4:]

dir_drop = "DATA/PRINCE"
all_folders_drop = get_dropbox_folders("/DATA/PRINCE", True)
folders_drop = all_folders_drop.loc[all_folders_drop["unique_id"].isin(plates)]
run_parallel_transfer(
    "from_drop.py",
    [directory_targ],
    folders_drop,
    1,
    "30:00",
    "staging",
    cpus=1,
    node="staging",
name_job = name_job
)
if stage>0:
    run_launcher('stitcher.py',[directory_targ,name_job,stage-1],plates,'20:00',
                 dependency=True,name_job = name_job)