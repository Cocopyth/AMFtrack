import sys
from amftrack.util.dbx import get_dropbox_folders
from amftrack.pipeline.launching.run_super import run_parallel_transfer

directory_targ = str(sys.argv[1])
plates = sys.argv[2:]
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
    name_job="one_shot.sh"
)