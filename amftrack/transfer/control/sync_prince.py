import os
import sys


# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from amftrack.util.dbx import sync_fold
from tqdm.autonotebook import tqdm

# directory = r"/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$/Data/Prince2/Images/"
directory = r"/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$,user=bisot/Data/Prince2/Images/"

update_plate_info(directory, strong_constraint=False, local=True)

all_folders = get_current_folders(directory, local=True)
directory2 = r"/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing"
directory2 = r"/mnt/sun-temp/TEMP/PRINCE_syncing"

folders = all_folders
run_info = folders.copy()
target = directory2
folder_list = list(run_info["total_path"])
with tqdm(total=len(folder_list), desc="transferred") as pbar:
    for folder in folder_list:
        origin = folder
        sync_fold(origin, target)
        pbar.update(1)
