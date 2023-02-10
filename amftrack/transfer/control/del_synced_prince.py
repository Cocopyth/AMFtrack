import os
import sys


# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from tqdm.autonotebook import tqdm
import checksumdir
import shutil

# directory_origin = r'/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing/'
directory_origin = r"/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$,user=bisot/Data/Prince2/Images/"
directory_target = (
    r"/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing/"
)
directory_target = r"/mnt/sun-temp/TEMP/PRINCE_syncing/"

update_plate_info(directory_origin, local=True)
all_folders_origin = get_current_folders(directory_origin, local=True)
update_plate_info(directory_target, local=True)


all_folders_target = get_current_folders(directory_target, local=True)
run_info = all_folders_target.copy()
run_info = run_info.sort_values("datetime")
folder_list = list(run_info["folder"])
with tqdm(total=len(folder_list), desc="deleted") as pbar:
    for folder in folder_list:
        origin = all_folders_origin.loc[all_folders_origin["folder"] == folder][
            "total_path"
        ]
        if len(origin) > 0:
            origin = origin.iloc[0]
        else:
            # print(folder)
            continue
        target = all_folders_target.loc[all_folders_target["folder"] == folder][
            "total_path"
        ].iloc[0]
        check_or = checksumdir.dirhash(origin)
        check_targ = checksumdir.dirhash(target)
        print(folder, (check_or == check_targ))

        if check_or == check_targ and origin != target:
            shutil.rmtree(origin)
        pbar.update(1)
