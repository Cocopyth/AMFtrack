import os
import sys


# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from datetime import datetime, timedelta
from amftrack.util.dbx import upload_folders
from time import time_ns
import concurrent.futures

directory_origin = r"/mnt/sun-temp/TEMP/PRINCE_syncing/"
# directory_origin = r"/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$/Data/Prince2/Images/"

dir_drop = "DATA/PRINCE"
suffix_data_info = str(time_ns())
update_plate_info(
    directory_origin,
    local=True,
    strong_constraint=False,
    suffix_data_info=suffix_data_info,
)
all_folders_origin = get_current_folders(
    directory_origin, local=True, suffix_data_info=suffix_data_info
)

all_folders_origin["date_datetime"] = pd.to_datetime(
    all_folders_origin["date"].astype(str), format="%d.%m.%Y, %H:%M:"
)
selection = (datetime.now() - all_folders_origin["date_datetime"]) <= timedelta(days=1)
current_prince = all_folders_origin.loc[selection]
plates_in_prince = current_prince["unique_id"].unique()
# old_folders = all_folders_origin.loc[
#     all_folders_origin["unique_id"].isin(plates_in_prince) == False
# ]
old_folders = all_folders_origin  # Upload all
# old_folders = old_folders.loc[old_folders['PrincePos'].astype(int)%4==mod]
old_folders["Plate"] = (
    old_folders["Plate"].str.replace("R", "66666").str.replace("[^0-9]", "")
)

# old_folders = all_folders_origin
old_folders = old_folders.sort_values(by=["datetime"], ignore_index=True)
print(len(old_folders), plates_in_prince)
NUM_THREADS = 4


def task(i):
    upload_folders(old_folders[i : i + 1], dir_drop=dir_drop, delete=True)


with concurrent.futures.ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
    executor.map(task, range(len(old_folders)))
