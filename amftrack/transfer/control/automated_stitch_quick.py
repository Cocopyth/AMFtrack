import os
import sys

#
# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from datetime import datetime, timedelta
from amftrack.pipeline.launching.run import run
from time import time_ns

directory_origin = r"/mnt/sun-temp/TEMP/PRINCE_syncing/"
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
selection = (datetime.now() - all_folders_origin["date_datetime"]).between(
    timedelta(days=2), timedelta(days=45)
)
folders = all_folders_origin.loc[selection]
compress = 5
args = [compress]
run(
    "stitch_quick.py",
    args,
    folders,
    pyt_vers="/home/ipausers/bisot/miniconda3/envs/amftrack/bin/python",
)
