import os
import sys

sys.path.insert(0, os.getenv("HOME") + "/pycode/MscThesis/")
# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from datetime import datetime, timedelta
from amftrack.util.video_util import make_video,make_video_tile

from amftrack.pipeline.launching.run import run
directory_origin = r"/mnt/sun-temp/TEMP/PRINCE_syncing/"
dir_drop = "DATA/PRINCE"
update_plate_info(directory_origin, local=True)
all_folders_origin = get_current_folders(directory_origin, local=True)

all_folders_origin["date_datetime"] = pd.to_datetime(
    all_folders_origin["date"].astype(str), format="%d.%m.%Y, %H:%M:"
)
selection = (datetime.now() - all_folders_origin["date_datetime"]) >= timedelta(days=45)
finished_plates = selection["unique_id"].unique()
selection = (datetime.now() - all_folders_origin["date_datetime"]) < timedelta(days=44)
to_plot = selection.loc[selection['unique_id'].is_in(finished_plates)==False]
plates = to_plot["unique_id"].unique()
for plate in plates:
    select = to_plot.loc[to_plot["unique_id"] == plate]
    select = select.sort_values('datetime')
    paths = list(select['total_path'])
    paths_list = [[os.path.join(path,'Img','Img_r08_c06.tif'),os.path.join(path,'Img','Img_r07_c06.tif')] for path in paths]
    texts = [(folder,'') for folder in list(select['folder'])]
    resize = (2048,1504)
    id_unique = (
        str(int(select["Plate"].iloc[0]))
        + "_"
        + str(int(str(select["CrossDate"].iloc[0]).replace("'", "")))
    )
    print(paths_list)
    dir_drop = "DATA/PRINCE"
    upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_single_tiled.mp4"
    make_video_tile(paths_list,texts,resize,save_path=None,upload_path=upload_path,fontScale=3)