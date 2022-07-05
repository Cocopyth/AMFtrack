import os
import sys


# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)

from datetime import datetime, timedelta
import imageio
import os
from amftrack.util.dbx import sync_fold
from tqdm.autonotebook import tqdm
from time import time_ns
import subprocess

path_code = os.getenv("HOME") + "/pycode/MscThesis/"


def make_stitching_loop(directory, dirname, op_id):
    a_file = open(
        f"{path_code}amftrack/pipeline/scripts/stitching_loops/stitching_loop.ijm", "r"
    )

    list_of_lines = a_file.readlines()

    list_of_lines[4] = f"mainDirectory = \u0022{directory}\u0022 ;\n"
    list_of_lines[29] = f"\t if(startsWith(list[i],\u0022{dirname}\u0022)) \u007b\n"
    file_name = f'{os.getenv("TEMP")}/stitching_loops/stitching_loop{op_id}.ijm'
    a_file = open(file_name, "w")

    a_file.writelines(list_of_lines)

    a_file.close()


def run_stitch(directory, folders):
    folder_list = list(folders["folder"])
    folder_list.sort()
    with tqdm(total=len(folder_list), desc="stitched") as pbar:
        for folder in folder_list:
            op_id = time_ns()
            im = imageio.imread(f"{directory}/{folder}/Img/Img_r03_c05.tif")
            for x in range(1, 11):
                for y in range(1, 16):
                    strix = str(x) if x >= 10 else f"0{x}"
                    striy = str(y) if y >= 10 else f"0{y}"
                    path = f"{directory}/{folder}/Img/Img_r{strix}_c{striy}.tif"
                    # print(striy,path,os.path.getsize(path))
                    if not os.path.isfile(path):
                        f = open(path, "w")
                    if os.path.getsize(path) == 0:
                        imageio.imwrite(path, im * 0)
            make_stitching_loop(directory, folder, op_id)
            command = [
                os.getenv("HOME") + "/Fiji.app/ImageJ-linux64",
                "--mem=8000m",
                "--headless",
                "--ij2",
                "--console",
                "-macro",
                f'{os.getenv("TEMP")}/stitching_loops/stitching_loop{op_id}.ijm',
            ]
            print(" ".join(command))
            process = subprocess.run(command)
            pbar.update(1)


def run_parallel(code, args, folders):
    op_id = time_ns()
    folders.to_json(f'{os.getenv("TEMP")}/{op_id}.json')  # temporary file
    folder_list = list(folders["folder"])
    folder_list.sort()
    args_str = [str(arg) for arg in args]
    arg_str = " ".join(args_str)
    with tqdm(total=len(folder_list), desc="stitched") as pbar:
        for index, folder in enumerate(folder_list):
            command = (
                [
                    "python",
                    f"{path_code}amftrack/pipeline/scripts/image_processing/{code}",
                ]
                + args_str
                + [f"{op_id}", f"{index}"]
            )
            print(" ".join(command))
            process = subprocess.run(command)
            pbar.update(1)


# directory_origin = r'/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing/'
directory_origin = r"/mnt/sun-temp/TEMP/PRINCE_syncing/"

# directory_target = r'/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_analysis/'
directory_target = r"/mnt/sun-temp/TEMP/PRINCE_analysis/"

update_plate_info(directory_origin, local=True)
all_folders_origin = get_current_folders(directory_origin, local=True)
update_plate_info(directory_target, local=True)
all_folders_target = get_current_folders(directory_target, local=True)

all_folders_origin["date_datetime"] = pd.to_datetime(
    all_folders_origin["date"].astype(str), format="%d.%m.%Y, %H:%M:"
)
selection = (datetime.now() - all_folders_origin["date_datetime"]) <= timedelta(
    hours=15
)
last_folders = all_folders_origin.loc[selection]
plates_to_stitch = last_folders["Plate"].unique()

for plate in plates_to_stitch:
    folders = (
        all_folders_origin.loc[all_folders_origin["Plate"] == plate]
        .sort_values("date_datetime", ascending=False)
        .head(1)
    )
    run_info = folders.copy()
    target = directory_target
    folder_list = list(run_info["total_path"])
    for folder in folder_list:
        origin = folder
        sync_fold(origin, target)

for plate in plates_to_stitch:
    folders = (
        all_folders_origin.loc[all_folders_origin["Plate"] == plate]
        .sort_values("date_datetime", ascending=False)
        .head(1)
    )
    run_stitch(directory_target, folders)
