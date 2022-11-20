import sys
import os

from amftrack.util.sys import get_dirname, temp_path
import pandas as pd
import ast
from scipy import sparse
import scipy.io as sio
import cv2
import imageio.v2 as imageio
import numpy as np
import scipy.sparse
import os
from time import time
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)

from amftrack.util.sys import get_dates_datetime, get_dirname
import shutil

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
directory = str(sys.argv[1])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
print(directory_name)
path_snap = os.path.join(directory, directory_name)
path_tile = os.path.join(path_snap, "Img/TileConfiguration.txt.registered")
try:
    tileconfig = pd.read_table(
        path_tile,
        sep=";",
        skiprows=4,
        header=None,
        converters={2: ast.literal_eval},
        skipinitialspace=True,
    )
except:
    print("error_name")
    path_tile = os.path.join(path_snap, "Img/TileConfiguration.registered.txt")
    tileconfig = pd.read_table(
        path_tile,
        sep=";",
        skiprows=4,
        header=None,
        converters={2: ast.literal_eval},
        skipinitialspace=True,
    )
with open(path_tile) as f:
    lines = f.readlines()
lines_modified = []
for line in lines[:4]:
    lines_modified.append(line)
for line in lines[4:]:
    x = str(float(line.split(";")[2].split("\n")[0].split(",")[0][2:]) * 2)
    y = str(float(line.split(";")[2].split("\n")[0].split(",")[1][:-1]) * 2)
    line_new = ";".join([line.split(";")[0], line.split(";")[1], f" ({x},{y}) \n"])
    lines_modified.append(line_new)
with open(path_tile, "w") as f:
    for line in lines_modified:
        f.write(f"{line}")
