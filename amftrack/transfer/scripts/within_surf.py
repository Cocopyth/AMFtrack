from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util.sys import get_dirname
import pandas as pd
import ast
from scipy import sparse
from datetime import datetime
import cv2
import imageio
import numpy as np
import os
from time import time
from amftrack.util.sys import get_dates_datetime, get_dirname, temp_path
from amftrack.pipeline.paths.directory import directory_scratch
from subprocess import call
from amftrack.util.dbx import upload, zip_file

directory = str(sys.argv[1])
target = str(sys.argv[2])
must_zip = eval(sys.argv[3])
must_unzip = eval(sys.argv[4])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f"{temp_path}/{op_id}.json")
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
path_origin = directory + directory_name

path_target = f"{target}{directory_name}"

if must_zip:
    zip_file(path_origin, path_target + ".zip")
if must_unzip:
    unzip_file(path_origin + ".zip", path_target)
