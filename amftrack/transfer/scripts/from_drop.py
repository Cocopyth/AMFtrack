from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import pandas as pd
import numpy as np
import os
from amftrack.util.sys import temp_path
from amftrack.util.dbx import download, unzip_file

directory = str(sys.argv[1])
dir_drop = str(sys.argv[2])
unzip = eval(sys.argv[3])
flatten = eval(sys.argv[4])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f"{temp_path}/{op_id}.json")
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
run_info["unique_id"] = (
    run_info["Plate"].astype(str) + "_" + run_info["CrossDate"].astype(str)
)

id_unique = run_info.loc[run_info["folder"] == directory_name]["unique_id"].iloc[0]

path_snap = directory + directory_name
API = str(np.load(os.getenv("HOME") + "/pycode/API_drop.npy"))
if unzip:
    path_zip = f'{os.getenv("TEMP")}{directory_name}.zip'


else:
    if flatten:
        path_zip = f"{path_snap}.zip"
    else:
        try:
            os.mkdir(f"{directory}/{id_unique}")
        except OSError:
            print("Creation of the directory failed")
        path_zip = f"{directory}/{id_unique}/{directory_name}.zip"

source = f"/{dir_drop}/{id_unique}/{directory_name}.zip"
print(path_zip, source)

download(source, path_zip)
if unzip:
    unzip_file(path_zip, path_snap)
    print(path_snap, path_zip)
    os.remove(path_zip)
