from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import pandas as pd
from amftrack.util.sys import temp_path
from amftrack.util.dbx import zip_file

directory = str(sys.argv[1])
target = str(sys.argv[2])
must_zip = eval(sys.argv[3])
must_unzip = eval(sys.argv[4])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
path_origin = directory + directory_name

path_target = f"{target}{directory_name}"

if must_zip:
    zip_file(path_origin, path_target + ".zip")
if must_unzip:
    unzip_file(path_origin + ".zip", path_target)
