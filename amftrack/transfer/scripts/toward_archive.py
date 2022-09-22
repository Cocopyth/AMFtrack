from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import pandas as pd
import numpy as np
import os
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload, zip_file

directory = str(sys.argv[1])
dir_drop = str(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
id_unique = run_info.loc[run_info["folder"] == directory_name]["unique_id"].iloc[0]
path_snap = directory + directory_name
API = str(np.load("/home/cbisot/pycode/API_drop.npy"))

path_zip = f"/scratch-shared/amftrack/temp/{directory_name}.zip"
zip_file(path_snap, path_zip)
upload(
    path_zip,
    f"/{dir_drop}/{id_unique}/{directory_name}.zip",
    chunk_size=256 * 1024 * 1024,
)
os.remove(path_zip)
