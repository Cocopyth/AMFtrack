from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import pandas as pd
import os
from amftrack.util.dbx import upload_folders
from amftrack.util.sys import temp_path

dir_drop = str(sys.argv[1])
delete = bool(sys.argv[2])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(
    os.path.join(temp_path, f"{op_id}.json"), dtype={"unique_id": str}
)
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
line = run_info.loc[run_info["folder"] == directory_name]
upload_folders(line, dir_drop=dir_drop, delete=delete, catch_exception=False)
