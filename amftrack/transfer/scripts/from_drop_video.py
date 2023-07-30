from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
import pandas as pd

from amftrack.util.sys import temp_path
from amftrack.util.dbx import download_video_folders_drop

directory = str(sys.argv[1])

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
run_info = run_info.sort_values("folder")
folders = run_info.iloc[i : i + 1]
download_video_folders_drop(folders, directory)
