from amftrack.pipeline.functions.post_processing.area_hulls import *
from amftrack.util.dbx import upload_folder
from amftrack.pipeline.launching.run_super import run_parallel, run_launcher
import sys
from amftrack.util.sys import (
    update_analysis_info,
    get_analysis_info,
)

directory_targ = str(sys.argv[1])
name_job = str(sys.argv[2])
plates = sys.argv[3:]
update_analysis_info(directory_targ)
analysis_info = get_analysis_info(directory_targ)
analysis_folders = analysis_info.loc[analysis_info["unique_id"].isin(plates)]
dir_drop = "DATA/PRINCE_ANALYSIS"

for index, row in analysis_folders.iterrows():
    folder = row["folder_analysis"]
    id_unique = row["unique_id"]
    path = os.path.join(directory_targ, folder)
    target_drop = f"/{dir_drop}/{id_unique}/{folder}"
    print(dir_drop)
    upload_folder(path, target_drop, delete=False)

# run_launcher(
#     "dropbox_uploader.py",
#     [directory_targ, name_job],
#     plates,
#     "24:00:00",
#     dependency=True,
#     name_job=name_job,
# )
