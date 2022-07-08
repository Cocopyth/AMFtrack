import sys
import pandas as pd
import os
import logging

from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.video_util import make_video_targeted
from amftrack.util.formatting import str_to_coord_list

# Arguments
directory = str(sys.argv[1])
coords = str_to_coord_list(str(sys.argv[2]))
size = int(sys.argv[3])
local = bool(sys.argv[4])
upload_bool = bool(sys.argv[5])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

logging.info(
    f"""Launched script {__file__} with parameter:\n 
            size: {size}, coords: {coords[:3]}.., i: {i}, op_id: {op_id}, local: {local}"""
)

if local:
    base_dir = "/home/cbisot/pycode/BAS"
else:
    base_dif = "/gpfs/work1/0/einf914/BAS"

# Fetch data and make Experiment obj
run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
logging.info(f"""unique_ids: {unique_ids}""")
logging.info(f"""len run_info: {len(run_info)}""")
# select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
id_unique = (
    str(int(run_info["Plate"].iloc[0]))
    + "_"
    + str(int(str(run_info["CrossDate"].iloc[0]).replace("'", "")))
)
exp = Experiment(directory)
exp.load(run_info, suffix="")

# Generate the images
directory_path = os.path.join(base_dir, id_unique)
if not os.path.isdir(directory_path):
    os.mkdir(directory_path)

for [x, y] in coords:
    bas_directory_path = os.path.join(directory_path, f"bas-x{x}-y{y}-size{size}")
    if not os.path.isdir(bas_directory_path):
        os.mkdir(bas_directory_path)
    path_list = make_video_targeted(
        exp, coordinate=[x, y], directory_path=bas_directory_path, size=size
    )

# Upload images on dropbox
if upload_bool:
    pass
    # TODO
    # dir_drop = "/DATA/BAS"
    # upload_path = os.path.join(dir_drop, id_unique, f"pos{x}-{y}-size{size}")
    # for path in path_list:
    #     upload(path, target_path=os.path)
