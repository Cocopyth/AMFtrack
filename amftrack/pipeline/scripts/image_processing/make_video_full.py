import sys
import pandas as pd
import numpy as np
from time import time_ns
from amftrack.util.sys import temp_path
from amftrack.util.video_util import make_images_track
import os
from amftrack.pipeline.functions.image_processing.experiment_util import (
    plot_full_image_with_features,
    get_all_edges,
    get_all_nodes,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from random import choice
from amftrack.util.video_util import make_video, make_video_tile

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
directory = str(sys.argv[1])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
id_unique = (
    str(int(select["Plate"].iloc[0]))
    + "_"
    + str(int(str(select["CrossDate"].iloc[0]).replace("'", "")))
)
select = select.sort_values("datetime")
exp = Experiment(directory)
exp.load(select)

paths_list = make_images_track(exp)
dir_drop = "DATA/PRINCE_ANALYSIS"
upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_tracked.mp4"
texts = [(folder, "", "", "") for folder in list(select["folder"])]
resize = (2048, 2048)
make_video_tile(
    paths_list, texts, resize, save_path=None, upload_path=upload_path, fontScale=3
)
for paths in paths_list:
    for path in paths:
        os.remove(path)
