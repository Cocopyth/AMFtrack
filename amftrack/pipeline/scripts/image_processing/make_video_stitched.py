import sys
from amftrack.util.sys import temp_path
import pandas as pd
from amftrack.util.video_util import make_video,make_video_tile
import os
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json")

unique_ids = list(set(run_info["unique_id"].values))
unique_ids.sort()
select = run_info.loc[run_info["unique_id"] == unique_ids[i]]
select = select.sort_values('datetime')
paths = list(select['total_path'])
paths = [os.path.join(path,'StitchedImage.tif') for path in paths]
texts = list(select['folder'])
id_unique = (
    str(int(select["Plate"].iloc[0]))
    + "_"
    + str(int(str(select["CrossDate"].iloc[0]).replace("'", "")))
)
dir_drop = "DATA/PRINCE"
upload_path = f"/{dir_drop}/{id_unique}/{id_unique}_stitched.mp4"
resize = (2624, 1312)
make_video(paths,texts,resize,save_path=None,upload_path=upload_path,fontScale=3)