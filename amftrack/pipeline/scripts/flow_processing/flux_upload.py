from amftrack.pipeline.functions.transport_processing.high_mag_videos.kymo_class import *
import amftrack.pipeline.functions.transport_processing.high_mag_videos.plot_data as dataplot
import sys
import pandas as pd
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload_folder
import numpy as np

directory = str(sys.argv[1])
GST_params = sys.argv[2:6]
upl_targ = str(sys.argv[6])
folders = str(sys.argv[3])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

print(f"This is iteration {i}, with parameters {GST_params}")
print(upl_targ)

dataframe = pd.read_json(f"{temp_path}/{op_id}.json")


dataframe = dataframe.iloc[i]
drop_targ = os.path.relpath(f"/{dataframe['tot_path_drop']}", upl_targ)

img_address = dataframe['analysis_folder']
db_address = f"{upl_targ}KymoSpeeDExtract/{drop_targ}"

upload_folder(img_address, db_address, delete=False)
print(f"{img_address} should be empty now!")
