import sys
from amftrack.util.sys import (
    update_plate_info,
    get_current_folders,
)
from amftrack.pipeline.launching.run_super import run_parallel, run_launcher
from amftrack.pipeline.development.high_mag_videos.kymo_class import *


# from amftrack.pipeline.functions.image_processing.extract_width_fun import (
#     get_width_info,
#     get_width_info_new,
# )


from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.util.sys import temp_path
import pickle
import networkx as nx
import pandas as pd
from time import time_ns

#
# directory_targ = "E:\\AMOLF_Data\\Plate_videos\\"
# suffix_data_info = time_ns()
# # plates= ["20221109_fl_04"]
# plates= ["462_20221013"]
# name_job="TEST"
#
# update_plate_info(directory_targ, local=True, suffix_data_info=suffix_data_info, strong_constraint=False)
# all_folders = get_current_folders(
#     directory_targ, local=True, suffix_data_info=suffix_data_info
# )
# print(all_folders)
directory = str(sys.argv[1])
name_job = str(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"folder": str}).iloc[i]
print(run_info)

imgs_address = directory
img_address = run_info.loc["total_path"]
print(img_address)
test_video = KymoVideoAnalysis(img_address, logging=True, vid_type='FLUO')
edge_list = test_video.edges

# print('\n To work with individual edges, here is a list of their indices:')
# for i, edge in enumerate(edge_list):
#     print('edge {}, {}'.format(i, edge))
    
# test_video.plot_extraction_img(save_img=True)
edge_objs = test_video.edge_objects

bin_nr = 2
bins = np.linspace(0, 1, bin_nr+1)
kymo = [edge_obj.extract_kymo(bounds=(0, 1)) for edge_obj in edge_objs]


