from amftrack.pipeline.development.high_mag_videos.kymo_class import *
import amftrack.pipeline.development.high_mag_videos.plot_data as dataplot
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
# print(i)
# print(dataframe[dataframe['xpos']==dataframe['xpos']])
#the selection frame is used to segment fluorescence videos based on a brightfield image of the same position
# selection_frame = dataframe[dataframe['ypos']==dataframe['ypos'].iloc[i]]
# selection_frame = selection_frame[selection_frame['xpos']==dataframe['xpos'].iloc[i]]
# selection_frame = selection_frame[selection_frame['mode']=='BF']
#when analysing older data (Hannah or Rachael) we don't need to give BF segmentation for fluorescence videos
selection_frame = pd.DataFrame()

# if len(selection_frame)>1:
#     selection_frame = selection_frame.iloc[0]
dataframe = dataframe.iloc[i]

if 'unique_id' in dataframe:
    drop_targ = os.path.relpath(f"/{dataframe['tot_path_drop']}", upl_targ)
    
    test_video = KymoVideoAnalysis(input_frame = dataframe, samepos_frame = selection_frame, logging=True)
    img_address = dataframe['analysis_folder']
    db_address = f"{upl_targ}KymoSpeeDExtract/{drop_targ}"
    print(f"HELLLO!!! {db_address}")

else:
    img_address = dataframe["address_total"]
    magnif = dataframe['magnification']
    test_video = KymoVideoAnalysis(img_address, samepos_frame = selection_frame, logging=True, vid_type=None,
                                   fps=None, binning=None,
                                   filter_step=[20, 50][magnif > 10],seg_thresh=12,
                                   show_seg=False,
                                   close_size = 200,
                                  thresh_adjust=-2)
    db_address = f"{upl_targ}KymoSpeeDExtract/{dataframe['parent_folder']}/"

    

    
target_length = int(2.4 * test_video.magnification)

edge_objs = test_video.edge_objects
test_video.makeVideo()

bin_nr = 1
img_seq = np.arange(len(edge_objs[0].video_analysis.selection_file))

for edge in edge_objs:
    edge.view_edge(img_frame=0, save_im=True, target_length=target_length)
    edge.view_edge(img_frame=img_seq, save_im=True, quality=6, target_length=target_length)
    edge.extract_multi_kymo(bin_nr, target_length=target_length, kymo_adj=False, kymo_normalize=True)
    edge.fourier_kymo(return_self=False)
    edge.extract_speeds(int(GST_params[0]), w_start=3, C_thresh=float(GST_params[1]), C_thresh_falloff=float(GST_params[2]), blur_size=3, preblur=True, speed_thresh=int(GST_params[3]))
    edge.extract_transport()
    edge.video_analysis.plot_speed_arrows(plot_flux=True, save_im=True, video_txt_size=40)

test_video.plot_extraction_img(target_length=target_length, save_img=True)
    
dataplot.plot_summary(edge_objs)
dataplot.save_raw_data(edge_objs, img_address)

print(db_address)

# this next function deletes all folders inside the given dropbox folder
# for folder in img_address:
#     delete folder
# print(f"Folder deleted: {db_address}")
# if os.path.exists(f"{db_address}/"):
#     print("it does exist, why did it not delete them?")
dataplot.delete_dropbox_folders(db_address)

print(f"Iteration {i}: {db_address}")
print(f"Iteration {i}: {img_address}")

upload_folder(img_address, db_address, delete=False)
