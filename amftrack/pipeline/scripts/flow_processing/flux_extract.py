from amftrack.pipeline.development.high_mag_videos.kymo_class import *
import amftrack.pipeline.development.high_mag_videos.plot_data as dataplot
import matplotlib.pyplot as plt
import sys
import pandas as pd
from amftrack.util.sys import temp_path
from amftrack.util.dbx import upload, upload_folder
import numpy as np
from tifffile import imwrite

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

if 'unique_id' in dataframe:
    test_video = Kymo_video_analysis(input_frame = dataframe, logging=True)
    img_address = dataframe['analysis_folder']
    db_address = f"{upl_targ}Analysis/{dataframe['unique_id'].split('_')[-3]}_{dataframe['unique_id'].split('_')[-2]}/{dataframe['unique_id'].split('_')[-1]}"

else:
    img_address = dataframe["address_total"]
    magnif = dataframe['magnification']
    test_video = Kymo_video_analysis(img_address, logging=True, vid_type=None,
                                     fps=None, binning=None, filter_step=[20, 70][magnif > 10],
                                     seg_thresh=12, show_seg=False)
    db_address = f"{upl_targ}Analysis/{dataframe['parent_folder']}/"

edge_list = test_video.edge_objects
target_length = int(2.1 * test_video.magnification)

test_video.plot_extraction_img(target_length=target_length, save_img=True)
edge_objs = test_video.edge_objects
# test_video.makeVideo()
print('\n To work with individual edges, here is a list of their indices:')
for i, edge in enumerate(edge_list):
    print('edge {}, {}'.format(i, edge.edge_name))

bin_nr = 1
img_seq = np.arange(len(edge_objs[0].video_analysis.selection_file))
kymos = []
edge_table = {
                'edge_name': [],
                'edge_length': [],
                'straight_length': [],
                'speed_max': [],
                'speed_min': [],
                'flux_avg': [],
                'flux_min': [],
                'flux_max': [],                
             }
data_edge = pd.DataFrame(data=edge_table)

for edge in edge_objs:
    edge_pic = edge.view_edge(img_frame=40, save_im=True, target_length=target_length)
#     edge_video = edge.view_edge(img_frame=img_seq, save_im=True, quality=6, target_length=target_length)
    space_res = edge.video_analysis.space_pixel_size
    time_res = edge.video_analysis.time_pixel_size
    video_kymos = edge.extract_multi_kymo(bin_nr, target_length=target_length, kymo_adj=False)
    kymos.append(video_kymos)
    imshow_extent = [0, edge.video_analysis.space_pixel_size * edge.kymos[0].shape[1],
                     edge.video_analysis.time_pixel_size * edge.kymos[0].shape[0], 0]
    kymos_lefts, kymos_rights = edge.fourier_kymo(bin_nr, save_im=True, save_array=False)
    speeds, times = edge.test_GST(int(GST_params[0]), w_start=3, C_thresh=float(GST_params[1]), C_thresh_falloff=float(GST_params[2]), blur_size=5, preblur=True,
                                  speed_thresh=int(GST_params[3]), plots=False)
    net_trans = edge.extract_transport(noise_thresh=0.15, plots=False, save_im=False, save_flux_array=False, margin=5)
    widths = edge.get_widths(img_frame=40, save_im=True, target_length=200)

    speed_max = np.nanpercentile(speeds.flatten(), 1)
    flux_max  = np.nanpercentile(net_trans.flatten(), 5)
    
#     data_table = {'times': times[0],
#                   'speed_right_mean': np.nanmean(speeds[0][1], axis=1),
#                   "speed_left_mean": np.nanmean(speeds[0][0], axis=1),
#                   'speed_right_std': np.nanstd(speeds[0][0], axis=1),
#                   'speed_left_std': np.nanstd(speeds[0][1], axis=1),
#                   'flux_mean': np.nanmean(net_trans, axis=1),
#                   'flux_std': np.nanstd(net_trans, axis=1),
#                   'flux_coverage': 1- np.count_nonzero(np.isnan(net_trans), axis=1) / len(net_trans[0]),
#                   'speed_left_coverage': 1 - np.count_nonzero(np.isnan(speeds[0][0]), axis=1) / len(net_trans[0]),
#                   'speed_right_coverage': 1 - np.count_nonzero(np.isnan(speeds[0][1]), axis=1) / len(net_trans[0])
#                   }
#     data_out = pd.DataFrame(data=data_table)
#     data_out.to_csv(f"{edge.edge_path}/{edge.edge_name}_data.csv")
    
#     straight_len = np.linalg.norm((edge.segments[0][0] + edge.segments[0][1])/2 - (edge.segments[-1][0] + edge.segments[-1][1])/2)*space_res
#     new_row = pd.DataFrame([{'edge_name':f'{edge.edge_name}',
#                              'edge_xpos_1': edge.video_analysis.pos[edge.edge_name[0]][0],
#                              'edge_ypos_1': edge.video_analysis.pos[edge.edge_name[0]][1],
#                              'edge_xpos_2': edge.video_analysis.pos[edge.edge_name[1]][0],
#                              'edge_ypos_2': edge.video_analysis.pos[edge.edge_name[1]][1], 
#                              'edge_length': space_res *edge.kymos[0].shape[1],
#                              'edge_width': np.mean(widths),
#                              'straight_length' : straight_len,
#                              'speed_max' : np.nanpercentile(speeds[0][1], 97),
#                              'speed_min' : np.nanpercentile(speeds[0][0], 3),
#                              'flux_avg'  : np.nanmean(net_trans),
#                              'flux_min'  : np.nanpercentile(net_trans, 3),
#                              'flux_max'  : np.nanpercentile(net_trans, 97)
#                             }])
#     data_edge = pd.concat([data_edge, new_row])
    

dataplot.plot_summary(edge_objs)
dataplot.save_raw_data(edge_objs, img_address)
    
# data_edge.to_csv(f"{img_address}/Analysis/edges_data.csv")

print(db_address)

print(f"Iteration {i}: {db_address}")
print(f"Iteration {i}: {img_address}Analysis/")

upload_folder(img_address, db_address)
