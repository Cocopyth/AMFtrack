from amftrack.pipeline.development.high_mag_videos.kymo_class import *
import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
import sys
import pandas as pd
from amftrack.util.sys import temp_path


directory = str(sys.argv[1])
test_len = str(sys.argv[2])
folders = str(sys.argv[3])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])

dataframe = pd.read_json(f"{temp_path}/{op_id}.json")
dataframe = dataframe.iloc[i]

img_address = dataframe["address_total"]

print(dataframe)

test_video = Kymo_video_analysis(img_address, logging=True, vid_type=None, 
                                 fps=None, binning=None, filter_step=80,
                                seg_thresh=10, show_seg=False)
edge_list = test_video.edge_objects
target_length = int(2.5*test_video.magnification)

test_video.plot_extraction_img(target_length=target_length, save_img=True)
edge_objs = test_video.edge_objects
# test_video.makeVideo()
print('\n To work with individual edges, here is a list of their indices:')
for i, edge in enumerate(edge_list):
    print('edge {}, {}'.format(i, edge.edge_name))
    
bin_nr = 1
img_seq    = np.arange(len(edge_objs[0].video_analysis.selection_file))
kymos = []



for edge in edge_objs:
    edge_pic   = edge.view_edge(img_frame=40 ,save_im=True, target_length = target_length)
    edge_video = edge.view_edge(img_frame = img_seq, save_im=True, quality = 6, target_length=target_length)
    space_res = edge.video_analysis.space_pixel_size
    time_res = edge.video_analysis.time_pixel_size
    video_kymos = edge.extract_multi_kymo(bin_nr, target_length=target_length, kymo_adj=False)
    kymos.append(video_kymos)
    imshow_extent = [0, edge.video_analysis.space_pixel_size * edge.kymos[0].shape[1],
                                  edge.video_analysis.time_pixel_size * edge.kymos[0].shape[0], 0]
    kymos_lefts, kymos_rights = edge.fourier_kymo(bin_nr, save_im=True, save_array=True)
    speeds, times = edge.test_GST(15, w_start=3, C_thresh=0.95, C_thresh_falloff = 0.001, blur_size = 5, preblur=True, speed_thresh=20, plots=False)    
    net_trans = edge.extract_transport(noise_thresh=0.15, plots=False, save_im=True, save_flux_array=True, margin=5)    


print(dataframe)