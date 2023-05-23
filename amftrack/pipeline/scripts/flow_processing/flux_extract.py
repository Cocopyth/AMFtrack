from amftrack.pipeline.development.high_mag_videos.kymo_class import *
# import os
# import imageio.v2 as imageio
import matplotlib.pyplot as plt
# import cv2
# from tqdm import tqdm
#
# from amftrack.pipeline.functions.image_processing.extract_graph import (
#     from_sparse_to_graph,
#     generate_nx_graph,
#     clean_degree_4,
# )
# import scipy
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

dataframe = pd.read_json(f"{temp_path}/{op_id}.json")
dataframe = dataframe.iloc[i]

img_address = dataframe["address_total"]

print(upl_targ)

test_video = Kymo_video_analysis(img_address, logging=True, vid_type=None,
                                 fps=None, binning=None, filter_step=80,
                                 seg_thresh=10, show_seg=False)
edge_list = test_video.edge_objects
target_length = int(2.5 * test_video.magnification)

test_video.plot_extraction_img(target_length=target_length, save_img=True)
edge_objs = test_video.edge_objects
# test_video.makeVideo()
print('\n To work with individual edges, here is a list of their indices:')
for i, edge in enumerate(edge_list):
    print('edge {}, {}'.format(i, edge.edge_name))

bin_nr = 1
img_seq = np.arange(len(edge_objs[0].video_analysis.selection_file))
kymos = []

for edge in edge_objs:
    edge_pic = edge.view_edge(img_frame=40, save_im=True, target_length=target_length)
    edge_video = edge.view_edge(img_frame=img_seq, save_im=True, quality=6, target_length=target_length)
    space_res = edge.video_analysis.space_pixel_size
    time_res = edge.video_analysis.time_pixel_size
    video_kymos = edge.extract_multi_kymo(bin_nr, target_length=target_length, kymo_adj=False)
    kymos.append(video_kymos)
    imshow_extent = [0, edge.video_analysis.space_pixel_size * edge.kymos[0].shape[1],
                     edge.video_analysis.time_pixel_size * edge.kymos[0].shape[0], 0]
    kymos_lefts, kymos_rights = edge.fourier_kymo(bin_nr, save_im=True, save_array=False)
    speeds, times = edge.test_GST(int(GST_params[0]), w_start=3, C_thresh=float(GST_params[1]), C_thresh_falloff=float(GST_params[2]), blur_size=5, preblur=True,
                                  speed_thresh=int(GST_params[3]), plots=False)
    net_trans = edge.extract_transport(noise_thresh=0.15, plots=False, save_im=True, save_flux_array=True, margin=5)

    data_table = {'times': times[0],
                  'speed_right_mean': np.nanmean(speeds[0][1], axis=1),
                  "speed_left_mean": np.nanmean(speeds[0][0], axis=1),
                  'speed_right_std': np.nanstd(speeds[0][0], axis=1),
                  'speed_left_std': np.nanstd(speeds[0][1], axis=1),
                  'flux_mean': np.nanmean(net_trans, axis=1),
                  'flux_std': np.nanstd(net_trans, axis=1),
                  'flux_coverage': 1- np.count_nonzero(np.isnan(net_trans), axis=1) / len(net_trans[0]),
                  'speed_left_coverage': 1 - np.count_nonzero(np.isnan(speeds[0][0]), axis=1) / len(net_trans[0]),
                  'speed_right_coverage': 1 - np.count_nonzero(np.isnan(speeds[0][1]), axis=1) / len(net_trans[0])
                  }
    data_out = pd.DataFrame(data=data_table)
    data_out.to_csv(f"{edge.edge_path}/{edge.edge_name}_data.csv")
    
    
    fig, ax = plt.subplots(3, figsize=(7, 7 * 3))
    imshow_extent = [0, space_res * video_kymos[0].shape[1],
                     time_res * video_kymos[0].shape[0], 0]

    ax[0].imshow(video_kymos[0], extent=imshow_extent, aspect='auto')
    ax[0].set_title("Full kymo")
    ax[1].errorbar(times[0], np.nanmean(speeds[0][0], axis=1), np.nanstd(speeds[0][0], axis=1), c='tab:blue',
                   label='Full kymo backward', errorevery=10)
    ax[1].errorbar(times[0], np.nanmean(speeds[0][1], axis=1), np.nanstd(speeds[0][1], axis=1), c='tab:orange',
                   label='Full kymo forward', errorevery=10)
    ax[1].set_title(f"Full mean speed")
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("speed ($\mu m/s$)")
    ax[1].legend()
    ax[1].grid(True)
    ax[2].errorbar(times[0], np.nanmean(net_trans, axis=1), np.nanstd(net_trans, axis=1), label='Full kymo',
                   errorevery=10)
    ax[2].set_title(f"Full mean flux")
    ax[2].set_xlabel("time (s)")
    ax[2].set_ylabel("flux ($q \mu m / s$)")
    ax[2].legend()
    ax[2].grid(True)
    fig.tight_layout()
    fig.savefig(f"{edge.edge_path}/{edge.edge_name}_summary.png")
    
    
    fig, ax = plt.subplots(2,2, figsize=(9,9))
    ax[0][0].imshow(video_kymos[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
    ax[0][1].imshow((kymos_lefts[0]+ kymos_rights[0])/2, cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
    ax[1][0].imshow(kymos_lefts[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
    ax[1][1].imshow(kymos_rights[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)

    ax[0][0].set_title("Unfiltered kymograph")
    ax[0][0].set_xlabel("space $(\mu m)$")
    ax[0][0].set_ylabel("time (s)")
    ax[0][1].set_title("Kymograph (no static)")
    ax[0][1].set_xlabel("space $(\mu m)$")
    ax[0][1].set_ylabel("time (s)")
    ax[1][0].set_title("Right kymograph")
    ax[1][0].set_xlabel("space $(\mu m)$")
    ax[1][0].set_ylabel("time (s)")
    ax[1][1].set_title("Left kymograph")
    ax[1][1].set_xlabel("space $(\mu m)$")
    ax[1][1].set_ylabel("time (s)")
    fig.tight_layout()
    fig.savefig(f"{edge.edge_path}/{edge.edge_name}_kymos.png")
    
    kymo_tiff = np.array([video_kymos[0],
                         kymos_lefts[0]+ kymos_rights[0],
                         kymos_lefts[0],
                         kymos_rights[0]], dtype=np.int16)
    imwrite(f"{edge.edge_path}/{edge.edge_name}_kymos_array.tiff", kymo_tiff, photometric='minisblack')
    
    
    speedmax = np.nanmax(abs(np.array([speeds[0][0], speeds[0][1]])))
    fig, ax = plt.subplots(2, 2, figsize=(9,9), layout='constrained')
    ax[0][0].imshow(speeds[0][0], cmap='coolwarm', aspect='auto', vmax=speedmax, vmin=-speedmax, extent=imshow_extent)
    cbar = ax[0][1].imshow(speeds[0][1], cmap='coolwarm', aspect='auto', vmax=speedmax, vmin=-speedmax, extent=imshow_extent)
    fig.colorbar(cbar, ax=ax[0,:])
    cbar2 = ax[1][1].imshow(net_trans, cmap='coolwarm', aspect='auto', extent=imshow_extent)
    fig.colorbar(cbar2, ax=ax[1][1])
    ax[1][0].plot(times[0], 1 - np.count_nonzero(np.isnan(speeds[0][0]), axis=1) / len(net_trans[0]), label="Left")
    ax[1][0].plot(times[0], 1 - np.count_nonzero(np.isnan(speeds[0][1]), axis=1) / len(net_trans[0]), label="Right")
    ax[1][0].plot(times[0], 1 - np.count_nonzero(np.isnan(net_trans), axis=1) / len(net_trans[0]), label="Total")

    ax[1][0].set_ylim([0, 1])
    ax[1][0].legend()
    ax[1][0].grid(True)
    ax[1][0].set_title("Speed coverage")
    ax[1][0].set_xlabel("time (s)")
    ax[1][0].set_ylabel("Hypha length coverage (%)")

    ax[0][0].set_title("Speeds left ($\mu m / s$)")
    ax[0][0].set_xlabel("space $(\mu m)$")
    ax[0][0].set_ylabel("time (s)")
    ax[0][1].set_title("Speeds right ($\mu m / s$)")
    ax[0][1].set_xlabel("space $(\mu m)$")
    ax[0][1].set_ylabel("time (s)")
    ax[1][1].set_title("Flux ($q\mu m / s$)")
    ax[1][1].set_xlabel("space $(\mu m)$")
    ax[1][1].set_ylabel("time (s)")
    fig.savefig(f"{edge.edge_path}/{edge.edge_name}_speeds.png")
    
    spds_tiff = np.array([
        speeds[0][0],
        speeds[0][1],
        net_trans
    ], dtype=float)
    imwrite(f"{edge.edge_path}/{edge.edge_name}_speeds_flux_array.tiff", spds_tiff, photometric='minisblack')



db_address = upl_targ + str(test_video.video_nr) + '/'

print(f"Iteration {i}: {db_address}")
print(f"Iteration {i}: {img_address}/Analysis/")

upload_folder(img_address + '/Analysis/', db_address)
