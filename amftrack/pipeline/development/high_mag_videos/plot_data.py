import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite
from tqdm import tqdm
import scipy
import pandas as pd
import numpy as np

def save_raw_data(edge_objs, img_address, spd_max_percentile = 99.9):
    if not os.path.exists(f"{img_address}/Analysis/"):
        os.makedirs(f"{img_address}/Analysis/")
    
    for edge in edge_objs:
        
        space_res = edge.video_analysis.space_pixel_size
        time_res  = edge.video_analysis.time_pixel_size        
        speed_max = np.nanpercentile(edge.speeds_tot.flatten(), 0.1)
        flux_max  = np.nanpercentile(edge.flux_tot.flatten(), 1)
        
        kymo_tiff = np.array([edge.kymos[0],
             edge.filtered_left[0] + edge.filtered_right[0],
             edge.filtered_left[0],
             edge.filtered_right[0]], dtype=np.int16)
        imwrite(f"{edge.edge_path}/{edge.edge_name}_kymos_array.tiff", kymo_tiff, photometric='minisblack')

        spd_tiff = np.array([
            edge.speeds_tot[0][0],
            edge.speeds_tot[0][1],
            edge.flux_tot
        ], dtype=float)
        imwrite(f"{edge.edge_path}/{edge.edge_name}_speeds_flux_array.tiff", spd_tiff, photometric='minisblack')
        speedmax = np.nanpercentile(abs(spd_tiff[0:2].flatten()), spd_max_percentile)
        
        vel_adj = np.where(np.isinf(np.divide(spd_tiff[2] , kymo_tiff[1])), np.nan, np.divide(spd_tiff[2] , kymo_tiff[1]))
        vel_adj = np.where(abs(vel_adj) > 2*speedmax, np.nan, vel_adj)
        vel_adj_mean = np.nanmean(vel_adj, axis=1)
        

        edge_table = {
                'edge_name': [],
                'edge_length': [],
                'straight_length': [],
                'speed_max': [],
                'speed_min': [],
                'speed_mean': [],
                'flux_avg': [],
                'flux_min': [],
                'flux_max': [],                
             }
        data_edge = pd.DataFrame(data=edge_table)
        
        data_table = {'times': edge.times[0],
                      'speed_right_mean': np.nanmean(edge.speeds_tot[0][1], axis=1),
                      "speed_left_mean": np.nanmean(edge.speeds_tot[0][0], axis=1),
                      "speed_weight_mean": vel_adj_mean,
                      'speed_right_std': np.nanstd(edge.speeds_tot[0][0], axis=1),
                      'speed_left_std': np.nanstd(edge.speeds_tot[0][1], axis=1),
                      'flux_mean': np.nanmean(edge.flux_tot, axis=1),
                      'flux_std': np.nanstd(edge.flux_tot, axis=1),
                      'flux_coverage': 1- np.count_nonzero(np.isnan(edge.flux_tot), axis=1) / len(edge.flux_tot[0]),
                      'speed_left_coverage': 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(edge.flux_tot[0]),
                      'speed_right_coverage': 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(edge.flux_tot[0])
                      }
        data_out = pd.DataFrame(data=data_table)
        data_out.to_csv(f"{edge.edge_path}/{edge.edge_name}_data.csv")

        straight_len = np.linalg.norm((edge.segments[0][0] + edge.segments[0][1])/2 - (edge.segments[-1][0] + edge.segments[-1][1])/2)*space_res
        new_row = pd.DataFrame([{'edge_name':f'{edge.edge_name}', 
                                 'edge_length': space_res *edge.kymos[0].shape[1],
                                 'straight_length' : straight_len,
                                 'speed_max' : np.nanpercentile(edge.speeds_tot[0][1], 97),
                                 'speed_min' : np.nanpercentile(edge.speeds_tot[0][0], 3),
                                 'speed_mean': np.nanmean(vel_adj_mean),
                                 'flux_avg'  : np.nanmean(edge.flux_tot),
                                 'flux_min'  : np.nanpercentile(edge.flux_tot, 3),
                                 'flux_max'  : np.nanpercentile(edge.flux_tot, 97)
                                }])
        data_edge = pd.concat([data_edge, new_row])

        

        data_edge.to_csv(f"{img_address}/Analysis/edges_data.csv")


    

def plot_summary(edge_objs, spd_max_percentile = 99.9):
    for edge in edge_objs:    
        space_res = edge.video_analysis.space_pixel_size
        time_res  = edge.video_analysis.time_pixel_size        
        speed_max = np.nanpercentile(edge.speeds_tot.flatten(), 0.1)
        flux_max  = np.nanpercentile(edge.flux_tot.flatten(), 1)
        
        kymo_tiff = np.array([edge.kymos[0],
             edge.filtered_left[0] + edge.filtered_right[0],
             edge.filtered_left[0],
             edge.filtered_right[0]], dtype=np.int16)
        kymo_tiff[1] = np.divide(kymo_tiff[1], 2.0)

        spd_tiff = np.array([
            edge.speeds_tot[0][0],
            edge.speeds_tot[0][1],
            edge.flux_tot
        ], dtype=float)
        
        speedmax = np.nanpercentile(abs(spd_tiff[0:2].flatten()), spd_max_percentile)
        
        back_thresh, forw_thresh = (edge.filtered_right[0], edge.filtered_left[0])
        speed_weight_left = np.nansum(np.prod((edge.speeds_tot[0][0], back_thresh), 0), 1) / np.nansum(back_thresh, axis=1)
        speed_weight_right = np.nansum(np.prod((edge.speeds_tot[0][1], forw_thresh), 0), 1) / np.nansum(forw_thresh, axis=1)
       
        speed_left_coverage = 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(edge.flux_tot[0])
        speed_right_coverage= 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(edge.flux_tot[0])
        coverage_sum = speed_left_coverage + speed_right_coverage
        
        vel_adj = np.where(np.isinf(np.divide(spd_tiff[2] , kymo_tiff[1])), np.nan, np.divide(spd_tiff[2] , kymo_tiff[1]))
        vel_adj = np.where(abs(vel_adj) > 2*speedmax, np.nan, vel_adj)
        vel_adj_mean = np.nanmean(vel_adj, axis=1)
        
        fig, ax = plt.subplots(2,2, figsize=(9,9), layout='constrained')

        fig.suptitle(f"Edge {edge.edge_name} Summary")
        imshow_extent = [0, space_res * edge.kymos[0].shape[1],
                         time_res * edge.kymos[0].shape[0], 0]
        ax[0][0].imshow(edge.kymos[0], extent=imshow_extent, aspect='auto')
    #     ax[0].set_title(f"Full kymo (length = {space_res * len(edge.kymos[0]):.5} $ \mu m$)")
        ax[0][0].set_ylabel("time (s)")
        ax[0][0].set_xlabel("space ($\mu m$)")
        ax[0][0].set_title("Kymograph")
        ax[1][1].plot(edge.times[0], np.nanmean(edge.speeds_tot[0][0], axis=1), c='tab:blue', label='Rightward')
        ax[1][1].fill_between(edge.times[0], 
                              np.nanmean(edge.speeds_tot[0][0], axis=1) + np.nanstd(edge.speeds_tot[0][0], axis=1), 
                              np.nanmean(edge.speeds_tot[0][0], axis=1) - np.nanstd(edge.speeds_tot[0][0], axis=1), 
                              alpha=0.5, facecolor='tab:blue')
        ax[1][1].plot(edge.times[0], np.nanmean(edge.speeds_tot[0][1], axis=1), c='tab:orange', label='Leftward')
        ax[1][1].fill_between(edge.times[0], 
                              np.nanmean(edge.speeds_tot[0][1], axis=1) + np.nanstd(edge.speeds_tot[0][1], axis=1), 
                              np.nanmean(edge.speeds_tot[0][1], axis=1) - np.nanstd(edge.speeds_tot[0][1], axis=1), 
                              alpha=0.5, facecolor='tab:orange')
        ax[1][1].axhline()
        ax[1][1].set_ylim([-speedmax,speedmax])
        ax[1][1].set_title(f"Split speeds")
        ax[1][1].set_xlabel("time (s)")
        ax[1][1].set_ylabel("speed ($\mu m/s$)")
        ax[1][1].legend()
        ax[1][1].grid(True)
        spd_colors = ax[0][1].imshow(vel_adj, aspect='auto', vmin = -speedmax, vmax = speedmax, extent = imshow_extent, cmap = 'coolwarm')
        ax[0][1].set_ylabel("time (s)")
        ax[0][1].set_xlabel("space ($\mu m$)")
        ax[0][1].set_title("Speed Field")
        cbar = fig.colorbar(spd_colors, location='right')
        cbar.ax.set_ylabel('speed ($\mu m / s$)', rotation=270)

        

        ax[1][0].plot(edge.times[0], vel_adj_mean, c='tab:blue')
        ax[1][0].set_title("Weighted mean speed")
        ax[1][0].set_xlabel("time (s)")
        ax[1][0].set_ylabel("speed ($\mu m/s$)")
        ax[1][0].grid(True)
        ax[1][0].set_ylim([-speedmax, speedmax])

#         fig.tight_layout()
        fig.savefig(f"{edge.edge_path}/{edge.edge_name}_summary.png")
    
        fig, ax = plt.subplots(2,2, figsize=(9,9))
        ax[0][0].imshow(kymo_tiff[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax[0][1].imshow(kymo_tiff[1], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax[1][0].imshow(kymo_tiff[2], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax[1][1].imshow(kymo_tiff[3], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)

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