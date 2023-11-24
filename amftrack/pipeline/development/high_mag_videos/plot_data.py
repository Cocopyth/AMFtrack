import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite
from tqdm import tqdm
import scipy
import pandas as pd
import numpy as np
import matplotlib as mpl
import datetime
import re
import dropbox
from amftrack.util.dbx import load_dbx

mpl.rcParams['figure.dpi'] = 150

#TODO: Eventually make this compatible with a width profile


def save_raw_data(edge_objs, img_address, spd_max_percentile=99.9):
    if not os.path.exists(f"{img_address}/"):
        os.makedirs(f"{img_address}/")

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

    for edge in edge_objs:
        space_res = edge.space_pixel_size
        # time_res = edge.video_analysis.time_pixel_size
        # speed_max = np.nanpercentile(edge.speeds_tot.flatten(), 0.1)
        # flux_max = np.nanpercentile(edge.flux_tot.flatten(), 1)

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

        vel_adj = np.where(np.isinf(np.divide(spd_tiff[2], kymo_tiff[1])), np.nan, np.divide(spd_tiff[2], kymo_tiff[1]))
        vel_adj = np.where(abs(vel_adj) > 2 * speedmax, np.nan, vel_adj)
        vel_adj_mean = np.nanmean(vel_adj, axis=1)
        if hasattr(edge, 'video_analysis'):
            widths = edge.get_widths(img_frame=40, save_im=True, target_length=200)
            straight_len = np.linalg.norm((edge.segments[0][0] + edge.segments[0][1]) / 2 - (
                    edge.segments[-1][0] + edge.segments[-1][1]) / 2) * space_res

        data_table = {'times': edge.times[0],
                      'speed_right_mean': np.nanmean(edge.speeds_tot[0][1], axis=1),
                      "speed_left_mean": np.nanmean(edge.speeds_tot[0][0], axis=1),
                      "speed_weight_mean": vel_adj_mean,
                      'speed_right_std': np.nanstd(edge.speeds_tot[0][0], axis=1),
                      'speed_left_std': np.nanstd(edge.speeds_tot[0][1], axis=1),
                      'flux_mean': np.nanmean(edge.flux_tot, axis=1),
                      'flux_std': np.nanstd(edge.flux_tot, axis=1),
                      'flux_coverage': 1 - np.count_nonzero(np.isnan(edge.flux_tot), axis=1) / len(edge.flux_tot[0]),
                      'speed_left_coverage': 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(
                          edge.flux_tot[0]),
                      'speed_right_coverage': 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(
                          edge.flux_tot[0])
                      }
        data_out = pd.DataFrame(data=data_table)
        data_out.to_csv(f"{edge.edge_path}/{edge.edge_name}_data.csv")
        
        #data_fourier = {'speed_range' : edge.angle_plot[0],
        #                'vel_prominence' : edge.angle_plot[1]}
        #data_fourier = pd.DataFrame(data=data_fourier)
        #data_fourier.to_csv(f"{edge.edge_path}/{edge.edge_name}_fourier_data.csv")

        new_row = pd.DataFrame([{'edge_name': f'{edge.edge_name}',
                                 'edge_length': space_res * edge.kymos[0].shape[1],
                                 'speed_max': np.nanpercentile(edge.speeds_tot[0][1], 97),
                                 'speed_min': np.nanpercentile(edge.speeds_tot[0][0], 3),
                                 'speed_left': np.nanmean(np.nanmean(edge.speeds_tot[0][0], axis=1)),
                                 'speed_right': np.nanmean(np.nanmean(edge.speeds_tot[0][1], axis=1)),
                                 'speed_mean': np.nanmean(vel_adj_mean),
                                 'speed_left_std': np.nanstd(np.nanmean(edge.speeds_tot[0][0], axis=1)),
                                 'speed_right_std': np.nanstd(np.nanmean(edge.speeds_tot[0][1], axis=1)),
                                 'flux_avg': np.nanmean(edge.flux_tot),
                                 'flux_min': np.nanpercentile(edge.flux_tot, 3),
                                 'flux_max': np.nanpercentile(edge.flux_tot, 97),
                                 'coverage_left': np.mean(
                                     1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(
                                         edge.flux_tot[0])),
                                 'coverage_right': np.mean(
                                     1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(
                                         edge.flux_tot[0])),
                                 'coverage_tot': np.mean(
                                     1 - np.count_nonzero(np.isnan(edge.flux_tot), axis=1) / len(edge.flux_tot[0]))}])
                                # 'fourier_speeds' : edge.fourier_peak_data[0]}])
        if hasattr(edge, 'video_analysis'):
            new_row['edge_width']       = np.mean(widths)
            new_row['straight_length']  = straight_len
            new_row['edge_xpos_1']      = edge.video_analysis.pos[edge.edge_name[0]][0]
            new_row['edge_ypos_1']      = edge.video_analysis.pos[edge.edge_name[0]][1]
            new_row['edge_xpos_2']      = edge.video_analysis.pos[edge.edge_name[1]][0]
            new_row['edge_ypos_2']      = edge.video_analysis.pos[edge.edge_name[1]][1]
        data_edge = pd.concat([data_edge, new_row])

    data_edge.to_csv(f"{img_address}/edges_data.csv")


def plot_summary(edge_objs, spd_max_percentile=99.5):
    for edge in edge_objs:
        space_res = edge.space_pixel_size
        time_res = edge.time_pixel_size
        # speed_max = np.nanpercentile(edge.speeds_tot.flatten(), 0.1)
        # flux_max = np.nanpercentile(edge.flux_tot.flatten(), 1)

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

        speedmax = np.max([np.nanpercentile(abs(spd_tiff[0:2].flatten()), spd_max_percentile), 5])

        vel_adj = np.where(np.isinf(np.divide(spd_tiff[2], kymo_tiff[1])), np.nan, np.divide(spd_tiff[2], kymo_tiff[1]))
        vel_adj = np.where(abs(vel_adj) > 2 * speedmax, np.nan, vel_adj)
        vel_adj_mean = np.nanmean(vel_adj, axis=1)

        speed_bins = np.linspace(-speedmax, speedmax, 1001)
        speed_bins_trunc = (abs(speed_bins) < 7.0)[:-1]
        speed_histo_left = np.array([np.histogram(row, speed_bins)[0] for row in edge.speeds_tot[0][0]])
        speed_histo_right = np.array([np.histogram(row, speed_bins)[0] for row in edge.speeds_tot[0][1]])
        speed_histo = (speed_histo_left + speed_histo_right) / (2 * len(edge.speeds_tot[0][0][0]))

        fig, ax = plt.subplot_mosaic([['kymo', 'speed_hist_zoom', 'speed_hist'],
                                      ['speed_plot', 'flux_plot', 'speed_hist']], figsize=(12, 8), layout='constrained')

        fig.suptitle(f"Edge {edge.edge_name} Summary")
        imshow_extent = [0, space_res * edge.kymos[0].shape[1],
                         time_res * edge.kymos[0].shape[0], 0]
        ax['kymo'].imshow(edge.kymos[0], extent=imshow_extent, aspect='auto')
        #     ax[0].set_title(f"Full kymo (length = {space_res * len(edge.kymos[0]):.5} $ \mu m$)")
        ax['kymo'].set_ylabel("time (s)")
        ax['kymo'].set_xlabel("space ($\mu m$)")
        ax['kymo'].set_title("Kymograph")
        ax['speed_plot'].plot(edge.times[0], np.nanmean(edge.speeds_tot[0][0], axis=1), c='tab:blue', label='To root')
        ax['speed_plot'].fill_between(edge.times[0],
                                      np.nanmean(edge.speeds_tot[0][0], axis=1) + np.nanstd(edge.speeds_tot[0][0],
                                                                                            axis=1),
                                      np.nanmean(edge.speeds_tot[0][0], axis=1) - np.nanstd(edge.speeds_tot[0][0],
                                                                                            axis=1),
                                      alpha=0.5, facecolor='tab:blue')
        ax['speed_plot'].plot(edge.times[0], np.nanmean(edge.speeds_tot[0][1], axis=1), c='tab:orange', label='To tip')
        ax['speed_plot'].fill_between(edge.times[0],
                                      np.nanmean(edge.speeds_tot[0][1], axis=1) + np.nanstd(edge.speeds_tot[0][1],
                                                                                            axis=1),
                                      np.nanmean(edge.speeds_tot[0][1], axis=1) - np.nanstd(edge.speeds_tot[0][1],
                                                                                            axis=1),
                                      alpha=0.5, facecolor='tab:orange')
        ax['speed_plot'].plot(edge.times[0], vel_adj_mean, c='black', alpha=0.5, label='effMean')
        ax['speed_plot'].set_title("Speed plots")
        ax['speed_plot'].set_xlabel("time (s)")
        ax['speed_plot'].set_ylabel("speed ($\mu m/s$)")
        ax['speed_plot'].grid(True)
        ax['speed_plot'].set_ylim([-speedmax, speedmax])
        ax['speed_plot'].legend()

        hist_cmap = 'magma'
        hist_cmap = 'gist_stern'

        ax['speed_hist'].imshow(speed_histo.T, extent=[0, len(speed_histo) * time_res, -speedmax, speedmax], origin='lower',
                                aspect='auto', cmap=hist_cmap)
        ax['speed_hist'].axhline(c='w', linestyle='--')
        ax['speed_hist'].set_title(f"Velocity histogram")
        ax['speed_hist'].set_xlabel("time (s)")
        ax['speed_hist'].set_ylabel("speed ($\mu m/s$)")

        ax['speed_hist_zoom'].imshow(speed_histo.T[speed_bins_trunc], extent=[0, len(speed_histo) * time_res, -7, 7],
                                     origin='lower', aspect='auto', cmap=hist_cmap)
        ax['speed_hist_zoom'].axhline(c='w', linestyle='--')
        ax['speed_hist_zoom'].set_title(f"Velocity histogram")
        ax['speed_hist_zoom'].set_xlabel("time (s)")
        ax['speed_hist_zoom'].set_ylabel("speed ($\mu m/s$)")

        ax['flux_plot'].plot(edge.times[0], np.nanmean(edge.flux_tot, axis=1), c='black', label='Average flux')
        ax['flux_plot'].fill_between(edge.times[0],
                                     np.nanmean(edge.flux_tot, axis=1) + np.nanstd(edge.flux_tot, axis=1),
                                     np.nanmean(edge.flux_tot, axis=1) - np.nanstd(edge.flux_tot, axis=1),
                                     alpha=0.5, facecolor='black')
        ax['flux_plot'].set_title("Flux plot")
        ax['flux_plot'].set_xlabel("time (s)")
        ax['flux_plot'].set_ylabel("flux ($q\mu m/s$)")
        ax['flux_plot'].grid(True)

        fig.savefig(f"{edge.edge_path}/{edge.edge_name}_summary.png")
        plt.close(fig)

        fig, ax = plt.subplot_mosaic([['kymo', 'kymo_left', 'kymo_right'],
                                      ['kymo_stat', 'spd_left', 'spd_right']], figsize=(12, 9), layout='constrained')
        ax['kymo'].imshow(kymo_tiff[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax['kymo_stat'].imshow(kymo_tiff[1], cmap='gray', aspect='auto', extent=imshow_extent)
        ax['kymo_right'].imshow(kymo_tiff[2], cmap='gray', aspect='auto', extent=imshow_extent)
        ax['kymo_left'].imshow(kymo_tiff[3], cmap='gray', aspect='auto', extent=imshow_extent)
        ax['spd_left'].imshow(spd_tiff[0], cmap='coolwarm', vmin=-speedmax, vmax=speedmax, aspect='auto',
                              extent=imshow_extent)
        spd_colors = ax['spd_right'].imshow(spd_tiff[1], cmap='coolwarm', vmin=-speedmax, vmax=speedmax, aspect='auto',
                                            extent=imshow_extent)

        ax['kymo'].set_title("Unfiltered kymograph")
        ax['kymo_stat'].set_title("Kymograph (no static)")
        ax['kymo_right'].set_title("Right kymograph")
        ax['kymo_left'].set_title("Left kymograph")
        ax['spd_left'].set_title("Speed field left")
        ax['spd_right'].set_title("Speed field right")
        cbar = fig.colorbar(spd_colors, location='right')
        cbar.ax.set_ylabel('speed ($\mu m / s$)', rotation=270)
        for axis in ax:
            ax[axis].set_xlabel("space $(\mu m)$")
            ax[axis].set_ylabel("time (s)")
        #         fig.tight_layout()
        fig.savefig(f"{edge.edge_path}{os.sep}{edge.edge_name}_kymos.png")
        plt.close(fig)
        
#         peaks, _ = edge.fourier_peak_data
        
#         fig, ax = plt.subplot_mosaic([['fft', 'polarfft'], ['angle_plot', 'angle_plot']], figsize=(10, 6))
        
#         ax['fft'].imshow(edge.ftabsimage, aspect='auto')
#         ax['fft'].set_xlabel('u')
#         ax['fft'].set_ylabel('v')
#         ax['fft'].set_title("Fourier Transform")

#         ax['polarfft'].imshow(edge.ftpolarimage, aspect='auto', extent=[0, 1, 0, np.pi])
#         ax['polarfft'].set_xlabel("radius")
#         ax['polarfft'].set_ylabel("orientation (rad)")
#         ax['polarfft'].set_title("Polar representation")

#         ax['angle_plot'].plot(edge.angle_plot[0], edge.angle_plot[1])
#         ax['angle_plot'].set_xlabel("Velocity $(\mu m / s)$")
#         ax['angle_plot'].set_ylabel("Prominence")
#         ax['angle_plot'].plot(edge.angle_plot[0][peaks], edge.angle_plot[1][peaks], "xr")
#         ax['angle_plot'].set_title(f"Angle intensity with found peaks {[round(speed, 4) for speed in edge.fourier_speeds]} $\mu m / s$")
#         ax['angle_plot'].set_xlim([-15, 15])
#         fig.suptitle(f"{edge.edge_name} Fourier analysis")
#         fig.tight_layout()
#         fig.savefig(f"{edge.edge_path}{os.sep}{edge.edge_name}_fourier.png")


# def delete_dropbox_folders(dropbox, path):
#     print("found updated function")
# #     dbx=load_dbx()
#     print(os.walk(path, topdown=False))
#     for root, dirs, files in os.walk(path, topdown=False):
#         print(dirs)
#         for name in dirs:
# #             print(name)
#             if "edge" in str(name):
# #                 print("succes!")
#                 folder_path=os.path.join(root,name)
#                 try:
#                     dbx.files_delete_v2(folder_path)
#                     print(f"Deleted folder: {folder_path}")
#                 except dbx.exceptions.ApiError as e:
#                     print(f"Error deleting folder: {folder_path}. {e}")
                
                
def delete_dropbox_folders(path):
    dbx=load_dbx()
    print(path)
    for entry in dbx.files_list_folder(path, recursive=False).entries:
#         metadata=entry.get_metadata
        if isinstance(entry, dropbox.files.FolderMetadata) and "edge" in entry.name:
            try:
                dbx.files_delete_v2(entry.path_display)
                print(f"Deleted folder: {entry.name}")
            except dropbox.exceptions.ApiError as e:
                print(f"Error deleting folder {entry.name}: {e}")





