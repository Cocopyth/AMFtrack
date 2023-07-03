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
mpl.rcParams['figure.dpi'] = 150


def save_raw_data(edge_objs, img_address, spd_max_percentile = 99.9):
    if not os.path.exists(f"{img_address}/Analysis/"):
        os.makedirs(f"{img_address}/Analysis/")
    
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
        widths = edge.get_widths(img_frame=40, save_im=True, target_length=200)

        
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
                                 'edge_width': np.mean(widths),
                                 'speed_max' : np.nanpercentile(edge.speeds_tot[0][1], 97),
                                 'speed_min' : np.nanpercentile(edge.speeds_tot[0][0], 3),
                                 'speed_left': np.nanmean(np.nanmean(edge.speeds_tot[0][0], axis=1)),
                                 'speed_right': np.nanmean(np.nanmean(edge.speeds_tot[0][1], axis=1)),
                                 'speed_mean': np.nanmean(vel_adj_mean),
                                 'speed_left_std' : np.nanstd(np.nanmean(edge.speeds_tot[0][0], axis=1)),
                                 'speed_right_std' : np.nanstd(np.nanmean(edge.speeds_tot[0][1], axis=1)),
                                 'flux_avg'  : np.nanmean(edge.flux_tot),
                                 'flux_min'  : np.nanpercentile(edge.flux_tot, 3),
                                 'flux_max'  : np.nanpercentile(edge.flux_tot, 97),
                                 'coverage_left' : np.mean(1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(edge.flux_tot[0])),
                                 'coverage_right' : np.mean(1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(edge.flux_tot[0])),
                                 'coverage_tot' : np.mean(1- np.count_nonzero(np.isnan(edge.flux_tot), axis=1) / len(edge.flux_tot[0])),
                                 'edge_xpos_1': edge.video_analysis.pos[edge.edge_name[0]][0],
                                 'edge_ypos_1': edge.video_analysis.pos[edge.edge_name[0]][1],
                                 'edge_xpos_2': edge.video_analysis.pos[edge.edge_name[1]][0],
                                 'edge_ypos_2': edge.video_analysis.pos[edge.edge_name[1]][1]
                                }])
        data_edge = pd.concat([data_edge, new_row])

        

    data_edge.to_csv(f"{img_address}/Analysis/edges_data.csv")


    

def plot_summary(edge_objs, spd_max_percentile = 99.5):
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
        
        speedmax = np.max([np.nanpercentile(abs(spd_tiff[0:2].flatten()), spd_max_percentile), 15])
        
        back_thresh, forw_thresh = (edge.filtered_right[0], edge.filtered_left[0])
        speed_weight_left = np.nansum(np.prod((edge.speeds_tot[0][0], back_thresh), 0), 1) / np.nansum(back_thresh, axis=1)
        speed_weight_right = np.nansum(np.prod((edge.speeds_tot[0][1], forw_thresh), 0), 1) / np.nansum(forw_thresh, axis=1)
       
        speed_left_coverage = 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][0]), axis=1) / len(edge.flux_tot[0])
        speed_right_coverage= 1 - np.count_nonzero(np.isnan(edge.speeds_tot[0][1]), axis=1) / len(edge.flux_tot[0])
        coverage_sum = speed_left_coverage + speed_right_coverage
        
        vel_adj = np.where(np.isinf(np.divide(spd_tiff[2] , kymo_tiff[1])), np.nan, np.divide(spd_tiff[2] , kymo_tiff[1]))
        vel_adj = np.where(abs(vel_adj) > 2*speedmax, np.nan, vel_adj)
        vel_adj_mean = np.nanmean(vel_adj, axis=1)
        
        speed_bins = np.linspace(-50, 50, 1001)
#         print(speed_bins[215*2:286*2])
        speed_histo_left = np.array([np.histogram(row, speed_bins)[0] for row in edge.speeds_tot[0][0]])
        speed_histo_right = np.array([np.histogram(row, speed_bins)[0] for row in edge.speeds_tot[0][1]])
        speed_histo = (speed_histo_left + speed_histo_right)/(2*len(edge.speeds_tot[0][0][0]))
#         print(np.max(speed_histo), np.min(speed_histo), np.sum(speed_histo))
#         print(speed_histo_left)
        
        fig, ax = plt.subplot_mosaic([['kymo', 'speed_hist_zoom', 'speed_hist'],
                                      ['speed_plot', 'flux_plot', 'speed_hist']], figsize=(12,8), layout='constrained')

        fig.suptitle(f"Edge {edge.edge_name} Summary")
        imshow_extent = [0, space_res * edge.kymos[0].shape[1],
                         time_res * edge.kymos[0].shape[0], 0]
        ax['kymo'].imshow(edge.kymos[0], extent=imshow_extent, aspect='auto')
    #     ax[0].set_title(f"Full kymo (length = {space_res * len(edge.kymos[0]):.5} $ \mu m$)")
        ax['kymo'].set_ylabel("time (s)")
        ax['kymo'].set_xlabel("space ($\mu m$)")
        ax['kymo'].set_title("Kymograph")
        ax['speed_plot'].plot(edge.times[0],np.nanmean(edge.speeds_tot[0][0], axis=1), c='tab:blue', label='To root')
        ax['speed_plot'].fill_between(edge.times[0], 
                              np.nanmean(edge.speeds_tot[0][0], axis=1) + np.nanstd(edge.speeds_tot[0][0], axis=1), 
                              np.nanmean(edge.speeds_tot[0][0], axis=1) - np.nanstd(edge.speeds_tot[0][0], axis=1), 
                              alpha=0.5, facecolor='tab:blue')
        ax['speed_plot'].plot(edge.times[0],np.nanmean(edge.speeds_tot[0][1], axis=1),  c='tab:orange', label='To tip')
        ax['speed_plot'].fill_between(edge.times[0], 
                              np.nanmean(edge.speeds_tot[0][1], axis=1) + np.nanstd(edge.speeds_tot[0][1], axis=1), 
                              np.nanmean(edge.speeds_tot[0][1], axis=1) - np.nanstd(edge.speeds_tot[0][1], axis=1), 
                              alpha=0.5, facecolor='tab:orange')
        ax['speed_plot'].plot(edge.times[0], vel_adj_mean, c='black', alpha=0.5, label='effMean')
        ax['speed_plot'].set_title("Speed plots")
        ax['speed_plot'].set_xlabel("time (s)")
        ax['speed_plot'].set_ylabel("speed ($\mu m/s$)")
        ax['speed_plot'].grid(True)
        ax['speed_plot'].set_ylim([-speedmax, speedmax])
#         ax[1][0].set_xlim(ax[1][0].get_ylim()[::-1])
        ax['speed_plot'].legend()
        
        ax['speed_hist'].imshow(speed_histo.T, extent=[ 0, len(speed_histo)*time_res, -50, 50], origin='lower', aspect='auto')
        ax['speed_hist'].axhline(c='w', linestyle='--')
        ax['speed_hist'].set_title(f"Velocity histogram")
        ax['speed_hist'].set_xlabel("time (s)")
        ax['speed_hist'].set_ylabel("speed ($\mu m/s$)")
        
        ax['speed_hist_zoom'].imshow(speed_histo.T[215*2:286*2 - 1], extent=[ 0, len(speed_histo)*time_res, -7, 7], origin='lower', aspect='auto')
        ax['speed_hist_zoom'].axhline(c='w', linestyle='--')
        ax['speed_hist_zoom'].set_title(f"Velocity histogram")
        ax['speed_hist_zoom'].set_xlabel("time (s)")
        ax['speed_hist_zoom'].set_ylabel("speed ($\mu m/s$)")
        
        ax['flux_plot'].plot(edge.times[0],np.nanmean(edge.flux_tot, axis=1),  c='black', label='Average flux')
        ax['flux_plot'].fill_between(edge.times[0], 
                              np.nanmean(edge.flux_tot, axis=1) + np.nanstd(edge.flux_tot, axis=1), 
                              np.nanmean(edge.flux_tot, axis=1) - np.nanstd(edge.flux_tot, axis=1), 
                              alpha=0.5, facecolor='black')
        ax['flux_plot'].set_title("Flux plot")
        ax['flux_plot'].set_xlabel("time (s)")
        ax['flux_plot'].set_ylabel("flux ($q\mu m/s$)")
        ax['flux_plot'].grid(True)
        

        fig.savefig(f"{edge.edge_path}/{edge.edge_name}_summary.png")
    
        fig, ax = plt.subplot_mosaic([['kymo', 'kymo_left', 'kymo_right'],
                                      ['kymo_stat', 'spd_left', 'spd_right']], figsize=(12,9), layout='constrained')
        ax['kymo'].imshow(kymo_tiff[0], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax['kymo_stat'].imshow(kymo_tiff[1], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax['kymo_right'].imshow(kymo_tiff[2], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax['kymo_left'].imshow(kymo_tiff[3], cmap='gray', vmin=0, aspect='auto', extent=imshow_extent)
        ax['spd_left'].imshow(spd_tiff[0], cmap='coolwarm', vmin=-speedmax, vmax=speedmax, aspect='auto', extent=imshow_extent)
        spd_colors = ax['spd_right'].imshow(spd_tiff[1], cmap='coolwarm', vmin=-speedmax, vmax=speedmax, aspect='auto', extent=imshow_extent)

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
        fig.savefig(f"{edge.edge_path}/{edge.edge_name}_kymos.png")
        
        
def month_to_num(x):
    months = {
         'jan': '01',
         'feb': '02',
         'mar': '03',
         'apr': '04',
         'may':'05',
         'jun':'06',
         'jul':'07',
         'aug':'08',
         'sep':'09',
         'oct':'10',
         'nov':'11',
         'dec':'12'
        }
    a = x.strip()[:3].lower()
    try:
        ez = months[a]
        return ez
    except:
        raise ValueError('Not a month')
        
def read_video_data(address_array, folders_frame):
    folders_frame['plate_id_csv'] = [f"{row['Date Imaged']}_Plate{row['Plate number']}" for index, row in folders_frame.iterrows()]
    folders_frame['unique_id_csv'] = [f"{row['plate_id_csv']}_{row['video'].split(os.sep)[0]}" for index, row in folders_frame.iterrows()]
    folders_frame['plate_id_xl'] = [f"{row['Date Imaged']}_Plate{row['Plate number']}" for index, row in folders_frame.iterrows()]
    folders_frame['unique_id_xl'] = [f"{row['plate_id_xl']}_{row['tot_path_drop'].split(os.sep)[-1].split('_')[-1]}" for index, row in folders_frame.iterrows()]
#     print(folders_frame['plate_id_csv'][0],
#           folders_frame['unique_id_csv'][0],
#           folders_frame['plate_id_xl'][0],
#           folders_frame['unique_id_xl'][0])
    excel_frame = pd.DataFrame()
    csv_frame = pd.DataFrame()
    txt_frame = pd.DataFrame()
    for address in tqdm(address_array):
#         print(address)
        suffix = address.split('.')[-1]
        if suffix == 'xlsx':
            raw_data = pd.read_excel(address)
            raw_data = raw_data[raw_data['Treatment'] == raw_data['Treatment']].reset_index()
            raw_data['plate_id_xl'] = [f"{entry.split('_')[-3]}_Plate{entry.split('_')[-2][5:]}" for entry in raw_data['Unnamed: 0']]
            folders_plate_frame = folders_frame[folders_frame['plate_id_xl'].str.lower().isin(raw_data['plate_id_xl'].str.lower())]
#             print(folders_frame['plate_id_xl'])
#             print(raw_data['plate_id_xl'])
            raw_data = raw_data.set_index('Unnamed: 0').join(folders_plate_frame.set_index('unique_id_xl'), lsuffix='', rsuffix='_folder')
            raw_data = raw_data.reset_index()
#             print(raw_data.iloc[0])
            excel_frame = pd.concat([excel_frame, raw_data])

        elif suffix == 'csv':
            df_comma = pd.read_csv(address, nrows=1,sep=",")
            df_semi = pd.read_csv(address, nrows=1, sep=";")
            if df_comma.shape[1]>df_semi.shape[1]:
                raw_data = pd.read_csv(address, sep=",")
            else:
                raw_data = pd.read_csv(address, sep=";")
            raw_data['file_name'] = [address.split(os.sep)[-1].split('.')[-2]] * len(raw_data)
#             print(address.split(os.sep)[-1].split('.')[-2])
            
            folders_plate_frame = folders_frame[folders_frame['plate_id_csv'].str.lower().isin(raw_data['file_name'].str.lower())].reset_index()
#             print(folders_plate_frame)
            raw_data['unique_id'] = folders_plate_frame['unique_id_csv']
#             print(raw_data['unique_id'])
            raw_data = raw_data.set_index('unique_id').join(folders_plate_frame.set_index('unique_id_csv'), lsuffix='', rsuffix='_folder')
            raw_data = raw_data[raw_data['tot_path_drop'] == raw_data['tot_path_drop']]
            raw_data['tot_path'] = [entry[5:] + os.sep for entry in raw_data['tot_path_drop']]
            raw_data = raw_data.reset_index()
            csv_frame = pd.concat([csv_frame, raw_data], axis=0, ignore_index=True)
            
        elif suffix == 'txt':
            if not os.path.exists(address):
                print(f"Could not find {address}, skipping for now")
                continue
            raw_data = pd.read_csv(address, sep=": ", engine='python').T
            raw_data = raw_data.dropna(axis=1, how='all')

#             raw_data = raw_data.reset_index(drop=True)
            raw_data['unique_id'] = [f"{address.split(os.sep)[-3]}_{address.split(os.sep)[-2]}"]
            raw_data['tot_path'] = [address[34:-13] + 'Img/']
            raw_data['tot_path_drop'] = ['DATA/' + raw_data['tot_path'][0]]
#             print(raw_data)
            try:
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)
            except:
                print(f"Weird concatenation with {address}, trying to reset index")
                print(raw_data.columns)
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)

    
    if len(excel_frame) >0:
        excel_frame['Binned (Y/N)'] = [np.where(entry == 'Y', 2, 1) for entry in excel_frame['Binned (Y/N)']]
        excel_frame['Time after crossing'] = [int(entry.split(' ')[-2]) for entry in excel_frame['Time after crossing']]
        excel_frame = excel_frame.rename(columns={
                'Unnamed: 0' : 'unique_id',
                'Treatment' : 'treatment',
                'Strain' : 'strain',
                'Time after crossing': 'days_after_crossing',
                'Growing temperature' : 'grow_temp',
                'Position mm' : 'xpos',
                'Unnamed: 6' : 'ypos',
                'dcenter mm' : 'dcenter',
                'droot mm' : 'droot',
                'Bright-field (BF)\nor\nFluorescence (F)' : 'mode',
                'Binned (Y/N)' : 'binning',
                'Magnification' : 'magnification',
                'FPS' : 'fps',
                'Video Length (s)' : 'time_(s)',
                'Comments' : 'comments',
        })
    if len(txt_frame) > 0:
#         print(txt_frame)
        txt_frame = txt_frame.dropna(axis=1, how='all')
        txt_frame = txt_frame.drop(['Computer', 'User', 'DataRate', 'DataSize', 'Frames Recorded', 'Fluorescence', 'Four Led Bar', 'Model', 'FrameSize'], axis=1)
        txt_frame['record_time'] = [entry.split(',')[-1] for entry in txt_frame['DateTime']]
        txt_frame['DateTime'] = [f"{entry.split(', ')[1].split(' ')[-1]}{month_to_num(entry.split(', ')[-2].split(' ')[-2])}{entry.split(', ')[1].split(' ')[-3]}" for entry in txt_frame['DateTime']]
        txt_frame['CrossDate'] = [str(int(entry)) for entry in txt_frame['CrossDate']]
        txt_frame['days_after_crossing'] = [(datetime.date(int(row['DateTime'][:4]), int(row['DateTime'][4:6]), int(row['DateTime'][6:])) - datetime.date(int(row['CrossDate'][:4]), int(row['CrossDate'][4:6]), int(row['CrossDate'][6:]))).days for index, row in txt_frame.iterrows()]
        
        txt_frame['X'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['X']]
        txt_frame['Y'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['Y']]
        txt_frame['Z'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['Z']]

        txt_frame['Binning'] = [int(entry[-1]) for entry in txt_frame['Binning']]
        txt_frame['FrameRate'] = [float(entry.split(' ')[-3]) for entry in txt_frame['FrameRate']]
        txt_frame['magnification'] = [float(entry.split(' ')[-2][:-1]) for entry in txt_frame['Operation']]
        txt_frame['Operation'] = [str(np.where(entry.split(' ')[-1] == 'Brightfield', 'BF', 'F')) for entry in txt_frame['Operation']]
        txt_frame['Time'] = [float(entry.split(' ')[-2]) for entry in txt_frame['Time']]
        
        txt_frame['ExposureTime'] = [entry.split('  ')[-1].split(' ')[1] for entry in txt_frame['ExposureTime']]
        txt_frame['ExposureTime'] = pd.to_numeric(txt_frame['ExposureTime'], errors='coerce')
        txt_frame['Run'] = [int(entry) for entry in txt_frame['Run']]
        txt_frame['Gain'] = [float(entry) for entry in txt_frame['Gain']]
        txt_frame['Gamma'] = [float(entry) for entry in txt_frame['Gamma']]
        txt_frame['Root'] = [entry.split(' ')[-1] for entry in txt_frame['Root']]
        txt_frame['Strain'] = [entry.split(' ')[-1] for entry in txt_frame['Strain']]
        txt_frame['StoragePath'] = [entry.split(' ')[-1] for entry in txt_frame['StoragePath']]
        txt_frame['Treatment'] = [entry.split(' ')[-1] for entry in txt_frame['Treatment']]
        txt_frame = txt_frame.rename(columns={
            'DateTime': 'imaging_day',
            'StoragePath' : 'storage_path',
            'Plate' : 'plate_id',
            'Root' : 'root',
            'Strain' : 'strain',
            'Treatment' : 'treatment',
            'CrossDate' : 'crossing_day',
            'Run' : 'video_int',
            'Time' : 'time_(s)',
            'Operation' : 'mode',
            'ExposureTime' : 'exposure_time_(us)',
            'FrameRate' : 'fps',
            'Binning' : 'binning',
            'Gain' : 'gain',
            'Gamma' : 'gamma',
            'X' : 'xpos',
            'Y' : 'ypos',
            'Z' : 'zpos',
        })

    if len(csv_frame) > 0:
#         print(csv_frame['unique_id'])
        csv_frame['video_id'] = [entry.split('_')[-1] for entry in csv_frame['unique_id']]
        csv_frame['plate_nr'] = [int(entry.split('_')[-2][5:]) for entry in csv_frame['unique_id']]
        csv_frame['Lens'] = csv_frame["Lens"].astype(float)
        csv_frame['fps'] = csv_frame["fps"].astype(float)
        csv_frame['time'] = csv_frame["time"].astype(float)
        csv_frame = csv_frame.rename(columns={
            'video' : 'video_int',
            'Treatment' : 'treatment',
            'Strain' : 'strain',
            'tGermination' : 'days_after_crossing',
            'Illumination' : 'mode',
            'Binned' : 'binning',
            'Lens' : 'magnification',
            'plate_id_xl' : 'plate_id',
            'time' : 'time_(s)',
        })
        csv_frame = csv_frame.drop(columns=['index', 'Plate number', 'video_folder', 'file_name'], axis=1)
    if len(csv_frame)>0 and len(txt_frame)>0:
#         print(txt_frame['unique_id'])
        merge_frame = pd.merge(txt_frame, csv_frame, how='outer', on='unique_id', suffixes=("", "_csv"))
        
        merge_frame = merge_frame.drop(columns=['unique_id_xl', 'plate_id', 'video_folder', 'Plate number', 'folder', 'file_name'],axis=1)
        merge_frame = merge_frame.rename(columns={'plate_id_xl' : 'plate_id'})
        merge_frame['imaging_day'] = merge_frame['imaging_day'].fillna(merge_frame['Date Imaged'])
        merge_frame['strain'] = merge_frame['strain'].fillna(merge_frame['strain_csv'])
        merge_frame['treatment'] = merge_frame['treatment'].fillna(merge_frame['treatment_csv'])
        merge_frame['video_int'] = merge_frame['video_int'].fillna(merge_frame['video_int_csv'])
        merge_frame['time_(s)'] = merge_frame['time_(s)'].fillna(merge_frame['time'])
        merge_frame['mode'] = merge_frame['mode'].fillna(merge_frame['mode_csv'])
        merge_frame['fps'] = merge_frame['fps'].fillna(merge_frame['fps_csv'])
        merge_frame['binning'] = merge_frame['binning'].fillna(merge_frame['binning_csv'])
        merge_frame['xpos'] = merge_frame['xpos'].fillna(merge_frame['xpos_csv'])
        merge_frame['ypos'] = merge_frame['ypos'].fillna(merge_frame['ypos_csv'])
        merge_frame['magnification'] = merge_frame['magnification'].fillna(merge_frame['magnification_csv'])
        merge_frame['tot_path'] = merge_frame['tot_path'].fillna(merge_frame['tot_path_csv'])
        merge_frame['days_after_crossing'] = merge_frame['days_after_crossing'].fillna(merge_frame['days_after_crossing_csv'])
        merge_frame = merge_frame.drop(columns=['root', 'video_int_csv', 'treatment_csv', 'strain_csv', 'days_after_crossing_csv', 'xpos_csv', 'ypos_csv', 'mode_csv', 'binning_csv', 'magnification_csv', 'fps_csv', 'plate_id_csv', 'Date Imaged', 'tot_path_csv', 'index'],axis=1)

    elif len(excel_frame) > 0 and len(txt_frame) > 0:
        merge_frame = pd.merge(excel_frame, csv_frame, how='left', on='unique_id', suffixes=("", "_csv"))
    elif len(txt_frame) > 0:
        merge_frame = txt_frame
        merge_frame['plate_id'] = [f"{row['imaging_day']}_Plate{int(row['plate_id'])}" for index, row in merge_frame.iterrows()]
    elif len(excel_frame) > 0:
        merge_frame = excel_frame
        merge_frame = merge_frame[merge_frame['tot_path_drop'] == merge_frame['tot_path_drop']]
        merge_frame['tot_path'] = [entry[5:] + '/Img/' for entry in merge_frame['tot_path_drop']]
        merge_frame = merge_frame.rename(columns={
            'plate_id_xl' : 'plate_id',
            'Plate number' : 'plate_nr',
            'Date Imaged' : 'imaging_day',
        })
        merge_frame = merge_frame.drop(columns=['plate_id_xl_folder', 'video', 'folder'], axis=1)
    elif len(csv_frame) > 0:
        csv_frame = csv_frame.rename(columns={'Date Imaged' : 'imaging_day', 'place_id_csv' : 'plate_id'})
        csv_frame = csv_frame.drop(columns=['plate_id_csv', 'unique_id_xl', 'folder'],axis=1)
        merge_frame = csv_frame
    else:
        raise("Could not find enough data!")
    return merge_frame
    
    
    
    