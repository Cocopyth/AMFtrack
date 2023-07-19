import copy

import numpy as np
import pandas as pd
from IPython.display import clear_output
import re
from amftrack.pipeline.development.high_mag_videos.kymo_class import *
from amftrack.pipeline.development.high_mag_videos.plot_data import (
    save_raw_data,
    plot_summary,
)
import matplotlib.patches as mpatches
from pathlib import Path, PurePath
import sys
import os
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite
from tqdm import tqdm
import matplotlib as mpl
from amftrack.util.dbx import upload_folders, download, read_saved_dropbox_state, save_dropbox_state, load_dbx, \
    download, get_dropbox_folders, get_dropbox_video_folders
import logging
import datetime

logging.basicConfig(stream=sys.stdout, level=logging.debug)
mpl.rcParams['figure.dpi'] = 300


def month_to_num(x:str):
    """
    Takes a string with the name of a month, and returns that month's corresponding number as a string.
    :param x:   String, preferably of a month
    :return:    Two-digit string number of that month
    """
    months = {
        'jan': '01',
        'feb': '02',
        'mar': '03',
        'apr': '04',
        'may': '05',
        'jun': '06',
        'jul': '07',
        'aug': '08',
        'sep': '09',
        'oct': '10',
        'nov': '11',
        'dec': '12'
    }
    a = x.strip()[:3].lower()
    try:
        ez = months[a]
        return ez
    except:
        raise ValueError('Not a month')


def index_videos_dropbox(analysis_folder, videos_folder, dropbox_folder,
                         REDO_SCROUNGING=False, CREATE_VIDEO_HIERARCHY=True):
    """
    Goes through the specified dropbox folder, and collects and organises all the relevant video data for all videos
    stored in that dropbox. Works recursively. On the local machine will also create a folder structure in the
    analysis and videos folder, populating the analysis folder with the video info. Returns a merged pandas dataframe
    with all video information.
    :param analysis_folder:         Local address where video info, and later analysis will be stored
    :param videos_folder:           Local address where raw video data will be stored later
    :param dropbox_folder:          Folder address on the dropbox with all videos you want to process
    :param REDO_SCROUNGING:         Boolean whether to redo the dropbox searching. Dropbox will be searched anyway if cache .json cannot be found
    :param CREATE_VIDEO_HIERARCHY:  Boolean whether to create the video hierarchy. Only set to False if you don't plan on downloading the raw data
    :return:                        Pandas dataframe with all video information
    """

    # Using PathLib allows for more OS-ambiguous functionality

    analysis_folder = Path(analysis_folder)
    dropbox_folder = Path(dropbox_folder)
    videos_folder = Path(videos_folder)

    analysis_json = analysis_folder.joinpath(dropbox_folder.relative_to('/DATA/')).joinpath("all_folders_drop.json")
    excel_json = analysis_folder.joinpath(dropbox_folder.relative_to('/DATA/')).joinpath("excel_drop.json")

    # First analysis_json and excel_json will be sought for, these serve as caches for the dropbox data.
    # If they can't be found, the information will be assembled from the indexing files on the Dropbox.

    if os.path.exists(analysis_json):
        all_folders_drop = pd.read_json(analysis_json)
    if os.path.exists(excel_json):
        excel_drop = pd.read_json(excel_json, typ='series')
    if not os.path.exists(analysis_json) or REDO_SCROUNGING:
        print("Redoing the dropbox scrounging, hold on tight.")
        all_folders_drop, excel_drop, txt_drop = get_dropbox_video_folders(dropbox_folder)

        clear_output(wait=False)
        print("Scrounging complete, downloading files...")

        excel_addresses = np.array([re.search("^.*Plate.*\/.*Plate.*$", entry, re.IGNORECASE) for entry in excel_drop])
        print(excel_addresses)
        if len(excel_addresses) > 0:
            excel_addresses = [i for i in excel_addresses if i is not None]
            print(excel_addresses)
            excel_addresses = [address.group(0) for address in excel_addresses]
        excel_drop = np.concatenate([excel_addresses, txt_drop])
        if not analysis_folder.joinpath(dropbox_folder.relative_to('/DATA/')).exists():
            analysis_folder.joinpath(dropbox_folder.relative_to('/DATA/')).mkdir()
        all_folders_drop.to_json(analysis_json)
        pd.Series(excel_drop).to_json(excel_json)
    info_addresses = []

    # Once addresses have been found, all relevant index files will be downloaded to their respective position

    for address in excel_drop:
        address_local = Path(address).relative_to('/DATA/')
        if not (analysis_folder / address_local.parent).exists():
            (analysis_folder / address_local.parent).mkdir(parents=True)
        if not (analysis_folder / address_local).exists() or REDO_SCROUNGING:
            download(address, (analysis_folder / address_local))
        info_addresses.append(analysis_folder / address_local)
    clear_output(wait=False)
    print("All files downloaded! Merging files...")

    # The downloaded information is then read and merged into one dataframe containing all relevant information.

    merge_frame = read_video_data(info_addresses, all_folders_drop, analysis_folder)
    merge_frame = merge_frame.rename(columns={'tot_path': 'folder'})
    merge_frame = merge_frame.sort_values('unique_id')
    merge_frame = merge_frame.reset_index(drop=True)
    merge_frame = merge_frame.loc[:, ~merge_frame.columns.duplicated()].copy()
    merge_frame['analysis_folder'] = [np.nan for i in range(len(merge_frame))]
    merge_frame['videos_folder'] = [np.nan for i in range(len(merge_frame))]

    for index, row in merge_frame.iterrows():
        target_anals_file = analysis_folder / row['folder'][:-4]
        target_video_file = videos_folder / row['folder']

        row.loc['analysis_folder'] = target_anals_file.as_posix()
        row.loc['videos_folder'] = target_video_file.as_posix()

        if not target_video_file.exists() and CREATE_VIDEO_HIERARCHY:
            target_video_file.mkdir(parents=True)
        row.to_json(target_anals_file / "video_data.json", orient="index")

    return merge_frame


def read_video_data(address_array, folders_frame, analysis_folder):
    """
    Takes a list of index files, a frame with basic video information and the analysis folder to merge the
    information altogether into a readable pandas dataframe with consistent columns and values.
    :param address_array:   List of addresses pointing to downloaded .xslx, .csv and .txt files.
    :param folders_frame:   Frame with basic video parameters which will be expanded, created with the dropbox download
                            functions
    :param analysis_folder: Folder where all of this will be found. Highest folder in the hierarchy.
    :return:                Merged frame with all video information that could be found in the index files.
    """
    folders_frame['plate_id_csv'] = [f"{row['Date Imaged']}_Plate{row['Plate number']}" for index, row in
                                     folders_frame.iterrows()]
    folders_frame['unique_id_csv'] = [f"{row['plate_id_csv']}_{str(row['video']).split(os.sep)[0]}" for index, row in
                                      folders_frame.iterrows()]
    folders_frame['plate_id_xl'] = [f"{row['Date Imaged']}_Plate{row['Plate number']}" for index, row in
                                    folders_frame.iterrows()]
    folders_frame['unique_id_xl'] = [f"{row['plate_id_xl']}_{row['tot_path_drop'].split(os.sep)[-1].split('_')[-1]}" for
                                     index, row in folders_frame.iterrows()]
    folders_frame['unique_id_xl'] = [entry.lower() for entry in folders_frame['unique_id_xl']]

    excel_frame = pd.DataFrame()
    csv_frame = pd.DataFrame()
    txt_frame = pd.DataFrame()
    for address in tqdm(address_array):
        suffix = address.suffix
        if suffix == '.xlsx':
            raw_data = pd.read_excel(address)
            if 'Binned (Y/N)' not in raw_data:
                raw_data['Binned (Y/N)'] = ['N' for entry in raw_data['Unnamed: 0']]
            raw_data["Binned (Y/N)"] = raw_data["Binned (Y/N)"].astype(str)
            raw_data = raw_data[raw_data['Treatment'] == raw_data['Treatment']].reset_index(drop=True)
            raw_data['Unnamed: 0'] = [entry.lower() for entry in raw_data['Unnamed: 0']]
            raw_data['plate_id_xl'] = [f"{entry.split('_')[-3]}_Plate{entry.split('_')[-2][5:]}" for entry in
                                       raw_data['Unnamed: 0']]
            folders_plate_frame = folders_frame[
                folders_frame['plate_id_xl'].str.lower().isin(raw_data['plate_id_xl'].str.lower())]
            raw_data = raw_data.join(folders_plate_frame.set_index('unique_id_xl'), lsuffix='', rsuffix='_folder',
                                     on='Unnamed: 0')
            raw_data = raw_data.reset_index()
            excel_frame = pd.concat([excel_frame, raw_data])

        elif suffix == '.csv':
            df_comma = pd.read_csv(address, nrows=1, sep=",")
            df_semi = pd.read_csv(address, nrows=1, sep=";")
            if df_comma.shape[1] > df_semi.shape[1]:
                raw_data = pd.read_csv(address, sep=",")
            else:
                raw_data = pd.read_csv(address, sep=";")
            raw_data['file_name'] = [address.stem] * len(raw_data)
            folders_plate_frame = folders_frame[
                folders_frame['plate_id_csv'].str.lower().isin(raw_data['file_name'].str.lower())].reset_index()
            raw_data['unique_id'] = folders_plate_frame['unique_id_csv']
            raw_data = raw_data.set_index('unique_id').join(folders_plate_frame.set_index('unique_id_csv'), lsuffix='',
                                                            rsuffix='_folder')
            raw_data = raw_data[raw_data['tot_path_drop'] == raw_data['tot_path_drop']]
            raw_data['tot_path'] = [entry[5:] + os.sep for entry in raw_data['tot_path_drop']]
            raw_data = raw_data.reset_index()
            csv_frame = pd.concat([csv_frame, raw_data], axis=0, ignore_index=True)

        elif suffix == '.txt':
            if not address.exists():
                print(f"Could not find {address}, skipping for now")
                continue
            raw_data = pd.read_csv(address, sep=": ", engine='python').T
            raw_data = raw_data.dropna(axis=1, how='all')
            raw_data['unique_id'] = [f"{address.parts[-3]}_{address.parts[-2]}"]
            raw_data["tot_path"] = (address.relative_to(analysis_folder).parent / "Img").as_posix()
            raw_data['tot_path_drop'] = ['DATA/' + raw_data['tot_path'][0]]
            try:
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)
            except:
                print(f"Weird concatenation with {address}, trying to reset index")
                print(raw_data.columns)
                txt_frame = pd.concat([txt_frame, raw_data], axis=0, ignore_index=True)

    if len(excel_frame) > 0:
        excel_frame['Binned (Y/N)'] = [np.where(entry == 'Y', 2, 1) for entry in excel_frame['Binned (Y/N)']]
        excel_frame["Binned (Y/N)"] = excel_frame["Binned (Y/N)"].astype(int)
        excel_frame['Time after crossing'] = [int(entry.split(' ')[-2]) for entry in excel_frame['Time after crossing']]
        excel_frame = excel_frame.rename(columns={
            'Unnamed: 0': 'unique_id',
            'Treatment': 'treatment',
            'Strain': 'strain',
            'Time after crossing': 'days_after_crossing',
            'Growing temperature': 'grow_temp',
            'Position mm': 'xpos',
            'Unnamed: 6': 'ypos',
            'dcenter mm': 'dcenter',
            'droot mm': 'droot',
            'Bright-field (BF)\nor\nFluorescence (F)': 'mode',
            'Binned (Y/N)': 'binning',
            'Magnification': 'magnification',
            'FPS': 'fps',
            'Video Length (s)': 'time_(s)',
            'Comments': 'comments',
        })
    if len(txt_frame) > 0:
        txt_frame = txt_frame.dropna(axis=1, how='all')
        txt_frame = txt_frame.drop(
            ['Computer', 'User', 'DataRate', 'DataSize', 'Frames Recorded', 'Fluorescence', 'Four Led Bar', 'Model',
             'FrameSize'], axis=1)
        txt_frame['record_time'] = [entry.split(',')[-1] for entry in txt_frame['DateTime']]
        txt_frame['DateTime'] = [
            f"{entry.split(', ')[1].split(' ')[-1]}{month_to_num(entry.split(', ')[-2].split(' ')[-2])}{entry.split(', ')[1].split(' ')[-3]}"
            for entry in txt_frame['DateTime']]
        txt_frame['CrossDate'] = [str(int(entry)) for entry in txt_frame['CrossDate']]
        txt_frame['days_after_crossing'] = [(datetime.date(int(row['DateTime'][:4]), int(row['DateTime'][4:6]),
                                                           int(row['DateTime'][6:])) - datetime.date(
            int(row['CrossDate'][:4]), int(row['CrossDate'][4:6]), int(row['CrossDate'][6:]))).days for index, row in
                                            txt_frame.iterrows()]

        txt_frame['X'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['X']]
        txt_frame['Y'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['Y']]
        txt_frame['Z'] = [float(entry.split('  ')[-1].split(' ')[0]) for entry in txt_frame['Z']]

        txt_frame['Binning'] = [int(entry[-1]) for entry in txt_frame['Binning']]
        txt_frame['FrameRate'] = [float(entry.split(' ')[-3]) for entry in txt_frame['FrameRate']]
        txt_frame['magnification'] = [float(entry.split(' ')[-2][:-1]) for entry in txt_frame['Operation']]
        txt_frame['Operation'] = [str(np.where(entry.split(' ')[-1] == 'Brightfield', 'BF', 'F')) for entry in
                                  txt_frame['Operation']]
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
            'StoragePath': 'storage_path',
            'Plate': 'plate_id',
            'Root': 'root',
            'Strain': 'strain',
            'Treatment': 'treatment',
            'CrossDate': 'crossing_day',
            'Run': 'video_int',
            'Time': 'time_(s)',
            'Operation': 'mode',
            'ExposureTime': 'exposure_time_(us)',
            'FrameRate': 'fps',
            'Binning': 'binning',
            'Gain': 'gain',
            'Gamma': 'gamma',
            'X': 'xpos',
            'Y': 'ypos',
            'Z': 'zpos',
        })

    if len(csv_frame) > 0:
        csv_frame['video_id'] = [entry.split('_')[-1] for entry in csv_frame['unique_id']]
        csv_frame['plate_nr'] = [int(entry.split('_')[-2][5:]) for entry in csv_frame['unique_id']]
        csv_frame['Lens'] = csv_frame["Lens"].astype(float)
        csv_frame['fps'] = csv_frame["fps"].astype(float)
        csv_frame['time'] = csv_frame["time"].astype(float)
        csv_frame = csv_frame.rename(columns={
            'video': 'video_int',
            'Treatment': 'treatment',
            'Strain': 'strain',
            'tGermination': 'days_after_crossing',
            'Illumination': 'mode',
            'Binned': 'binning',
            'Lens': 'magnification',
            'plate_id_xl': 'plate_id',
            'time': 'time_(s)',
        })
        csv_frame = csv_frame.drop(columns=['index', 'Plate number', 'video_folder', 'file_name'], axis=1)
    if len(csv_frame) > 0 and len(txt_frame) > 0:
        merge_frame = pd.merge(txt_frame, csv_frame, how='outer', on='unique_id', suffixes=("", "_csv"))
        merge_frame = merge_frame.drop(
            columns=['unique_id_xl', 'plate_id', 'video_folder', 'Plate number', 'folder', 'file_name'], axis=1)
        merge_frame = merge_frame.rename(columns={'plate_id_xl': 'plate_id'})
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
        merge_frame['days_after_crossing'] = merge_frame['days_after_crossing'].fillna(
            merge_frame['days_after_crossing_csv'])
        merge_frame = merge_frame.drop(
            columns=['video_int_csv', 'treatment_csv', 'strain_csv', 'days_after_crossing_csv', 'xpos_csv', 'ypos_csv',
                     'mode_csv', 'binning_csv', 'magnification_csv', 'fps_csv', 'plate_id_csv', 'Date Imaged',
                     'tot_path_csv', 'index'], axis=1)

    elif len(excel_frame) > 0 and len(txt_frame) > 0:
        merge_frame = pd.merge(excel_frame, csv_frame, how='left', on='unique_id', suffixes=("", "_csv"))
    elif len(txt_frame) > 0:
        merge_frame = txt_frame
        merge_frame['plate_id'] = [f"{row['imaging_day']}_Plate{int(row['plate_id'])}" for index, row in
                                   merge_frame.iterrows()]
    elif len(excel_frame) > 0:
        merge_frame = excel_frame.reset_index(drop=True)
        merge_frame = merge_frame[merge_frame['tot_path_drop'] == merge_frame['tot_path_drop']]
        merge_frame['tot_path'] = [entry[5:] + os.sep + 'Img' + os.sep for entry in merge_frame['tot_path_drop']]
        merge_frame = merge_frame.rename(columns={
            'plate_id_xl': 'plate_id',
            'Plate number': 'plate_nr',
            'Date Imaged': 'imaging_day',
        })
        merge_frame = merge_frame.drop(columns=['plate_id_xl_folder', 'video', 'folder'], axis=1)
    elif len(csv_frame) > 0:
        csv_frame = csv_frame.rename(columns={'Date Imaged': 'imaging_day', 'place_id_csv': 'plate_id'})
        csv_frame = csv_frame.drop(columns=['plate_id_csv', 'unique_id_xl', 'folder'], axis=1)
        merge_frame = csv_frame
    else:
        raise "Could not find enough data!"
    return merge_frame


class HighmagDataset(object):
    def __init__(self,
                 dataframe: pd.DataFrame,
                 analysis_folder: str,
                 videos_folder: str):
        self.analysis_folder = analysis_folder
        self.videos_folder = videos_folder
        self.video_frame = dataframe
        self.video_objs = np.array(
            [VideoDataset(row, analysis_folder, videos_folder) for index, row in self.video_frame.iterrows()])
        self.edge_objs = np.concatenate([video.edge_objs for video in self.video_objs])
        self.edges_frame = pd.concat([edg_obj.mean_data for edg_obj in self.edge_objs], axis=1).T.reset_index(drop=True)

    def filter_edges(self, column, compare, constant):
        filter_self = copy.deepcopy(self)
        if compare == '>=':
            filter_self.edges_frame = filter_self.edges_frame[filter_self.edges_frame[column].ge(constant)]
        elif compare == '==':
            filter_self.edges_frame = filter_self.edges_frame[filter_self.edges_frame[column] == constant]
        elif compare == '<=':
            filter_self.edges_frame = filter_self.edges_frame[filter_self.edges_frame[column].le(constant)]
        else:
            print("Comparison symbol not recognised. Please use >=, ==, or <=.")
            raise "Comparison symbol not recognised. Please use >, ==, or <."

        filter_self.edge_objs = filter_self.edge_objs[filter_self.edges_frame.index.to_numpy()]
        is_video = filter_self.video_frame['unique_id'].isin(filter_self.edges_frame['unique_id'])
        filter_self.video_objs = filter_self.video_objs[is_video[is_video].index.values]
        filter_self.video_frame = filter_self.video_frame[is_video].reset_index(drop=True)
        filter_self.edges_frame = filter_self.edges_frame.reset_index(drop=True)
        return filter_self

    def context_4x(self, pd_row, FLUO=True):
        space_res = 2 * 1.725 / pd_row['magnification'] * pd_row['binning']
        pic_4x_res = np.array([[4088, 3000],[2044, 1500]][pd_row['binning'] == 2]) * space_res / 2
        frame_4x_filt = self.filter_edges('plate_id', '==', pd_row['plate_id'])
        frame_4x_filt = frame_4x_filt.filter_edges('magnification', '==', 50.0)
        frame_4x_filt = frame_4x_filt.filter_edges('mode', '==', ['BF', 'F'][FLUO])
        frame_4x_filt = frame_4x_filt.filter_edges('xpos', '>=', pd_row['xpos'] - pic_4x_res[0])
        frame_4x_filt = frame_4x_filt.filter_edges('xpos', '<=', pd_row['xpos'] + pic_4x_res[0])
        frame_4x_filt = frame_4x_filt.filter_edges('ypos', '>=', pd_row['ypos'] - pic_4x_res[1])
        frame_4x_filt = frame_4x_filt.filter_edges('ypos', '<=', pd_row['ypos'] + pic_4x_res[1])
        frame_4x_filt.video_frame = pd.concat([pd_row.to_frame().T, frame_4x_filt.video_frame])

        return frame_4x_filt

    def plot_locs(self, analysis_folder, FLUO=True, x_adj = -500, y_adj = 400):
        fig, ax = plt.subplots()

        space_res = 2*1.725 / self.video_frame.iloc[0]['magnification'] * self.video_frame.iloc[0]['binning']
        pic_4x_res = np.array([[4088, 3000],[2044, 1500]][self.video_frame.iloc[0]['binning'] == 2]) * space_res /2

        vidcap = cv2.VideoCapture(f"{analysis_folder}{self.video_frame.iloc[0]['folder'][:-4]}{self.video_frame.iloc[0]['unique_id']}_video.mp4")
        success, image = vidcap.read()
        if success:
            ax.imshow(image, extent = [(self.video_frame.iloc[0]['xpos'] - pic_4x_res[0]) + x_adj,
                                       (self.video_frame.iloc[0]['xpos'] + pic_4x_res[0]) + x_adj,
                                       (-self.video_frame.iloc[0]['ypos'] - pic_4x_res[1]) + y_adj,
                                       (-self.video_frame.iloc[0]['ypos'] + pic_4x_res[1]) + y_adj
                                       ])

        for index, row in self.edges_frame.iterrows():
            space_res = 2 * 1.725 / row['magnification'] * row['binning']
            ax.annotate(row['unique_id'].split('_')[-1], (row['xpos'], -row['ypos']), c='white')

            arrow_start = np.array([(row['xpos'] + space_res * row['edge_ypos_2'],
                                     (row['ypos'] + space_res * row['edge_xpos_2']) * -1)])[0]
            arrow_end = np.array([(row['xpos'] + space_res * row['edge_ypos_1'],
                                   (row['ypos'] + space_res * row['edge_xpos_1']) * -1)])[0] - arrow_start
            ax.quiver(arrow_start[0], arrow_start[1], arrow_end[0], arrow_end[1], angles='xy',
                      color=['tab:green', 'gray'][row['magnification'] == 50], scale_units='xy', scale=1)
        #     print(arrow_start, arrow_end)

        ax.scatter(self.video_frame.iloc[1:]['xpos'], -self.video_frame.iloc[1:]['ypos'])
        # ax.annotate(self.video_frame['unique_id'].astype(str), (self.video_frame['xpos'], self.video_frame['ypos']))
        ax.axis('equal')
        ax.set_title(f"4x overview of {self.video_frame.iloc[0]['unique_id']}")
        fig.tight_layout()
        return None

    def filter_videos(self, column, compare, constant):
        return None

    def bin_values(self, column, bins, new_column_name='edge_bin_values'):
        bin_series = pd.cut(self.edges_frame[column], bins, labels=False)
        self.edges_frame[new_column_name] = bin_series
        return bin_series

    def plot_histo(self, bins, column, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.edges_frame[column], bins=bins, edgecolor='black')
        ax.set_xlabel(column)
        ax.set_ylabel("counts")
        if ax is None:
            fig.tight_layout()
        return ax

    def plot_violins(self, column, bins=None, bin_separator='edge_bin_values', ax=None, c='tab:blue', labels=None):
        if ax is None:
            fig, ax = plt.subplots()
        if bins is not None:
            violin_data = [self.edges_frame[column][self.edges_frame[bin_separator] == i].astype(float) for i in
                       range(len(bins))]
        else:
            violin_data = [self.edges_frame[column][self.edges_frame[bin_separator] == i].astype(float) for i in
                            self.edges_frame[bin_separator].unique()]
            x_labels = self.edges_frame[bin_separator].unique()
            bins = range(self.edges_frame[bin_separator].nunique())
            ax.set_xticks(bins, labels = x_labels)
        violin_data_d = []
        for data in violin_data:
            if data.empty:
                violin_data_d.append([np.nan, np.nan])
            else:
                violin_data_d.append(data)
        violin_parts = ax.violinplot(dataset=violin_data_d, positions=bins, widths = (bins[1] - bins[0])*0.6, showmeans=False, showmedians=False, showextrema=False)
        violin_means = [np.nanmean(data, axis=0) for data in violin_data_d]
        ax.scatter(bins, violin_means, s=4, c='black')
        for vp in violin_parts['bodies']:
            vp.set_edgecolor('black')
            vp.set_alpha(1)
            vp.set_facecolor(c)
        if labels is not None:
            labels.append((mpatches.Patch(color=c), column))

        return ax

    def return_video_frame(self):
        return self.video_frame

    def return_edge_frame(self):
        return self.edges_frame

    def return_edge_objs(self):
        return self.edge_objs

    def return_vid_objs(self):
        return self.video_objs


class VideoDataset(object):
    def __init__(self,
                 series,
                 analysis_folder,
                 videos_folder):
        self.dataset = series
        if os.path.exists(f"{analysis_folder}{self.dataset['folder'][:-4]}edges_data.csv"):
            self.edges_frame = pd.read_csv(f"{analysis_folder}{self.dataset['folder'][:-4]}edges_data.csv")
            self.edge_objs = [EdgeDataset(pd.concat([row, self.dataset]), analysis_folder, videos_folder) for index, row
                              in self.edges_frame.iterrows()]
            self.dataset['nr_of_edges'] = len(self.edges_frame)
        else:
            print(f"Couldn't find the edges data file. Check analysis for {self.dataset['unique_id']}")
            self.dataset['nr_of_edges'] = 0
            self.edge_objs = []

    def show_summary(self):
        if os.path.exists(self.dataset['analysis_folder'] + 'Detected edges.png'):
            print('Index, edge name')
            print(self.edges_frame['edge_name'].to_string())
            extraction_img = imageio.imread(self.dataset['analysis_folder'] + 'Detected edges.png')
            fig, ax = plt.subplots()
            ax.imshow(extraction_img)
            ax.set_axis_off()
            ax.set_title(f"{self.dataset['unique_id']}")
            fig.tight_layout()

    def show_segmentation(self):
        if os.path.exists(self.dataset['analysis_folder'] + 'Video segmentation.png'):
            extraction_img = imageio.imread(self.dataset['analysis_folder'] + 'Video segmentation.png')
            fig, ax = plt.subplots()
            ax.imshow(extraction_img)
            ax.set_axis_off()
            fig.tight_layout()

    def scatter_speeds_video(self):
        fig, ax = plt.subplots()
        # ax.grid(True)
        ax.axhline(c='black', linestyle='--', alpha=0.5)
        ax.scatter(self.edges_frame.index, self.edges_frame['speed_mean'], c='black', label='effMean')
        ax.errorbar(self.edges_frame.index, self.edges_frame['speed_right'],
                    self.edges_frame['speed_right_std'],
                    c='tab:orange', label='to tip', marker='o', linestyle='none', capsize=10)
        ax.errorbar(self.edges_frame.index, self.edges_frame['speed_left'],
                    self.edges_frame['speed_left_std'],
                    c='tab:blue', label='to root', marker='o', linestyle='none', capsize=10)
        ax.legend()
        ax.set_xticks(self.edges_frame.index)
        ax.set_xticklabels(self.edges_frame['edge_name'], rotation=45)
        ax.set_title(f"Edge speed overview for {self.dataset['unique_id']}")
        ax.set_ylabel("Velocity $(\mu m /s)$")
        fig.tight_layout()


class EdgeDataset(object):
    def __init__(self,
                 dataframe,
                 analysis_folder,
                 videos_folder):
        self.mean_data = dataframe
        self.edge_name = self.mean_data['edge_name']
        self.time_data = pd.read_csv(
            f"{analysis_folder}{self.mean_data['folder'][:-4]}edge {self.mean_data['edge_name']}{os.sep}{self.mean_data['edge_name']}_data.csv")

    def show_summary(self):
        summ_img = imageio.imread(
            self.mean_data['analysis_folder'] + 'edge ' + self.mean_data['edge_name'] + os.sep + self.mean_data[
                'edge_name'] + '_summary.png')
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.imshow(summ_img)
        ax.set_axis_off()

    def plot_flux(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_data['times'], self.time_data['flux_mean'])
        fig.tight_layout()
