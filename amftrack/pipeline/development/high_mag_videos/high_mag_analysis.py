import copy

import numpy as np
import pandas as pd
from IPython.display import clear_output
import re
from amftrack.pipeline.development.high_mag_videos.kymo_class import *
from amftrack.pipeline.development.high_mag_videos.plot_data import (
    save_raw_data,
    plot_summary,
    read_video_data
)
import sys
import os
import imageio.v3 as imageio
import matplotlib.pyplot as plt
import cv2
from tifffile import imwrite
from tqdm import tqdm
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
import matplotlib as mpl

from amftrack.pipeline.launching.run import (
    run_transfer,
)
from amftrack.pipeline.launching.run_super import run_parallel_transfer
import dropbox
from amftrack.util.dbx import upload_folders, download, read_saved_dropbox_state, save_dropbox_state, load_dbx, \
    download, get_dropbox_folders, get_dropbox_video_folders
from subprocess import call
import logging
import datetime
import glob
import json
from amftrack.pipeline.launching.run_super import run_parallel

logging.basicConfig(stream=sys.stdout, level=logging.debug)
mpl.rcParams['figure.dpi'] = 300


def index_videos_dropbox(analysis_folder, video_folder, dropbox_folder, REDO_SCROUNGING=False):
    analysis_json = f"{analysis_folder}{dropbox_folder[6:]}all_folders_drop.json"
    if os.path.exists(analysis_json):
        all_folders_drop = pd.read_json(analysis_json)
    excel_json = f"{analysis_folder}{dropbox_folder[6:]}excel_drop.json"
    if os.path.exists(excel_json):
        excel_drop = pd.read_json(excel_json, typ='series')
    if not os.path.exists(analysis_json) or REDO_SCROUNGING:
        print("Redoing the dropbox scrounging, hold on tight.")
        all_folders_drop, excel_drop, txt_drop = get_dropbox_video_folders(dropbox_folder, True)

        clear_output(wait=False)
        print("Scrounging complete, merging files...")

        excel_addresses = np.array([re.search("^.*Plate.*\/.*Plate.*$", entry, re.IGNORECASE) for entry in excel_drop])
        excel_addresses = excel_addresses[excel_addresses != None]
        excel_addresses = [address.group(0) for address in excel_addresses]
        excel_drop = np.concatenate([excel_addresses, txt_drop])
        if not os.path.exists(f"{analysis_folder}{dropbox_folder[6:]}"):
            os.makedirs(f"{analysis_folder}{dropbox_folder[6:]}")
        all_folders_drop.to_json(analysis_json)
        pd.Series(excel_drop).to_json(excel_json)


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
        self.edges_frame = pd.concat([edg_obj.mean_data for edg_obj in self.edge_objs], axis =1).T.reset_index(drop=True)

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
            raise("Comparison symbol not recognised. Please use >, ==, or <.")

        filter_self.edge_objs = filter_self.edge_objs[filter_self.edges_frame.index.to_numpy()]
        is_video = filter_self.video_frame['unique_id'].isin(filter_self.edges_frame['unique_id'])
        filter_self.video_objs = filter_self.video_objs[is_video[is_video].index.values]
        filter_self.video_frame = filter_self.video_frame[is_video].reset_index(drop=True)
        filter_self.edges_frame = filter_self.edges_frame.reset_index(drop=True)
        return filter_self

    def filter_videos(self, column, compare, constant):
        return None

    def bin_values(self, column, bins, new_column_name ='edge_bin_values'):
        bin_series = pd.cut(self.edges_frame[column], bins, labels=False)
        self.edges_frame[new_column_name] = bin_series
        return bin_series
    
    def plot_histo(self, bins, column, ax=None):
        if ax is None:
            fig, ax = plt.subplots()
        ax.hist(self.edges_frame[column], bins = bins, edgecolor='black')
        ax.set_xlabel(column)
        ax.set_ylabel("counts")
        fig.tight_layout()
        return ax
        
    def plot_violins(self, column, bins, bin_separator='edge_bin_values', ax=None, c='tab:blue'):
        if ax is None:
            fig, ax = plt.subplots()
        violin_data = [ self.edges_frame[column][self.edges_frame[bin_separator] == i].astype(float) for i in range(len(bins))]
        violin_data_d = []
        for data in violin_data:
            if data.empty:
                violin_data_d.append([np.nan, np.nan])
            else:
                violin_data_d.append(data)
        violin_parts = ax.violinplot(dataset = violin_data_d)
        for vp in violin_parts['bodies']:
            vp.set_edgecolor('black')
            vp.set_facecolor(c)
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
