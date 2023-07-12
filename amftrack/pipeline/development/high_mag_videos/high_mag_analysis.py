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


class HighmagDataset(object):
    def __init__(self,
                 dataframe:pd.DataFrame):
        self.dataset = dataframe
        self.video_objs = [VideoDataset(row) for index, row in self.dataset.iterrows()]

    def bin_dataset(self, column, bins):
        return None

    def return_vid_objs(self):
        return None


class VideoDataset(object):
    def __init__(self,
                 series):
        self.dataset = series
        if os.path.exists(self.dataset['analysis_folder']+'edges_data.csv'):
            self.edges_frame = pd.read_csv(self.dataset['analysis_folder']+'edges_data.csv')
        else:
            print(f"Couldn't find the edges data file. Check analysis for {self.dataset['unique_id']}")
        self.edge_objs = [EdgeDataset(pd.concat([row, self.dataset])) for index, row in self.edges_frame.iterrows()]

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

    def scatter_speeds(self):
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
                 dataframe):
        self.mean_data = dataframe
        self.edge_name = self.mean_data['edge_name']
        self.time_data = pd.read_csv(self.mean_data['analysis_folder']+'edge '+self.mean_data['edge_name']+os.sep+self.mean_data['edge_name']+'_data.csv')

    def show_summary(self):
        summ_img = imageio.imread(self.mean_data['analysis_folder']+'edge '+self.mean_data['edge_name']+os.sep+self.mean_data['edge_name']+'_summary.png')
        fig = plt.figure()
        ax = fig.add_axes([0, 0, 1, 1], frameon=False, aspect=1)
        ax.imshow(summ_img)
        ax.set_axis_off()

    def plot_flux(self):
        fig, ax = plt.subplots()
        ax.plot(self.time_data['times'], self.time_data['flux_mean'])
        fig.tight_layout()

