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
import imageio.v2 as imageio
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
from amftrack.util.dbx import upload_folders, download, read_saved_dropbox_state, save_dropbox_state, load_dbx, download, get_dropbox_folders, get_dropbox_video_folders
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
                dataframe):
        self.dataset = dataframe
        
    def bin_dataset(self, column, bins):
        return None
        
    def return_vid_objs(self):
        return None
    
class VideoDataset(object):
    def __init__(self,
                series):
        self. dataset = series
        
class EdgeDataset(object):
    def __init__(self,
                 dataframe):
        self.data = dataframe
        