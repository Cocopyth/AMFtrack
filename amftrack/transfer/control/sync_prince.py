import os
import sys

sys.path.insert(0, os.getenv("HOME") + "/pycode/MscThesis/")
# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util.sys import (
    get_dates_datetime,
    get_dirname,
    get_data_info,
    update_plate_info,
    get_current_folders,
    get_folders_by_plate_id,
)

import ast
from amftrack.plotutil import plot_t_tp1
from scipy import sparse
from datetime import datetime
import pickle
import scipy.io as sio
from pymatreader import read_mat
from matplotlib import colors
import cv2
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage import filters
from random import choice
import scipy.sparse
import os
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    sparse_to_doc,
)
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.pipeline.paths.directory import (
    run_parallel_transfer,
    find_state,
    directory_scratch,
    directory_project,
    directory_archive,
)
import dropbox
from amftrack.util.dbx import upload, zip_file, sync_fold
from subprocess import call
from tqdm.autonotebook import tqdm

directory = r"/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$/Data/Prince/Images/"
update_plate_info(directory,local=True)

all_folders = get_current_folders(directory,local=True)
directory2 = r"/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing"
directory2 = r"/mnt/sun-temp/TEMP/PRINCE_syncing"

folders = all_folders
run_info = folders.copy()
target = directory2
folder_list = list(run_info["total_path"])
with tqdm(total=len(folder_list), desc="transferred") as pbar:
    for folder in folder_list:
        origin = folder
        sync_fold(origin, target)
        pbar.update(1)
