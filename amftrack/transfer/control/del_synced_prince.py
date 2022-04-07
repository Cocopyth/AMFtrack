import os  
import sys  
sys.path.insert(0, os.getenv('HOME')+'/pycode/MscThesis/')
# sys.path.insert(0,r'C:\Users\coren\Documents\PhD\Code\AMFtrack')

import pandas as pd
from amftrack.util import get_dates_datetime, get_dirname, get_data_info, update_plate_info, \
get_current_folders, get_folders_by_plate_id

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
from amftrack.pipeline.functions.image_processing.extract_graph import from_sparse_to_graph, generate_nx_graph, sparse_to_doc
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.pipeline.paths.directory import run_parallel_transfer, find_state, directory_scratch, directory_project, directory_archive
import dropbox
from amftrack.transfer.functions.transfer import upload, zip_file
from subprocess import call
from tqdm.autonotebook import tqdm
import checksumdir
import shutil

# directory_origin = r'/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing/'
directory_origin = r'/run/user/357100554/gvfs/smb-share:server=prince.amolf.nl,share=d$/Data/Prince/Images/'
directory_target = r'/mnt/sun/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE_syncing/'
directory_target = r'/mnt/sun-temp/TEMP/PRINCE_syncing/'

update_plate_info(directory_origin)
update_plate_info(directory_target)

all_folders_origin = get_current_folders(directory_origin)
all_folders_target = get_current_folders(directory_target)
run_info = all_folders_target.copy()
folder_list = list(run_info['folder'])
with tqdm(total=len(folder_list), desc="deleted") as pbar:
    for folder in folder_list:
        origin = all_folders_origin.loc[all_folders_origin['folder']==folder]['total_path']
        if len(origin)>0:
            origin = origin.iloc[0]
        else:
            # print(folder)
            continue
        target = all_folders_target.loc[all_folders_target['folder']==folder]['total_path'].iloc[0]     
        check_or = checksumdir.dirhash(origin)
        check_targ = checksumdir.dirhash(target)
        print(folder,(check_or==check_targ))        
        
        if check_or==check_targ and origin!= target:
            shutil.rmtree(origin)
        pbar.update(1)