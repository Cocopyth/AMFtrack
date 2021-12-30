import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
import pandas as pd
from amftrack.util import get_dates_datetime, get_dirname, get_plate_number, get_postion_number

import ast
from amftrack.plotutil import plot_t_tp1
from scipy import sparse
from datetime import datetime
from amftrack.pipeline.functions.image_processing.node_id import orient
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
from amftrack.pipeline.pipeline.paths.directory import run_parallel, find_state, directory_scratch, directory_project, path_code
from IPython.display import clear_output
from amftrack.notebooks.analysis.data_info import *
def get_time(dates_datetimes,i,j):
    seconds = (dates_datetimes[j]-dates_datetimes[i]).total_seconds()
    return(seconds/3600)
directory = directory_project

results={}
for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        plate_label = plate_number[inst] 
        plate,begin,end = inst
        print(inst)
        dates_datetime = get_dates_datetime(directory,plate)
        dates_datetime_chosen=dates_datetime[begin:end+1]
        dates = [f'{date.year}{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}' for date in dates_datetime_chosen]
        skels = []
        ims = []
        kernel = np.ones((5,5),np.uint8)
        itera = 1
        for date in dates:
            directory_name=f'{date}_Plate{0 if plate<10 else ""}{plate}'
            path_snap=directory+directory_name
            skel_info = read_mat(path_snap+'/Analysis/skeleton_pruned_compressed.mat')
            skel = skel_info['skeleton']
            skels.append(cv2.dilate(skel.astype(np.uint8),kernel,iterations = itera))
            im = read_mat(path_snap+'/Analysis/raw_image.mat')['raw']
            ims.append(im)
        start=0
        finish = end-begin
        for i in range(start,finish):
            plt.close('all')
            clear_output(wait=True)
            plot_t_tp1([], [], None, None, skels[i], ims[i], save=f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/im{i}',time=f't = {int(get_time(dates_datetime_chosen,0,i))}h')
        img_array = []
        for t in range(start,finish):
            img = cv2.imread(f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/im{t}.png')
            img_array.append(img)
        imageio.mimsave(f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/movie{plate_label}temp_ {begin}_{end}_{dates[start]}_{dates[finish]}.gif', img_array,duration = 1)
        

results={}
for treatment in treatments.keys():
    insts = treatments[treatment]
    for inst in insts:
        print(inst)
        plate_label = plate_number[inst] 
        plate,begin,end = inst
        dates_datetime = get_dates_datetime(directory,plate)
        dates_datetime_chosen=dates_datetime[begin:end+1]
        dates = [f'{date.year}{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}' for date in dates_datetime_chosen]
        skels = []
        ims = []
        kernel = np.ones((5,5),np.uint8)
        itera = 2
        for date in dates:
            directory_name=f'{date}_Plate{0 if plate<10 else ""}{plate}'
            path_snap=directory+directory_name
            skel_info = read_mat(path_snap+'/Analysis/skeleton_realigned_compressed.mat')
            skel = skel_info['skeleton']
            skels.append(cv2.dilate(skel.astype(np.uint8),kernel,iterations = itera))
            im = read_mat(path_snap+'/Analysis/raw_image.mat')['raw']
            ims.append(im)
        start=0
        finish = end-begin
        for i in range(start,finish):
            plt.close('all')
            clear_output(wait=True)
            plot_t_tp1([], [], None, None, skels[i], skels[i], save=f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/im*{i}',time=f't = {int(get_time(dates_datetime_chosen,0,i))}h')
        img_array = []
        for t in range(start,finish):
            img = cv2.imread(f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/im*{t}.png')
            img_array.append(img)
        imageio.mimsave(f'/home/cbisot/pycode/MscThesis/amftrack/notebooks/plotting/Figure/movie*{plate_label}temp_ {begin}_{end}_{dates[start]}_{dates[finish]}.gif', img_array,duration = 1)