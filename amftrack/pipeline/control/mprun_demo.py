import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
from amftrack.pipeline.functions.post_processing.extract_study_zone import *
import pandas as pd
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
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.paths.directory import run_parallel, find_state, directory_scratch, directory_project
from amftrack.notebooks.analysis.util import * 
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
from collections import Counter
from IPython.display import clear_output
from amftrack.notebooks.analysis.data_info import *
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
plt.rcParams.update({
    "font.family": "verdana",
'font.weight' : 'normal',
'font.size': 20})
from amftrack.plotutil import plot_node_skel
from amftrack.notebooks.validation.util import *
from amftrack.pipeline.paths.directory import *
from amftrack.util import *
from amftrack.notebooks.post_processing.util import *
import pickle

def sum_of_lists(N):

    directory = directory_project
    update_analysis_info(directory)
    analysis_info = get_analysis_info(directory)
    select = analysis_info
    num = 1
    rows = [row for (index, row) in select.iterrows()]
    for index,row in enumerate(rows):
        path = f'{directory}{row["folder_analysis"]}'
        print(index,row["Plate"])
        try:
            a = np.load(f'{path}/center.npy')
        except:
            print(index,row["Plate"])
        if index == num:
            path_exp = f'{directory}{row["path_exp"]}'
            exp = pickle.load(open(path_exp, "rb"))
            exp.dates.sort()
            break
