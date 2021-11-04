import sys  
sys.path.insert(0, '/home/cbisot/pycode/MscThesis/')
sys.path.append( '/home/cbisot/pycode/MscThesis/amftrack/pipeline/functions')

from amftrack.notebooks.analysis.util import *
from amftrack.pipeline.pipeline.paths.directory import path_code, directory_scratch, directory_project
from amftrack.notebooks.analysis.data_info import *
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
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment, Edge, Node
from amftrack.pipeline.pipeline.paths.directory import run_parallel, find_state, directory_scratch, directory_project
from amftrack.notebooks.analysis.util import *
from scipy import stats
from scipy.ndimage.filters import uniform_filter1d
from scipy import sparse
import numpy as np
import os
from datetime import datetime, timedelta
import pandas as pd
path_code = "/home/cbisot/pycode/"
plate_info = pd.read_excel(path_code + 'MscThesis/plate_info/SummaryAnalizedPlates.xlsx',engine='openpyxl',header=3,)

def get_Pside(plate_number):
    return(plate_info.loc[plate_info['Plate #'] == plate_number]['P-bait'])

def get_raw(exp,t):
    date = exp.dates[t]
    directory_name = get_dirname(date,exp.plate)
    path_snap = exp.directory + directory_name
    im = read_mat(path_snap+'/Analysis/raw_image.mat')['raw']
    return(im)

def get_pos_baits(exp,t):
    raw_im = get_raw(exp,t)
    image = cv2.blur(raw_im,(100,100))
    output = image.copy()
    gray = image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 2, 3000,minRadius = 200,maxRadius = 600,param1=10)
    # ensure at least some circles were found
    print('circles', circles)
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
    for (x, y, r) in circles[:2]:
        # draw the circle in the output image, then draw a rectangle
        # corresponding to the center of the circle
        cv2.circle(output, (x, y), r, (0, 255, 0), 4)
        cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
        # show the output image
#     fig = plt.figure(figsize=(10,9))
#     ax = fig.add_subplot(111)
#     ax.imshow(output)
    return(circles[:2])

def get_pos_baits_aligned(exp,t):
    pos_bait = get_pos_baits(exp,t)
    date = exp.dates[t]
    directory_name = get_dirname(date, exp.plate)
    path_snap = exp.directory + directory_name
    path_tile = path_snap + "/Img/TileConfiguration.txt.registered"
    skel = read_mat(path_snap + "/Analysis/skeleton_pruned_realigned.mat")
    Rot = skel["R"]
    trans = skel["t"]
    print(Rot,trans)
    real_pos=[]
    for x,y,r in pos_bait:
        compression = 5
        xs,ys = x*compression,y*compression
        rottrans = np.dot(Rot, np.array([ys, xs])) + trans
#         rottrans = np.array([xs, ys])
        xs, ys = round(rottrans[0]), round(rottrans[1])
        real_pos.append((xs,ys))
    pos_real = {}
    if real_pos[0][1]>=real_pos[1][1]:
        pos_real['right'] = real_pos[0]
        pos_real['left'] = real_pos[1]
    else:
        pos_real['right'] = real_pos[1]
        pos_real['left'] = real_pos[0]
    return(pos_real)
        

for treatment in ['25','baits']:
    insts = treatments[treatment]
    for inst in insts:
        exp = get_exp(inst,directory_project)
        pos_baits_time = []
        for t in range(exp.ts):
            baits=get_pos_baits_aligned(exp,t)
            pos_baits_time.append(baits)
        pickle.dump(pos_baits_time, open(f'{path_code}/MscThesis/Results/baits_{inst}.pick', "wb"))