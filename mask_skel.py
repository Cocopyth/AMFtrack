from util import get_path, get_dates_datetime, get_dirname
import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from extract_graph import generate_nx_graph, transform_list, generate_skeleton, generate_nx_graph_from_skeleton, from_connection_tab, from_nx_to_tab
from node_id import whole_movement_identification, second_identification
import ast
from plotutil import plot_t_tp1, compress_skeleton
from scipy import sparse
from sparse_util import dilate, zhangSuen
from realign import realign
from datetime import datetime,timedelta
from node_id import orient
import pickle
from matplotlib.widgets import CheckButtons
import scipy.io as sio
import imageio
from pymatreader import read_mat
from matplotlib import colors
from copy import deepcopy,copy
from collections import Counter
import cv2 
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage.morphology import thin
from skimage import data, filters
from random import choice
import scipy.sparse
import os
from time import time
from skimage.feature import hessian_matrix_det
import sys
from extract_graph import dic_to_sparse, from_sparse_to_graph, generate_nx_graph, prune_graph, from_nx_to_tab, from_nx_to_tab_matlab,sparse_to_doc, connections_pixel_list_to_tab, transform_list, clean_degree_4
from sparse_util import dilate, zhangSuen


i = int(sys.argv[-1])
plate = int(sys.argv[1])
thresh = int(sys.argv[2])
from directory import directory

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime
dates = dates_datetime_chosen
date = dates[i]
directory_name = get_dirname(date, plate)
path_snap=directory+directory_name
path_tile=path_snap+'/Img/TileConfiguration.txt.registered'
try:
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
except:
    print('error_name')
    path_tile=path_snap+'/Img/TileConfiguration.registered.txt'
    tileconfig = pd.read_table(path_tile,sep=';',skiprows=4,header=None,converters={2 : ast.literal_eval},skipinitialspace=True)
dirName = path_snap+'/Analysis'
shape = (3000,4096)
try:
    os.mkdir(path_snap+'/Analysis') 
    print("Directory " , dirName ,  " Created ")
except FileExistsError:
    print("Directory " , dirName ,  " already exists")  
t=time()
xs =[c[0] for c in tileconfig[2]]
ys =[c[1] for c in tileconfig[2]]
dim = (int(np.max(ys)-np.min(ys))+4096,int(np.max(xs)-np.min(xs))+4096)
ims = []
for name in tileconfig[0]:
    imname = '/Img/'+name.split('/')[-1]
#     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
    ims.append(imageio.imread(directory+directory_name+imname))
mask = np.zeros(dim,dtype=np.uint8)
for index,im in enumerate(ims):
    im_cropped = im
    im_blurred =cv2.blur(im_cropped, (100, 100))
    boundaries = int(tileconfig[2][index][0]-np.min(xs)),int(tileconfig[2][index][1]-np.min(ys))
    mask[boundaries[1]:boundaries[1]+shape[0],boundaries[0]:boundaries[0]+shape[1]] += im_blurred>thresh
    
skel_info = read_mat(path_snap+'/Analysis/skeleton.mat')
skel = skel_info['skeleton']
masker = mask>0
kernel = np.ones((100,100),np.uint8)
output = 1-cv2.dilate(1-masker.astype(np.uint8),kernel,iterations = 1)
result = output*np.array(skel.todense())
sio.savemat(path_snap+'/Analysis/skeleton_masked.mat',{'skeleton' : scipy.sparse.csc_matrix(result)})
compressed = cv2.resize(result,(dim[1]//5,dim[0]//5))
sio.savemat(path_snap+'/Analysis/skeleton_masked_compressed.mat',{'skeleton' : compressed})
mask_compressed= cv2.resize(output,(dim[1]//5,dim[0]//5))
sio.savemat(path_snap+'/Analysis/mask.mat',{'mask' : mask_compressed})

          