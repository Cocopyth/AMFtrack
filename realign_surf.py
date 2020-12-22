from util import get_path
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
import scipy.sparse
from realign import transform_skeleton_final

plate = int(sys.argv[1])
begin = int(sys.argv[2])
end = int(sys.argv[3])
directory = "/scratch/shared/mrozemul/Fiji.app/" 
listdir=os.listdir(directory) 
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
ss=[name.split('_')[0] for name in list_dir_interest]
ff=[name.split('_')[1] for name in list_dir_interest]
dates_datetime=[datetime(year=int(ss[i][:4]),month=int(ss[i][4:6]),day=int(ss[i][6:8]),hour=int(ff[i][0:2]),minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()

dates_datetime_chosen=dates_datetime[begin:end]
dates = [f'{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}' for date in dates_datetime_chosen]
dilateds=[]
skels = []
skel_docs = []
directory_name=f'2020{dates[0]}_Plate{0 if plate<10 else ""}{plate}'
path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
skel_info = read_mat(path_snap+'/Analysis/skeleton.mat')
skel = skel_info['skeleton']
skels.append(skel)
skel_doc = sparse_to_doc(skel)
skel_docs.append(skel_doc)
Rs=[]
ts=[]
for date in dates[1:]:
    directory_name=f'2020{date}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    skel_info = read_mat(path_snap+'/Analysis/skeleton.mat')
    skel = skel_info['skeleton']
    skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)
    transform = sio.loadmat(path_snap+'/Analysis/transform.mat')
    R,t = transform['R'],transform['t']
    Rs.append(R)
    ts.append(t)

skel_doc = skel_docs[0]
skel_aligned_t = skels[0]
skel_sparse = scipy.sparse.csc_matrix(skels[0])
directory_name=f'2020{dates[0]}_Plate{0 if plate<10 else ""}{plate}'
path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
sio.savemat(path_snap+'/Analysis/skeleton_realigned.mat',{'skeleton' : skel_sparse,'R' : np.array([[1,0],[0,1]]),'t' : np.array([0,0])})
R0 = np.array([[1,0],[0,1]])
t0 = np.array([0,0])
for i,skel in enumerate(skel_docs):
    print('treatin',i)
    directory_name=f'2020{dates[i]}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    skel_aligned = transform_skeleton_final(skel,R0,t0)
    skel_sparse = scipy.sparse.csc_matrix(skel_aligned)
    sio.savemat(path_snap+'/Analysis/skeleton_realigned.mat',{'skeleton' : skel_sparse,'R' : R0,'t' : t0})
    R0 = np.dot(np.transpose(Rs[i]),R0)
    t0 = -np.dot(ts[i],np.transpose(Rs[i]))+np.dot(t0,np.transpose(Rs[i]))