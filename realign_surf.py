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
from extract_graph import dic_to_sparse, from_sparse_to_graph, generate_nx_graph, prune_graph, from_nx_to_tab, from_nx_to_tab_matlab,sparse_to_doc, connections_pixel_list_to_tab, transform_list, clean_degree_4
from time import sleep
from pycpd import RigidRegistration, DeformableRegistration
from realign import realign_final
plate = 9
directory = "/scratch/shared/mrozemul/Fiji.app/" 
listdir=os.listdir(directory) 
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
ss=[name.split('_')[0] for name in list_dir_interest]
ff=[name.split('_')[1] for name in list_dir_interest]
dates_datetime=[datetime(year=int(ss[i][:4]),month=int(ss[i][4:6]),day=int(ss[i][6:8]),hour=int(ff[i][0:2]),minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()
dates_datetime_chosen=dates_datetime[1:48]
dates = [f'{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}' for date in dates_datetime_chosen]
dilateds=[]
skels = []
skel_docs = []
graph_pos=[]
for date in dates:
    directory_name=f'2020{date}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    skel = read_mat(path_snap+'/Analysis/skeleton.mat')['skeleton']
    skels.append(skel)
    skel_doc = sparse_to_doc(skel)
    skel_docs.append(skel_doc)

skel_doc = skel_docs[0]
skel_aligned_t = skels[0]
skel_sparse = scipy.sparse.csc_matrix(skels[0])
directory_name=f'2020{dates[0]}_Plate{0 if plate<10 else ""}{plate}'
path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
sio.savemat(path_snap+'/Analysis/skeleton_realigned.mat',{'skeleton' : skel_sparse,'R' : np.array([[1,0],[0,1]]),'t' : np.array([0,0])})
for i,skel in enumerate(skel_docs[1:]):
    print('treatin',i)
    directory_name=f'2020{dates[i+1]}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    skel_aligned,skel_doc,R,t = realign_final(skel,skel_doc)
    skel_sparse = scipy.sparse.csc_matrix(skel_aligned)
    sio.savemat(path_snap+'/Analysis/skeleton_realigned.mat',{'skeleton' : skel_sparse,'R' : R,'t' : t})
#     plot_t_tp1([],[],None,None,skel_aligned[19000:21000,44000:46000],skel_aligned_t[19000:21000,44000:46000])
    skel_aligned_t = skel_aligned