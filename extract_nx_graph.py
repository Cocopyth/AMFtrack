import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
from scipy import sparse
import cv2
from pymatreader import read_mat
# from extract_graph import dic_to_sparse
from util import get_path, shift_skeleton
from plotutil import show_im,overlap, show_im_rgb, plot_nodes, plot_nodes_from_list,plot_t_tp1
from extract_graph import generate_graph_tab_from_skeleton,generate_nx_graph_from_skeleton,generate_skeleton,clean
import networkx as nx
from node_id import second_identification, whole_movement_identification,first_identification,relabel_nodes, clean_nodes, orient
from extract_graph import dic_to_sparse, from_sparse_to_graph, generate_nx_graph, prune_graph, from_nx_to_tab, from_nx_to_tab_matlab,sparse_to_doc, connections_pixel_list_to_tab, transform_list, clean_degree_4
from sparse_util import dilate, zhangSuen
from realign import realign, reconnect, realign2
from util import get_path
import pandas as pd
from datetime import datetime,timedelta
import ast
import os
import scipy.sparse
import scipy.io as sio
import pickle
import sys

plate = int(sys.argv[1])
i = int(sys.argv[-1])
threshold = int(sys.argv[2])
directory = "/scratch/shared/mrozemul/Fiji.app/" 
listdir=os.listdir(directory) 
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
ss=[name.split('_')[0] for name in list_dir_interest]
ff=[name.split('_')[1] for name in list_dir_interest]
dates_datetime=[datetime(year=int(ss[i][:4]),month=int(ss[i][4:6]),day=int(ss[i][6:8]),hour=int(ff[i][0:2]),minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()
date_datetime=dates_datetime[i]
date = f'{0 if date_datetime.month<10 else ""}{date_datetime.month}{0 if date_datetime.day<10 else ""}{date_datetime.day}_{0 if date_datetime.hour<10 else ""}{date_datetime.hour}{0 if date_datetime.minute<10 else ""}{date_datetime.minute}'
directory_name=f'2020{date}_Plate{0 if plate<10 else ""}{plate}'
path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
skel = read_mat(path_snap+'/Analysis/skeleton_realigned.mat')['skeleton']
skeleton = scipy.sparse.dok_matrix(skel)
    
# nx_graph_poss=[generate_nx_graph(from_sparse_to_graph(skeleton)) for skeleton in skels_aligned]
# nx_graphs_aligned=[nx_graph_pos[0] for nx_graph_pos in nx_graph_poss]
# poss_aligned=[nx_graph_pos[1] for nx_graph_pos in nx_graph_poss]
# nx_graph_pruned=[clean_degree_4(prune_graph(nx_graph),poss_aligned[i])[0] for i,nx_graph in enumerate(nx_graphs_aligned)]
nx_graph,pos = generate_nx_graph(from_sparse_to_graph(skeleton))
nx_graph_pruned = clean_degree_4(prune_graph(nx_graph,threshold),pos)[0]
directory_name=f'2020{date}_Plate{0 if plate<10 else ""}{plate}'
path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
path_save = path_snap +'/Analysis/nx_graph_pruned.p'
pickle.dump((nx_graph_pruned,pos),open( path_save, "wb" ))
skeleton = generate_skeleton(nx_graph_pruned,(26322, 49527))
sio.savemat(path_snap+'/Analysis/skeleton_realigned_pruned.mat',{'skeleton' : scipy.sparse.csc_matrix(skeleton)})