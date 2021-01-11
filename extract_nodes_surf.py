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
begin = int(sys.argv[2])
end = int(sys.argv[3])
directory = "/scratch/shared/mrozemul/Fiji.app/" 
listdir=os.listdir(directory) 
list_dir_interest=[name for name in listdir if name.split('_')[-1]==f'Plate{0 if plate<10 else ""}{plate}']
ss=[name.split('_')[0] for name in list_dir_interest]
ff=[name.split('_')[1] for name in list_dir_interest]
dates_datetime=[datetime(year=int(ss[i][:4]),month=int(ss[i][4:6]),day=int(ss[i][6:8]),hour=int(ff[i][0:2]),minute=int(ff[i][2:4])) for i in range(len(list_dir_interest))]
dates_datetime.sort()
dates_datetime_chosen=dates_datetime[begin:end+1]
dates = [f'{0 if date.month<10 else ""}{date.month}{0 if date.day<10 else ""}{date.day}_{0 if date.hour<10 else ""}{date.hour}{0 if date.minute<10 else ""}{date.minute}' for date in dates_datetime_chosen]
nx_graph_pos = []
for date in dates:
    directory_name=f'2020{date}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    path_save = path_snap +'/Analysis/nx_graph_pruned.p'
    nx_graph_pos.append(pickle.load(open( path_save, "rb" )))
nx_graph_pruned=[c[0] for c in nx_graph_pos]
poss_aligned = [c[1] for c in nx_graph_pos]
downstream_graphs = []
downstream_pos = []
begin=len(dates)-1
downstream_graphs=[nx_graph_pruned[begin]]
downstream_poss=[poss_aligned[begin]]
for i in range (begin-1,-1,-1):
    print("i=",i)
    new_graphs,new_poss = second_identification(nx_graph_pruned[i],downstream_graphs[0],poss_aligned[i],downstream_poss[0],50,downstream_graphs[1:],downstream_poss[1:],tolerance=30)
    downstream_graphs=new_graphs
    downstream_poss=new_poss
    
nx_graph_pruned=downstream_graphs
poss_aligned=downstream_poss
for i,g in enumerate(nx_graph_pruned):
    directory_name=f'2020{dates[i]}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    path_save = path_snap +'/Analysis/nx_graph_pruned_labeled.p'
    pos = poss_aligned[i]
    pickle.dump((g,pos),open( path_save, "wb" ))
    
for i, date in enumerate(dates):
    tab = from_nx_to_tab(nx_graph_pruned[i],poss_aligned[i])
    directory_name=f'2020{dates[i]}_Plate{0 if plate<10 else ""}{plate}'
    path_snap='/scratch/shared/mrozemul/Fiji.app/'+directory_name
    path_save = path_snap +'/Analysis/graph_full_labeled.mat'
    sio.savemat(path_save, {name: col.values for name, col in tab.items()})
