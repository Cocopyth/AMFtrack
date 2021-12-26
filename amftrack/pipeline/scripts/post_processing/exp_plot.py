from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from amftrack.pipeline.functions.image_processing.extract_width_fun import *
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment
from amftrack.util import get_dates_datetime, get_dirname
import pickle
import networkx as nx
import pandas as pd
from amftrack.pipeline.paths.directory import directory_scratch
from path import path_code_dir
import os
import json
from datetime import datetime
from pymatreader import read_mat
import cv2
import matplotlib.pyplot as plt
from IPython.display import clear_output
from amftrack.plotutil import plot_t_tp1
from amftrack.notebooks.analysis.util import directory_scratch
import imageio
from amftrack.pipeline.functions.image_processing.experiment_class_surf import Experiment, save_graphs, load_graphs
from amftrack.transfer.functions.transfer import upload, zip_file

directory = str(sys.argv[1])
overwrite =  eval(sys.argv[2])
i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
run_info = pd.read_json(f'{directory_scratch}temp/{op_id}.json')
list_f,list_args = pickle.load(open(f'{directory_scratch}temp/{op_id}.pick', "rb"))
folder_list = list(run_info['folder_analysis'])
directory_name = folder_list[i]
select = run_info.loc[run_info['folder_analysis'] == directory_name]
row = [row for index, row in select.iterrows()][0]
plate_num = row['Plate']
path_exp = f'{directory}{row["path_exp"]}'
exp = pickle.load(open(path_exp, "rb"))
load_graphs(exp,indexes = [])
exp.dates.sort()
plate = exp.plate
run_inf = exp.folders
id_unique = (run_inf['Plate'].astype(str) + "_" + run_inf['CrossDate'].astype(str)).iloc[0]
folders = list(exp.folders['folder'])
folders.sort()



skels = []
ims = []
kernel = np.ones((5,5),np.uint8)
itera = 1
for folder in folders:
    directory_name=folder
    path_snap=directory+directory_name
    skel_info = read_mat(path_snap+'/Analysis/skeleton_pruned_compressed.mat')
    skel = skel_info['skeleton']
    skels.append(cv2.dilate(skel.astype(np.uint8),kernel,iterations = itera))
    im = read_mat(path_snap+'/Analysis/raw_image.mat')['raw']
    ims.append(im)
start=0
finish = len(exp.dates)
for i in range(start,finish):
    plt.close('all')
    clear_output(wait=True)
    plot_t_tp1([], [], None, None, skels[i], ims[i], save=f'{directory_scratch}temp/{plate_num}_im{i}.png',time=f't = {int(get_time(exp,0,i))}h')
img_array = []
for t in range(start,finish):
    img = cv2.imread(f'{directory_scratch}temp/{plate_num}_im{t}.png')
    img_array.append(img)
    
API = str(np.load(os.getenv('HOME')+'/pycode/API_drop.npy'))
dir_drop = 'prince_data'
path_movie = f'{directory_scratch}temp/{plate_num}.gif'
imageio.mimsave(path_movie, img_array,duration = 1)
upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_raw.gif',chunk_size=256 * 1024 * 1024)
path_movie = f'{directory_scratch}temp/{plate_num}.mp4'
imageio.mimsave(path_movie, img_array)
upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_raw.mp4',chunk_size=256 * 1024 * 1024)
skels = []
ims = []
kernel = np.ones((5,5),np.uint8)
itera = 2
for folder in folders:
    directory_name=folder
    path_snap=directory+directory_name
    skel_info = read_mat(path_snap+'/Analysis/skeleton_realigned_compressed.mat')
    skel = skel_info['skeleton']
    skels.append(cv2.dilate(skel.astype(np.uint8),kernel,iterations = itera))
start=0
finish = len(exp.dates)
for i in range(start,finish):
    plt.close('all')
    clear_output(wait=True)
    plot_t_tp1([], [], None, None, skels[i], skels[i], save=f'{directory_scratch}temp/{plate_num}_im{i}',time=f't = {int(get_time(exp,0,i))}h')
img_array = []
for t in range(start,finish):
    img = cv2.imread(f'{directory_scratch}temp/{plate_num}_im{t}.png')
    img_array.append(img)
path_movie = f'{directory_scratch}temp/{plate_num}.gif'
imageio.mimsave(path_movie, img_array,duration = 1)
upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_aligned.gif',chunk_size=256 * 1024 * 1024)
path_movie = f'{directory_scratch}temp/{plate_num}.mp4'
imageio.mimsave(path_movie, img_array)
upload(API,path_movie,f'/{dir_drop}/{id_unique}/movie_aligned.mp4',chunk_size=256 * 1024 * 1024)
