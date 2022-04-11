from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util.sys import get_dirname
import pandas as pd
import ast
from scipy import sparse
from datetime import datetime
from amftrack.pipeline.functions.image_processing.node_id import orient
import scipy.io as sio
import cv2
import imageio
import numpy as np
from skimage.filters import frangi
from skimage import filters
import scipy.sparse
import os
from time import time
from skimage.feature import hessian_matrix_det
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
)
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_tip_ext,
)
from amftrack.util.sys import get_dates_datetime, get_dirname
from amftrack.pipeline.paths.directory import directory_scratch

i = int(sys.argv[-1])
k = int(sys.argv[-2])
op_id = int(sys.argv[-3])
low = int(sys.argv[1])
high = int(sys.argv[2])
dist = int(sys.argv[3])
directory = str(sys.argv[4])

run_info = pd.read_json(f"{directory_scratch}temp/{op_id}.json")
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
path_snap = directory + directory_name
path_tile = path_snap + "/Img/TileConfiguration.txt.registered"
try:
    tileconfig = pd.read_table(
        path_tile,
        sep=";",
        skiprows=4,
        header=None,
        converters={2: ast.literal_eval},
        skipinitialspace=True,
    )
except:
    print("error_name")
    path_tile = path_snap + "/Img/TileConfiguration.registered.txt"
    tileconfig = pd.read_table(
        path_tile,
        sep=";",
        skiprows=4,
        header=None,
        converters={2: ast.literal_eval},
        skipinitialspace=True,
    )
dirName = path_snap + "/Analysis"
shape = (3000, 4096)
try:
    os.mkdir(path_snap + "/Analysis")
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")
t = time()
xs = [c[0] for c in tileconfig[2]]
ys = [c[1] for c in tileconfig[2]]
dim = (int(np.max(ys) - np.min(ys)) + 4096, int(np.max(xs) - np.min(xs)) + 4096)
ims = []
for name in tileconfig[0][k]:
    imname = "/Img/" + name.split("/")[-1]
    #     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
    ims.append(imageio.imread(directory + directory_name + imname))
for index, im in enumerate(ims):
    print(index)
    segmented = extract_skel_tip_ext(im, low, high, dist)
    np.save(f"{directory_scratch}temp/{directory+directory_name+imname}", dilated)
