from path import path_code_dir
import sys  
sys.path.insert(0, path_code_dir)
from sample.util import get_dates_datetime, get_dirname
import pandas as pd
import ast
import scipy.io as sio
import cv2
import imageio
import numpy as np
import os
from time import time

i = int(sys.argv[-1])
plate = int(sys.argv[1])
directory = str(sys.argv[2])

dates_datetime = get_dates_datetime(directory,plate)
dates_datetime.sort()
dates_datetime_chosen = dates_datetime
dates = dates_datetime_chosen
date = dates[i]
directory_name = get_dirname(date, plate)
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
for name in tileconfig[0]:
    imname = "/Img/" + name.split("/")[-1]
    #     ims.append(imageio.imread('//sun.amolf.nl/shimizu-data/home-folder/oyartegalvez/Drive_AMFtopology/PRINCE'+date_plate+plate_str+'/Img/'+name))
    ims.append(imageio.imread(directory + directory_name + imname))
mask = np.zeros(dim, dtype=np.uint8)
for index, im in enumerate(ims):
    im_cropped = im
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
        tileconfig[2][index][1] - np.min(ys)
    )
    mask[
        boundaries[1] : boundaries[1] + shape[0],
        boundaries[0] : boundaries[0] + shape[1],
    ] = im

output = mask
mask_compressed = cv2.resize(output, (dim[1] // 5, dim[0] // 5))
sio.savemat(path_snap + "/Analysis/raw_image.mat", {"raw": mask_compressed})
