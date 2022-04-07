from path import path_code_dir
import sys

sys.path.insert(0, path_code_dir)
from amftrack.util import get_dates_datetime, get_dirname
import pandas as pd
import ast
import scipy.io as sio
import cv2 as cv
import imageio
import numpy as np
import os
from time import time
from amftrack.pipeline.paths.directory import directory_scratch

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
directory = str(sys.argv[1])

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
mask = np.zeros(dim, dtype=np.uint8)

for index, name in enumerate(tileconfig[0]):
    imname = "/Img/" + name.split("/")[-1]
    im = imageio.imread(directory + directory_name + imname)
    im_cropped = im
    im_blurred = cv.blur(im_cropped, (200, 200))
    im_back_rem = (
        (im_cropped)
        / ((im_blurred == 0) * np.ones(im_blurred.shape) + im_blurred)
        * 120
    )
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
        tileconfig[2][index][1] - np.min(ys)
    )
    mask[
        boundaries[1] : boundaries[1] + shape[0],
        boundaries[0] : boundaries[0] + shape[1],
    ] = im_back_rem

output = mask
mask_compressed = cv.resize(output, (dim[1] // 5, dim[0] // 5))
sio.savemat(path_snap + "/Analysis/raw_image.mat", {"raw": mask_compressed})
