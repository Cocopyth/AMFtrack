import sys
import os

from amftrack.util.sys import get_dirname, temp_path
import pandas as pd
import ast
from scipy import sparse
import scipy.io as sio
import cv2
import imageio.v2 as imageio
import numpy as np
import scipy.sparse
import os
from time import time
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)

from amftrack.util.sys import get_dates_datetime, get_dirname
import shutil

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
hyph_width = int(sys.argv[1])
perc_low = float(sys.argv[2])
perc_high = float(sys.argv[3])
minlow = float(sys.argv[4])
minhigh = float(sys.argv[5])

directory = str(sys.argv[6])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
print(directory_name)
run_back_sub(directory, directory_name)
path_snap = os.path.join(directory, directory_name)
path_tile = os.path.join(path_snap, "Img/TileConfiguration.txt.registered")
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
    path_tile = os.path.join(path_snap, "Img/TileConfiguration.registered.txt")
    tileconfig = pd.read_table(
        path_tile,
        sep=";",
        skiprows=4,
        header=None,
        converters={2: ast.literal_eval},
        skipinitialspace=True,
    )
dirName = path_snap + "/Analysis"
try:
    os.mkdir(path_snap + "/Analysis")
    print("Directory ", dirName, " Created ")
except FileExistsError:
    print("Directory ", dirName, " already exists")
t = time()
xs = [c[0] for c in tileconfig[2]]
ys = [c[1] for c in tileconfig[2]]
name = tileconfig[0][0]
imname = "/Img3/" + name.split("/")[-1]
im = imageio.imread(directory + directory_name + imname)
dim = (
    int(np.max(ys) - np.min(ys)) + max(im.shape),
    int(np.max(xs) - np.min(xs)) + max(im.shape),
)
ims = []
skel = np.zeros(dim, dtype=np.uint8)
params = [30]
for index, name in enumerate(tileconfig[0]):
    # for index, name in enumerate(list_debug):
    print(name)
    imname = "/Img3/" + name.split("/")[-1]
    im = imageio.imread(directory + directory_name + imname)
    imname2 = "/Img/" + name.split("/")[-1]
    im2 = imageio.imread(directory + directory_name + imname2)
    bowled2 = bowler_hat(-im2, 32, params)
    im[bowled2 <= 0.09] = np.maximum(im[bowled2 <= 0.09], 250)
    shape = im.shape
    print("segmenting")
    segmented = extract_skel_new_prince(im, [hyph_width], perc_low, perc_high, minlow,minhigh)
    # low = np.percentile(-im+255, perc_low)
    # high = np.percentile(-im+255, perc_high)
    # segmented = filters.apply_hysteresis_threshold(-im+255, low, high)
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
        tileconfig[2][index][1] - np.min(ys)
    )
    skel[
        boundaries[1] : boundaries[1] + shape[0],
        boundaries[0] : boundaries[0] + shape[1],
    ] += segmented
print("number to reduce : ", np.sum(skel > 0), np.sum(skel <= 0))
skeletonized = cv2.ximgproc.thinning(np.array(255 * (skel > 0), dtype=np.uint8))
skel_sparse = sparse.lil_matrix(skel)
sio.savemat(
    path_snap + "/Analysis/skeleton.mat",
    {"skeleton": scipy.sparse.csc_matrix(skeletonized)},
)
compressed = cv2.resize(skeletonized, (dim[1] // 5, dim[0] // 5))
sio.savemat(path_snap + "/Analysis/skeleton_compressed.mat", {"skeleton": compressed})
print("time=", time() - t)
im_fold = "/Img3"
to_delete = directory + directory_name + im_fold
shutil.rmtree(to_delete)
