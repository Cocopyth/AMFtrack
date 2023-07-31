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
from amftrack.pipeline.functions.image_processing.blob_detection import (
    detect_blobs,
    plot_blobs_upload,
)
from random import choice
from amftrack.util.sys import get_dates_datetime, get_dirname
import shutil

i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
directory = str(sys.argv[1])
run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder_list = list(run_info["folder"])
folder_list.sort()
directory_name = folder_list[i]
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

xs = [c[0] for c in tileconfig[2]]
ys = [c[1] for c in tileconfig[2]]
complete_spores = []
choosen_picture = choice(tileconfig[0])
for index, name in enumerate(tileconfig[0]):
    imname = "/Img/" + name.split("/")[-1]
    im = imageio.imread(directory + directory_name + imname)
    blobs = detect_blobs(im)
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
        tileconfig[2][index][1] - np.min(ys)
    )
    for blob in blobs:
        x, y, r = blob
        x_tot = x + boundaries[0]
        y_tot = y + boundaries[1]
        complete_spores.append((x_tot, y_tot, r))
    # if name == choosen_picture:
    #     if len(blobs) > 0:
    #         plot_blobs_upload(im)

complete_spores = np.array(complete_spores)
# removing duplicate due to image overlap
to_remove = set()
for i in range(len(complete_spores)):
    distances = np.linalg.norm(complete_spores[i][:2] - complete_spores[:, :2], axis=1)
    if len(distances) > 1:
        index = np.argpartition(distances, 1)[1]
        if distances[index] < 10 and index < i:
            to_remove.add((i))
spore_filtered = np.delete(complete_spores, list(to_remove), axis=0)
sio.savemat(path_snap + "/Analysis/spores.mat", {"spores": spore_filtered})
