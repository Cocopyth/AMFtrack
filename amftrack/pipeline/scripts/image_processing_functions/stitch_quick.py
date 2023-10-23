from amftrack.util.sys import path_code
import sys

from amftrack.util.sys import temp_path
import pandas as pd
import ast
import cv2
import imageio
import numpy as np
import os


i = int(sys.argv[-1])
op_id = int(sys.argv[-2])
compress = int(sys.argv[1])


run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
folder = run_info.iloc[i]["total_path"]
path_tile = (
    f"{path_code}pipeline/scripts/stitching_loops/TileConfiguration.txt.registered"
)
tileconfig = pd.read_table(
    path_tile,
    sep=";",
    skiprows=4,
    header=None,
    converters={2: ast.literal_eval},
    skipinitialspace=True,
)
xs = [c[0] for c in tileconfig[2]]
ys = [c[1] for c in tileconfig[2]]
dim = (int(np.max(ys) - np.min(ys)) + 5500, int(np.max(xs) - np.min(xs)) + 5500)
dim = (dim[0] // compress, dim[1] // compress)
mask = np.zeros(dim, dtype=np.uint8)

for index, name in enumerate(tileconfig[0]):
    imname = "Img/" + name.split("/")[-1]
    path = os.path.join(folder, imname)
    im = imageio.imread(path)
    shape = im.shape
    shape = shape[0] // compress, shape[1] // compress
    im_cropped = im
    im_back_rem = im
    im_back_rem = cv2.resize(im_back_rem, (shape[1], shape[0]))
    boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
        tileconfig[2][index][1] - np.min(ys)
    )
    boundaries = boundaries[0] // compress, boundaries[1] // compress
    mask[
        boundaries[1] : boundaries[1] + shape[0],
        boundaries[0] : boundaries[0] + shape[1],
    ] = im_back_rem

imageio.imsave(os.path.join(folder, "StitchedImage.tif"), mask)
