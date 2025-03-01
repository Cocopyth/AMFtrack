import ast
import os
from time import time

import imageio.v2 as imageio
import numpy as np
import pandas as pd
import scipy.io as sio
import scipy.sparse
from PIL import Image

from amftrack.pipeline.functions.image_processing.extract_skel import remove_holes, remove_component
from amftrack.ml.unet import get_model, make_segmentation_prediction
from amftrack.sparse_util import zhang_suen_thinning
from amftrack.util.sys import get_dirname, temp_path


def process(args):
    i = int(args[-1])
    op_id = int(args[-2])
    directory = str(args[1])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    directory_name = folder_list[i]
    model = get_model()
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
    dirName = os.path.join(path_snap, "Analysis")
    try:
        os.mkdir(dirName)
        print("Directory ", dirName, " Created ")
    except FileExistsError:
        print("Directory ", dirName, " already exists")
    t = time()
    xs = [c[0] for c in tileconfig[2]]
    ys = [c[1] for c in tileconfig[2]]
    name = tileconfig[0][0]
    imname = os.path.join("Img", name.split("/")[-1])
    im = imageio.imread(os.path.join(directory,directory_name,imname))
    dim = (
        int(np.max(ys) - np.min(ys)) + max(im.shape),
        int(np.max(xs) - np.min(xs)) + max(im.shape),
    )
    skel = np.zeros(dim, dtype=bool)
    for index, name in enumerate(tileconfig[0]):
        print(directory)

        # for index, name in enumerate(list_debug):
        imname = os.path.join("Img", name.split("/")[-1])
        im = Image.open(os.path.join(directory,directory_name,imname))
        # im = Image.open(os.path.join(directory_name, imname))
        print("segmenting")
        shape = im.size[1], im.size[0]

        segmented = make_segmentation_prediction(
            im,
            model,
            overlap=128
        )
        segmented = remove_holes(segmented)
        segmented = segmented.astype(np.uint8)
        segmented = remove_component(segmented)
        # imname = os.path.join("Img3", name.split("/")[-1])
        # imageio.imsave(os.path.join(directory_name, imname), segmented)
        boundaries = int(tileconfig[2][index][0] - np.min(xs)), int(
            tileconfig[2][index][1] - np.min(ys)
        )
        skel[
        boundaries[1]: boundaries[1] + shape[0],
        boundaries[0]: boundaries[0] + shape[1],
        ] += segmented.astype(bool)
    print("time_individual=", time() - t)
    t = time()
    skel = zhang_suen_thinning(skel)
    sio.savemat(
        path_snap + "/Analysis/skeleton.mat",
        {"skeleton": scipy.sparse.csc_matrix(skel)},
    )
    print("time_skelet=", time() - t)
    # im_fold = "Img3"
    # to_delete = os.path.join(directory_name, im_fold)
    # shutil.rmtree(to_delete)
