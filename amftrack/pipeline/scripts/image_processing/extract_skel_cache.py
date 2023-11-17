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
import concurrent.futures
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[logging.FileHandler("debug.log"),
                              logging.StreamHandler(sys.stdout)])

def process(args):

    i = int(args[-1])
    op_id = int(args[-2])
    hyph_width = int(args[1])
    perc_low = float(args[2])
    perc_high = float(args[3])
    minlow = float(args[4])
    minhigh = float(args[5])

    directory = str(args[6])

    run_info = pd.read_json(f"{temp_path}/{op_id}.json", dtype={"unique_id": str})
    folder_list = list(run_info["folder"])
    folder_list.sort()
    directory_name = folder_list[i]
    print(directory_name)
    # run_back_sub(directory, directory_name)
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
    params = [30]

    def process_image(name):
        try:
            logging.info(f"Processing image: {name}")
            print(name)
            imname = "/Img3/" + name.split("/")[-1]
            im = imageio.imread(directory + directory_name + imname)
            imname2 = "/Img/" + name.split("/")[-1]
            im2 = imageio.imread(directory + directory_name + imname2)
            bowled2 = bowler_hat(-im2, 32, params)
            im[bowled2 <= 0.09] = np.maximum(im[bowled2 <= 0.09], 250)
            print("segmenting")
            segmented = extract_skel_new_prince(
                im, [hyph_width], perc_low, perc_high, minlow, minhigh
            )
            path = directory + directory_name + imname
            print(path)
            imageio.imwrite(path, segmented)
            logging.info(f"Image processed: {name}")
        except Exception as e:
            logging.error(f"Error processing image {name}: {e}", exc_info=True)
        return(None)
    print('processing')
    t = time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=64) as executor:
        # Adjust the max_workers parameter as needed
        names = [name for _, name in enumerate(tileconfig[0])]
        executor.map(process_image, names)
        # futures = [executor.submit(process_image, name) for _, name in enumerate(tileconfig[0][begin:begin+num])]
        # print(len(futures))
        # concurrent.futures.wait(futures)
        logging.info('End processing')
    print("time to run",time()-t)
    return(None)


if __name__ == "__main__":
    process(sys.argv)
