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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler("debug.log"), logging.StreamHandler(sys.stdout)],
)


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
    run_back_sub(directory, directory_name)


if __name__ == "__main__":
    process(sys.argv)
