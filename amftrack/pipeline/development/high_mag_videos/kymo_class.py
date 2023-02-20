import os
import imageio.v2 as imageio
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm
from amftrack.pipeline.functions.image_processing.extract_graph import (
    from_sparse_to_graph,
    generate_nx_graph,
    clean_degree_4,
)
import scipy
from amftrack.pipeline.functions.image_processing.node_id import remove_spurs
from amftrack.pipeline.functions.image_processing.extract_skel import remove_component, remove_holes
import numpy as np
from amftrack.pipeline.development.high_mag_videos.high_mag_videos_fun import *
from scipy import signal
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)
from scipy.interpolate import griddata

from skimage.morphology import skeletonize
from amftrack.util.sys import temp_path
import pandas as pd
from PIL import Image
from scipy.optimize import curve_fit
from amftrack.pipeline.functions.image_processing.extract_skel import (
    extract_skel_new_prince,
    run_back_sub,
    bowler_hat,
)


class Kymo_analysis(object):
    def __init__(self, imgs_address, format='tiff', logging=False, fps=20, binning=2, magnification=50, im_range=(0, -1), thresh=5e-07):
        self.imgs_address = imgs_address
        self.fps = fps
        self.range=im_range
        self.time_pixel_size = 1/self.fps
        self.binning = binning
        self.magnification = magnification
        self.video_name = imgs_address.split("/")[-1]
        self.kymos_path = "/".join(
            imgs_address.split("/")[:-1] + ["_".join((self.video_name, "kymos"))]
        )
        if not os.path.exists(self.kymos_path):
            os.mkdir(self.kymos_path)
        self.files = os.listdir(self.imgs_address)
        self.images_total_path = [os.path.join(self.imgs_address, file) for file in self.files]
        self.images_total_path.sort()
        self.selection_file = self.images_total_path
        self.selection_file.sort()
        self.selection_file = self.selection_file[self.range[0]:self.range[1]]
        self.segmented, self.nx_graph_pruned, self.pos = segment_brightfield(imageio.imread(self.selection_file[self.range[0]]), thresh=thresh)

    def plot_start_end(self):
        image = imageio.imread(self.selection_file[self.range[0]])
        image2 = imageio.imread(self.selection_file[self.range[1]])
        fig, ax = plt.subplots()
        ax.imshow(image, cmap="gray")
        ax.imshow(image2, alpha=0.5)
        plt.show()
        return None

