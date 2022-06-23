from math import sqrt
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread, imshow
from skimage.color import rgb2gray
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.morphology import (erosion, dilation, closing, opening,
                                area_closing, area_opening)

def detect_blobs(im,thresh=230,max_size=50):
    im2 = 255 - im
    im2_bw = im2 >= 230
    img_morph = area_closing(area_opening(im2_bw, 200), 200)
