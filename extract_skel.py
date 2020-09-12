import cv2 
import imageio
import matplotlib.pyplot as plt
import numpy as np
from skimage.filters import frangi
from skimage.morphology import thin
from skimage import data, filters
from util import get_path


def extract_skeleton(date,plate,row,column):
    im = imageio.imread(get_path(date,plate,False,row,column))
    im_cropped=im
    im_blurred=cv2.blur(im_cropped, (200, 200))
    im_back_rem= im_cropped/(im_blurred+1)
    im_back_rem=cv2.normalize(im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    im_back_rem_inv = cv2.normalize(255-im_back_rem, None, 0, 255, cv2.NORM_MINMAX)
    frangised=frangi(im_back_rem,sigmas=range(1,20,4))
    frangised = cv2.normalize(frangised, None, 0, 255, cv2.NORM_MINMAX)
    transformed=frangised
    low = 70
    high = 150
    lowt = (transformed > low).astype(int)
    hight = (transformed > high).astype(int)
    hyst = filters.apply_hysteresis_threshold(transformed, low, high)
    kernel = np.ones((3,3),np.uint8)
    dilation = cv2.dilate(hyst.astype(np.uint8) * 255,kernel,iterations = 1)
    dilated = dilation>0
    skeletonized = thin(dilated)
    np.save(f'Data/imbackrem_{date}_{plate}_{row}_{column}',im_back_rem)
    np.save(f'Data/frangised_{date}_{plate}_{row}_{column}',frangised)
    np.save(f'Data/dilated_{date}_{plate}_{row}_{column}',dilated)
    np.save(f'Data/skeletonized_{date}_{plate}_{row}_{column}',skeletonized)
    return(skeletonized)