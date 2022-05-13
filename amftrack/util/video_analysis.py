import numpy as np
from skimage.measure import profile_line
import logging
import os
import re
from PIL import Image


def extract_kymograph(path, x1, y1, x2, y2, validation_fun=None) -> np.array:
    """
    Takes as input the path to a folder of consecutive images (a movie) of size (m, n)
    and coordinates for a segment on the image.

    :param validation_fun: is a function returning True or False on file names to only select some images
    :returns: the evolution of this segment over time in a (p, k) array where p is the
    number of images and k is the length of the segment.
    :NB: [x, y] coordinates are in the same referential as the GENERAL referential (x vertical axis)
    :NB: Example of validation_fun

    model = re.compile(pattern)
    def is_valid(name):
        match = model.search(name)
        if match:
            return True
        return False
    """

    if validation_fun is None:

        def validation_fun(filename):
            return True

    listdir = [file_name for file_name in os.listdir(path) if validation_fun(file_name)]
    l = []
    for file_name in listdir:
        im = Image.open(os.path.join(path, file_name))
        im_np = np.array(im)
        profile = profile_line(im_np, [x1, y1], [x2, y2], mode="constant")
        profile = profile.reshape((1, len(profile)))
        l.append(profile)
    logging.info(f"Number of images handled: {len(l)}")

    return np.concatenate(l, axis=0)


def variance_over_time_on_segment(path, x1, y1, x2, y2, validation_fun=None):
    """
    Compute the variance over several images of each pixel on a segment.
    Return an array of shape (p, ) where p is the number of images in the folder `path`.
    """
    return np.std(
        extract_kymograph(path, x1, y1, x2, y2, validation_fun=validation_fun), axis=0
    )
