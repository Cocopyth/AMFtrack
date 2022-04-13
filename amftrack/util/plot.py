from typing import List
import numpy as np
import os
import re
from PIL import Image
from skimage.measure import profile_line
import logging
from matplotlib import image
import matplotlib.pyplot as plt


def show_image(image_path: str) -> None:
    """Plot an image, from image path"""
    # TODO(FK): distinguish between extensions
    im = image.imread(image_path)
    plt.imshow(im)


def find_transformation(old_coord_list: List, new_coord_list: List):
    """
    Compute the rotation and translation to transform the old plane into the new one.
    old_coord_list and new_coord_list must contain at least 2 points.
    Ex:
    find_transformation([[16420,26260],[17120, 28480]], [[15760, 26500],[16420, 28780]])
    :return: Two arrays: a rotation and a translation
    """

    H = np.dot(
        np.transpose(np.array(old_coord_list) - np.mean(old_coord_list, axis=0)),
        np.array(new_coord_list) - np.mean(new_coord_list, axis=0),
    )

    U, S, V = np.linalg.svd(H)
    R = np.dot(V, np.transpose(U))
    t = np.mean(new_coord_list, axis=0) - np.dot(R, np.mean(old_coord_list, axis=0))

    return R, t


def get_transformation(R, t):
    """Returns the function to apply to the points to apply the transformation"""
    return lambda x: np.dot(R, x) + t


def extract_kymograth(path, x1, y1, x2, y2, pattern=None) -> np.array:
    """
    Takes as input the path to a folder of consecutive images (a movie) of size (m, n)
    and coordinates for a segment on the image.
    Returns the evolution of this segment over time in a (p, k) array where p is the
    number of images and k is the lenght of the segment.

    'pattern' is a parameter to only consider images matching a certain pattern.
    The output is a np.array of dimension (n, m) where n is the number of images
    and m is the length of the segment.
    WARNING: (x, y) coordinates are in the same system as matplotlib
    """

    point1 = np.array([y1, x1])  # NB: we switch to (y, x) instead of (x, y)
    point2 = np.array([y2, x2])  # This is because of the profile_line function

    if pattern is None:

        def is_valid(name):
            return True

    else:
        model = re.compile(pattern)

        def is_valid(name):
            match = model.search(name)
            if match:
                return True
            else:
                return False

    listdir = [file_name for file_name in os.listdir(path) if is_valid(file_name)]
    l = []
    for file_name in listdir:
        im = Image.open(os.path.join(path, file_name))
        im_np = np.array(im)
        profile = profile_line(im_np, point1, point2, mode="constant")
        profile = profile.reshape((1, len(profile)))
        l.append(profile)
    logging.info(f"Number of images handled: {len(l)}")

    return np.concatenate(l, axis=0)


def variance_over_time_on_segment(path, x1, y1, x2, y2, pattern=None):
    """
    Compute the variance over several images of each pixel on a segment.
    Return an array of shape (p, ) where p is the number of images in the folder `path`.
    """
    return np.std(extract_kymograth(path, x1, y1, x2, y2, pattern=pattern), axis=0)


if __name__ == "__main__":
    R, t = find_transformation(
        [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
    )
    f = get_transformation(R, t)
    print(f([16420, 26260]))
