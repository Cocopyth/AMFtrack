from typing import List
import numpy as np
import os
import re
from PIL import Image
from skimage.measure import profile_line
import logging
from matplotlib import image
import matplotlib.pyplot as plt
from amftrack.util.aliases import coord_int


def show_image(image_path: str) -> None:
    """Plot an image, from image path"""
    # TODO(FK): distinguish between extensions
    im = image.imread(image_path)
    plt.imshow(im)


def show_image_with_segment(image_path: str, x1, y1, x2, y2):
    """Show the image with a segment drawn on top of it"""
    show_image(image_path)
    plt.plot(x1, y1, marker="x", color="white")
    plt.plot(x2, y2, marker="x", color="white")
    plt.plot([x1, x2], [y1, y2], color="white", linewidth=2)


def pixel_list_to_matrix(pixels: List[coord_int], t: int, margin=0) -> np.array:
    """
    Returns a binary image of the Edge
    :param margin: white border added around edge pixels
    """
    x_max = np.max([pixel[0] for pixel in pixels])
    x_min = np.min([pixel[0] for pixel in pixels])
    y_max = np.max([pixel[1] for pixel in pixels])
    y_min = np.min([pixel[1] for pixel in pixels])
    M = np.zeros((x_max - x_min + 1 + 2 * margin, (y_max - y_min + 1 + 2 * margin)))
    for pixel in pixels:
        M[pixel[0] + margin][pixel[1] + margin] = 1
    return M
