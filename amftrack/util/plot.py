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


def show_image_with_segment(image_path: str, x1, y1, x2, y2):
    """Show the image with a segment drawn on top of it"""
    show_image(image_path)
    plt.plot(x1, y1, marker="x", color="white")
    plt.plot(x2, y2, marker="x", color="white")
    plt.plot([x1, x2], [y1, y2], color="white", linewidth=2)
