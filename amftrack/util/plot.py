from typing import List
import numpy as np
import random

from matplotlib import image
import matplotlib.pyplot as plt
from amftrack.util.aliases import coord_int

import math
import statsmodels.api as sm


def make_stat(x0, ax):
    def statistic(x, y):
        X = sm.add_constant(x, prepend=False)
        model = sm.OLS(y, X)
        res = model.fit()
        a, b = res.params[0], res.params[1]
        ax.plot(x0, np.array(x0) * a + b, color="grey", alpha=0.01)
        return a

    return statistic


def gridplot(
    n: int,
    ncols=None,
    subw: float = 4.0,
    subh: float = 4.0,
    **kwargs,
):
    """This function creates a grid of subplots for displaying multiple plots in a single figure. The function takes in the number of plots to be displayed and optionally, the number of columns to use in the grid, and the width and height of each subplot. The function returns the figure object and an iterator over the subplots.

    Parameters:

        n (int): The number of plots to be displayed.
        ncols (int): The number of columns to use in the grid. Defaults to None, which sets the number of columns to be equal to n.
        subw (float): The width of each subplot in inches. Defaults to 4.0.
        subh (float): The height of each subplot in inches. Defaults to 4.0.
        **kwargs: Additional keyword arguments that are passed to the plt.subplots function.

    Returns:

        fig (matplotlib.figure.Figure): The figure object that contains the subplots.
        axs (iterator): An iterator over the subplots that can be used to plot data."""
    if ncols is None:
        ncols = n
    nrows = math.ceil(n / ncols)
    figsize = (subw * ncols, subh * nrows)
    fig, axs = plt.subplots(nrows, ncols, figsize=figsize, **kwargs)
    return fig, iter(axs.flatten())


def gridplot_fig(
    n: int,
    ncols=None,
    subw: float = 4.0,
    subh: float = 4.0,
    wspace: float = 1.0,
    **kwargs,
):
    if ncols is None:
        ncols = n
    nrows = math.ceil(n / ncols)
    figsize = (subw * ncols, subh * nrows)
    fig = plt.figure(figsize=figsize, **kwargs)
    subfigs = fig.subfigures(nrows, ncols, wspace=wspace, hspace=wspace)
    axs = [subfig.subplots() for subfig in iter(subfigs.flatten())]
    return fig, axs, iter(subfigs.flatten())


def make_random_color(seed):
    """
    From an int value, hands out a color with its transparency channel as np.array.
    """
    random.seed(seed)
    color = [random.randrange(255) for i in range(3)]
    color.append(255)  # alpha channel
    color = np.array(color)
    return color


def show_image(image_path: str) -> None:
    """Plot an image, from image path"""
    # TODO(FK): distinguish between extensions
    im = image.imread(image_path)
    plt.imshow(im)


def show_image_with_segment(image_path: str, x1, y1, x2, y2):
    """
    Show the image with a segment drawn on top of it
    WARNING: Coordinates are given in PLT referential.
    """
    show_image(image_path)
    plt.plot(y1, x1, marker="x", color="white")
    plt.plot(y2, x2, marker="x", color="white")
    plt.plot([y1, y2], [x1, x2], color="white", linewidth=2)


def pixel_list_to_matrix(pixels: List[coord_int], margin=0) -> np.array:
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
        M[pixel[0] - x_min + margin][pixel[1] - y_min + margin] = 1
    return M


def crop_image(matrix: np.array, region: List[coord_int]):
    """
    Crops the image to the given region.
    This is a robust function to avoid error when regions are out of bound.
    Coordinates of region are in the form [x, y]
    The lower bound are included, the upper bound is not.
    :param region: [[1, 2], [10, 10]] for example
    """

    dim_x = matrix.shape[0]
    dim_y = matrix.shape[1]
    x_min = np.min([np.max([0, np.min([region[0][0], region[1][0]])]), dim_x])
    x_max = np.max([np.min([dim_x, np.max([region[0][0], region[1][0]])]), 0])
    y_min = np.min([np.max([0, np.min([region[0][1], region[1][1]])]), dim_y])
    y_max = np.max([np.min([dim_y, np.max([region[0][1], region[1][1]])]), 0])

    return matrix[x_min:x_max, y_min:y_max]


if __name__ == "__main__":
    make_random_color(12)
