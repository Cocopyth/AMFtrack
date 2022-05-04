from random import choice
import numpy as np
from typing import List, Tuple
import cv2 as cv
import matplotlib.pyplot as plt

from amftrack.util.aliases import coord_int, coord
from amftrack.util.param import DIM_X, DIM_Y
from amftrack.util.image_analysis import is_in_image
from amftrack.util.geometry import (
    generate_index_along_sequence,
    distance_point_pixel_line,
    get_closest_line_opt,
)
from amftrack.util.plot import crop_image
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from amftrack.util.sparse import dilate_coord_list
from random import randrange



def get_random_edge(exp: Experiment, t=0) -> Edge:
    "Select randomly an edge of Experiment at timestep t"
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edge_coord = choice(list(G.edges))
    edge = Edge(Node(edge_coord[0], exp), Node(edge_coord[1], exp), exp)
    return edge


def get_all_edges(exp: Experiment, t: int) -> List[Edge]:
    """
    Return a list of all Edge objects at timestep t in the `experiment`
    """
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    return [
        Edge(Node(edge_coord[0], exp), Node(edge_coord[1], exp), exp)
        for edge_coord in list(G.edges)
    ]


def find_nearest_edge(point: coord, exp: Experiment, t: int) -> Edge:
    """
    Find the nearest edge to `point` in `exp` at timestep `t`.
    The coordonates are given in the GENERAL ref.
    :return: Edge object
    """
    edges = get_all_edges(exp, t)
    l = [edge.pixel_list(t) for edge in edges]
    return edges[get_closest_line_opt(point, l)[0]]


def distance_point_edge(point: coord, edge: Edge, t: int, step=1):
    """
    Compute the minimum distance between the `point` and the `edge` at timestep t.
    The `step` parameter determine how frequently we compute the distance along the edge.

    There can be several use cases:
    - step == 1: we compute the distance for every point on the edge.
    It is computationally expensive but the result will have no error.
    - step == n: we compute the distance every n point on the edge.
    The higher the n, the less expensive the function is, but the minimal distance won't be exact.

    NB: With a high n, the error will be mostly important for edges that are closed to the point.
    A point on an edge, that should have a distance of 0 could have a distance of n/2 for example.
    """

    pixel_list = edge.pixel_list(t)
    return distance_point_pixel_line(point, pixel_list, step)


def aux_plot_edge(
    edge: Edge, t: int, mode=0, points=10, f=None
) -> Tuple[np.array, List[coord]]:
    """
    This intermediary function returns the data that will be plotted.
    See plot_edge for more information.
    :return: image, coordinate of all the points in image
    """
    # TODO(FK): use a mask instead of points
    exp = edge.experiment
    number_points = 10
    # Fetch coordinates of edge points in general referential
    if mode == 0:
        list_of_coord = [edge.begin.pos(t), edge.end.pos(t)]
    else:
        list_of_coord = edge.pixel_list(t)
        if mode == 2:
            pace = len(list_of_coord) // number_points
            l = [list_of_coord[i * pace] for i in range(number_points)]
            l.append(list_of_coord[-1])
            list_of_coord = l
        elif mode == 3:
            indexes = f(len(list_of_coord))
            list_of_coord = [list_of_coord[i] for i in indexes]

    # Image index: we take the first one we find for the first node
    x, y = list_of_coord[0]
    try:
        i = exp.find_im_indexes_from_general(x, y, t)[0]
    except:
        raise Exception("There is no image for this Edge")
    # Convert coordinates to image coordinates
    list_of_coord_im = [exp.general_to_image(coord, t, i) for coord in list_of_coord]
    # Fetch the image
    im = exp.get_image(t, i)
    return im, list_of_coord_im


def plot_edge(edge: Edge, t: int, mode=2, points=10, save_path=None, f=None):
    """
    Plot the Edge in its source images, if one exists.
    It plots the points of the pixel list (after the hypha has been thined).
    :WARNING: If the edge is accross two image, only a part of the edge will be plotted
    :param mode:
    - mode 0: only begin end end
    - mode 1: plot whole edge
    - mode 2: plot only a number of points
    - mode 3: specify a function that chooses point indexes based on the length
    :param points: number of points to plot
    :param save_path: doesn't save if None, else save the image as `save_path`
    :param f: function of the signature f(n:int)->List[int]
    """
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points, f)
    plt.imshow(im)
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][0], list_of_coord_im[i][1], marker="x", color="red"
        )
    if save_path is not None:
        plt.savefig(save_path)


def plot_edge_cropped(
    edge: Edge, t: int, mode=2, points=10, margin=50, f=None, save_path=None
):
    """
    Same as plot_edge but the image is cropped with a margin around points of interest.
    """
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points, f)
    x_min = np.min([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    x_max = np.max([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    y_min = np.min([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])
    y_max = np.max([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])

    x_min = np.max([0, x_min - margin])
    x_max = np.min([im.shape[1], x_max + margin])  # NB: Careful: inversion
    y_min = np.max([0, y_min - margin])
    y_max = np.min([im.shape[0], y_max + margin])

    list_of_coord_im = [[p[0] - x_min, p[1] - y_min] for p in list_of_coord_im]
    im = im[int(y_min) : int(y_max), int(x_min) : int(x_max)]

    fig = plt.figure()
    plt.imshow(im)  # TODO(FK): shouldn't plot in save mode
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][0], list_of_coord_im[i][1], marker="x", color="red"
        )
    if save_path is not None:
        plt.savefig(save_path)


def plot_edge_skeleton(edge: Edge, t: int, downsizing=5, dilation=50):
    """
    PROV
    This function plots an edge on the original image.
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: thickness of the hyphas
    """
    # TODO(FK): plot better
    exp = edge.experiment
    f = lambda c: list(
        (np.array(exp.general_to_timestep(c, t)) / downsizing).astype(int)
    )
    # Construct the edge
    # kernel = np.ones((dilation, dilation), np.uint8)
    # skel = pixel_list_to_matrix(edge.pixel_list(t), margin=dilation)
    # dilated_skel = cv.dilate(skel.astype(np.uint8), kernel, iterations=1)
    # ------------------
    skel = [f(coordinates) for coordinates in edge.pixel_list(t)]
    # TODO(FK): add dilation warning with np coordinates
    size = int(dilation / downsizing)
    dilated_skel = dilate_coord_list(skel, iteration=size)
    # Chose a color
    color = np.array(
        [randrange(255), randrange(255), randrange(255)]
    )  # TODO (FK): make it prettier
    # ------------------
    # Generate the full image and convert it to a
    im = reconstruct_image(exp, t, downsizing=downsizing)  # greyscale image downsided
    im_ = np.reshape(im, (im.shape[0], im.shape[1], 1))
    im_rgb = np.concatenate((im_, im_, im_), axis=2)  # QUICKFIX
    # Add the edge on it

    # Plot
    for c in dilated_skel:
        # TODO check if out of bound here
        x, y = c[0], c[1]
        im_rgb[x][y][:] = color

    plt.imshow(im_rgb)




def reconstruct_image(exp: Experiment, t: int, downsizing=1) -> np.array:
    """
    This function reconstructs the full size image at timestep t and return it as an np array.
    :param downsizing: factor by which the image is downsized, 1 returns the original image
    WARNING: the image is a very heavy object (2 Go)
    NB: To plot objects in this image, the TIMESTEP referential must be used
    """
    if exp.image_coordinates is None:
        exp.load_tile_information(t)

    image_coodinates = exp.image_coordinates[t]

    # Find the maximum dimension
    m_x = np.max([c[0] for c in image_coodinates])
    m_y = np.max([c[1] for c in image_coodinates])

    # Dimensions
    d_x = int(m_x + DIM_X)
    d_y = int(m_y + DIM_Y)

    # Create the general image frame
    full_im = np.ones((d_y, d_x), dtype=np.uint8)

    # Copy each image into the frame
    for i, im_coord in enumerate(image_coodinates):
        im = exp.get_image(t, i)
        im_coord = [
            int(im_coord[0]),
            int(im_coord[1]),
        ]  # original im coordinates are float
        full_im[
            im_coord[1] : im_coord[1] + DIM_Y, im_coord[0] : im_coord[0] + DIM_X
        ] = im

    full_im = cv.resize(
        full_im, (full_im.shape[0] // downsizing, full_im.shape[1] // downsizing)
    )

    return full_im


def plot_full_image(
    exp: Experiment, t: int, downsizing=10, save="", region: List[coord_int] = None
) -> np.array:
    """
    This function plots the full size image at timestep t.
    :param downsizing: factor by which the image is downsized
    :param save: path (including the file name) where we want to store the image
    :param region: two points that delimit a square to extract from the image.
    NB: the two points for region are given in the TIMESTEP referential
    WARNING: the image is a very heavy object (2 Go), without downsizing it can crash
    """
    full_im = reconstruct_image(exp, t, downsizing=downsizing)
    dim_x = full_im.shape[1]  # careful with the order
    dim_y = full_im.shape[0]

    if region != None:
        for i in range(2):
            for j in range(2):
                region[i][j] = region[i][j] // downsizing
        full_im = crop_image(full_im, region)

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(full_im, cmap="gray", interpolation="none")
    if save:
        plt.savefig(save)
        plt.close(fig)
    else:
        plt.show()

