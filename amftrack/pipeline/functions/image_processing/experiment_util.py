from random import choice
import random
import numpy as np
from typing import List, Tuple, Optional, Callable
import cv2 as cv
from random import randrange
from scipy import ndimage
import logging
import os
from amftrack.util.aliases import coord_int, coord
from amftrack.util.geometry import (
    distance_point_pixel_line,
    get_closest_line_opt,
    get_closest_lines,
    format_region,
    intersect_rectangle,
    get_overlap,
    get_bounding_box,
    expand_bounding_box,
    is_in_bounding_box,
)
from amftrack.util.plot import crop_image, make_random_color
from amftrack.util.image_analysis import extract_inscribed_rotated_image
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from amftrack.pipeline.functions.image_processing.extract_skel import bowler_hat
from pymatreader import read_mat
from matplotlib import cm
import cv2
from shapely.geometry import Polygon, shape, Point
from shapely.affinity import affine_transform, rotate
from amftrack.util.sparse import dilate_coord_list
from amftrack.util.other import is_in
import geopandas as gpd
import matplotlib as mpl


def get_random_edge(exp: Experiment, t=0) -> Edge:
    "Randomly select an edge of Experiment at timestep t"
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edge_nodes = choice(list(G.edges))
    edge = Edge(Node(edge_nodes[0], exp), Node(edge_nodes[1], exp), exp)
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


def get_dimX_dimY(exp):
    try:
        return exp.dimX_dimY
    except:
        im = exp.get_image(0, 0)
        exp.dimX_dimY = im.shape
        return exp.dimX_dimY


def get_all_nodes(exp, t) -> List[Node]:
    """
    Return a list of all Node objects at timestep t in the `experiment`
    """
    return [Node(i, exp) for i in exp.nx_graph[t].nodes]


def find_nearest_edge(point: coord, exp: Experiment, t: int, edge_list=None) -> Edge:
    """
    Find the nearest edge to `point` in `exp` at timestep `t`.
    The coordonates are given in the GENERAL ref.
    If edge list is given, will only look in that list
    :return: Edge object
    """
    edges = get_all_edges(exp, t) if edge_list is None else edge_list
    l = [edge.pixel_list(t) for edge in edges]
    return edges[get_closest_line_opt(point, l)[0]]


def find_neighboring_edges(
    point: coord, exp: Experiment, t: int, n_nearest=5, step=50
) -> List[Edge]:
    """
    Find the nearest edges to `point` in `exp` at timestep `t`.
    The coordonates are given in the GENERAL ref.
    :return: List of Edge object
    :param n_nearest: number of neihgboring edges to select
    :param step: step along edge to compute the distance. step = 1 yield exact result but expensive
    """
    edges = get_all_edges(exp, t)
    l = [edge.pixel_list(t) for edge in edges]
    indexes = get_closest_lines(point, l, step=50, n_nearest=10)[0]
    kept_edges = [edges[i] for i in indexes]
    return kept_edges


def get_edge_from_node_labels(
    exp: Experiment, t: int, start_node: int, end_node: int
) -> Optional[Edge]:
    """
    From int label of starting node and ending node, returns the edge object if it exists.
    Else return None.
    """
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edge_nodes = list(G.edges)
    if (start_node, end_node) in edge_nodes:
        edge = Edge(Node(start_node, exp), Node(end_node, exp), exp)
        return edge
    if (end_node, start_node) in edge_nodes:
        edge = Edge(Node(end_node, exp), Node(start_node, exp), exp)
        return edge
    else:
        return None


def distance_point_edge(point: coord, edge: Edge, t: int, step=1):
    """
    Compute the minimum distance between the `point` and the `edge` at timestep t.
    The `step` parameter determines how many points along the edge are used to find the distance.

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
    :return: np binary image, coordinate of all the points in image
    """
    # TODO(FK): use reconstruct image instead
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
    - mode 0: only beginning and end of the edge
    - mode 1: plot every edge point
    - mode 2: plot only a number of points equal to `points`
    - mode 3: specify a function `f` that chooses point indexes based on the length
    :param points: number of points to plot
    :param save_path: doesn't save if None, else save the image as `save_path`
    :param f: function of the signature f(n:int)->List[int]
    """
    # TODO(FK): use reconstruct image instead
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points, f)
    plt.imshow(im)
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][1], list_of_coord_im[i][0], marker="x", color="red"
        )
    if save_path is not None:
        plt.savefig(save_path)


def plot_edge_cropped(
    edge: Edge, t: int, mode=2, points=10, margin=50, f=None, save_path=None
):
    """
    Same as plot_edge but the image is cropped with a margin around points of interest.
    """
    # TODO(FK): merge with plot_edge
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points, f)
    x_min = np.min([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    x_max = np.max([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    y_min = np.min([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])
    y_max = np.max([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])

    x_min = np.max([0, x_min - margin])
    x_max = np.min([im.shape[0], x_max + margin])
    y_min = np.max([0, y_min - margin])
    y_max = np.min([im.shape[1], y_max + margin])

    list_of_coord_im = [[p[0] - x_min, p[1] - y_min] for p in list_of_coord_im]
    im = im[int(x_min) : int(x_max), int(y_min) : int(y_max)]

    fig = plt.figure()
    plt.imshow(im)  # TODO(FK): shouldn't plot in save mode
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][1], list_of_coord_im[i][0], marker="x", color="red"
        )
    if save_path is not None:
        plt.savefig(save_path)
        plt.close()


def plot_full_image_with_features(
    exp: Experiment,
    t: int,
    region=None,
    edges: List[Edge] = [],
    points: List[coord_int] = [],
    segments: List[List[coord_int]] = [],
    nodes: List[Node] = [],
    downsizing=5,
    dilation=1,
    save_path="",
    prettify=False,
) -> None:
    """
    This is the general purpose function to plot the full image or a region `region` of the image at timestep `t`,
    downsized by a chosen factor `downsized` with additionnal features such as: edges, nodes, points, segments.
    The coordinates for all the objects are provided in the TIMESTEP referential.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param edges: list of edges to plot, it is the pixel list that is plotted, not a straight line
    :param nodes: list of nodes to plot
    :param points: points such as [123, 234] to plot with a red cross on the image
    :param segments: plot lines between two points that are provided
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param prettify: if True, the image will be enhanced by smoothing the intersections between images

    NB: the typical full region of a full image is [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    NB: the colors are chosen randomly for edges
    NB: giving a smaller region greatly increase computation time
    """

    # TODO(FK): fetch image size from experiment object here, and use it in reconstruct image
    # TODO(FK): add is_in_bounding_box to discard points out of interest zone
    # NB: possible other parameters that could be added: alpha between layers, colors for object, figure_size
    DIM_X, DIM_Y = get_dimX_dimY(exp)
    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X
        region[1][1] += DIM_Y

    # 1/ Image layer
    im, f = reconstruct_image(
        exp,
        t,
        downsizing=downsizing,
        region=region,
        prettify=prettify,
        white_background=False,  # TODO(FK): add image dimention here dimx = ..
    )
    f_int = lambda c: f(c).astype(int)
    new_region = [
        f_int(region[0]),
        f_int(region[1]),
    ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 2/ Edges layer
    skel_im, _ = reconstruct_skeletton_from_edges(
        exp,
        t,
        edges=edges,
        region=region,
        color_seeds=None,
        downsizing=downsizing,
        dilation=dilation,
    )

    # 3/ Fusing layers
    fig = plt.figure(
        figsize=(12, 8)
    )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap="gray", interpolation="none")
    ax.imshow(skel_im, alpha=0.5, interpolation="none")

    # 3/ Plotting the Nodes
    size = 5
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nodes:
        general_pos = list(node.pos(t))
        ts_pos = exp.general_to_timestep(general_pos, t)
        c = f(ts_pos)
        if is_in_bounding_box(c, new_region):
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                size=size,
                bbox=bbox_props,
            )
    # 4/ Plotting coordinates
    points = [f(c) for c in points]
    for c in points:
        if is_in_bounding_box(c, new_region):
            plt.plot(c[1], c[0], marker="x", color="red")
    # 5/ Plotting segments
    segments = [[f(segment[0]), f(segment[1])] for segment in segments]
    for s in segments:
        plt.plot(
            [s[0][1], s[1][1]],  # x1, x2
            [s[0][0], s[1][0]],  # y1, y2
            color="white",
            linewidth=2,
        )

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
    return ax


font_path = os.path.join(mpl.get_data_path(), "fonts/ttf/lucidasansdemibold.ttf")
if os.path.exists(font_path):
    fpath = Path(mpl.get_data_path(), "fonts/ttf/lucidasansdemibold.ttf")
else:
    fpath = None


def plot_full(
    exp: Experiment,
    t: int,
    region=None,
    edges: List[Edge] = [],
    points: List[coord_int] = [],
    segments: List[List[coord_int]] = [],
    nodes: List[Node] = [],
    downsizing=5,
    dilation=1,
    save_path="",
    prettify=False,
    with_point_label=False,
    figsize=(12, 8),
    dpi=None,
    node_size=5,
) -> None:
    """
    This is the general purpose function to plot the full image or a region `region` of the image at
    any given timestep t. The region being specified in the GENERAL coordinates.
    The image can be downsized by a chosen factor `downsized` with additionnal features such as: edges, nodes, points, segments.
    The coordinates for all the objects are provided in the GENERAL referential.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param edges: list of edges to plot, it is the pixel list that is plotted, not a straight line
    :param nodes: list of nodes to plot (only nodes in the `region` will be shown)
    :param points: points such as [123, 234] to plot with a red cross on the image
    :param segments: plot lines between two points that are provided
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param prettify: if True, the image will be enhanced by smoothing the intersections between images
    :param with_point_label: if True, the index of the point is ploted on top of it

    NB: the full region of a full image is typically [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    NB: the colors are chosen randomly for edges
    NB: giving a smaller region greatly increase computation time
    """

    # TODO(FK): fetch image size from experiment object here, and use it in reconstruct image
    # TODO(FK): colors for edges are not consistent
    # NB: possible other parameters that could be added: alpha between layers, colors for object, figure_size
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X  # TODO(FK): Shouldn't be hardcoded
        region[1][1] += DIM_Y

    # 1/ Image layer
    im, f = reconstruct_image_from_general(
        exp,
        t,
        downsizing=downsizing,
        region=region,
        prettify=prettify,
        white_background=False,  # TODO(FK): add image dimention here dimx = ..
    )
    f_int = lambda c: f(c).astype(int)
    new_region = [
        f_int(region[0]),
        f_int(region[1]),
    ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 2/ Edges layer
    skel_im, _ = reconstruct_skeletton(
        [edge.pixel_list(t) for edge in edges],
        region=region,
        color_seeds=[(edge.begin.label + edge.end.label) % 255 for edge in edges],
        downsizing=downsizing,
        dilation=dilation,
    )

    # 3/ Fusing layers
    fig = plt.figure(
        figsize=figsize
    )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap="gray", interpolation="none")
    ax.imshow(skel_im, alpha=0.5, interpolation="none")

    # 3/ Plotting the Nodes
    size = node_size
    for node in nodes:
        c = f(list(node.pos(t)))
        color = make_random_color(node.label)[:3]
        reciprocal_color = 255 - color
        color = tuple(color / 255)
        reciprocal_color = tuple(reciprocal_color / 255)
        bbox_props = dict(boxstyle="circle", fc=color, edgecolor="none")
        if is_in_bounding_box(c, new_region):
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                bbox=bbox_props,
                font=fpath,
                fontdict={"color": reciprocal_color},
                size=size,
                # alpha = 0.5
            )
    # 4/ Plotting coordinates
    points = [f(c) for c in points]
    for i, c in enumerate(points):
        if is_in_bounding_box(c, new_region):
            plt.plot(c[1], c[0], marker="x", color="red", markersize=10, alpha=0.5)
            if with_point_label:
                plt.text(c[1], c[0], f"{i}")

    # 5/ Plotting segments
    segments = [[f(segment[0]), f(segment[1])] for segment in segments]
    for s in segments:
        plt.plot(
            [s[0][1], s[1][1]],  # x1, x2
            [s[0][0], s[1][0]],  # y1, y2
            color="white",
            linewidth=2,
        )

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return ax


def reconstruct_image_simple(
    exp: Experiment, t: int, downsizing=1, white_background=True
) -> np.array:
    """
    DEPRECATED
    This function is the older version of `reconstruct_image`. It has much less features but it
    is kept because it is much simpler than the other one. For general purpose, use the other
    one which is much more optimize and gives much more flexibility.

    This function reconstructs the full size image at timestep t and return it as an np array.

    :param downsizing: factor by which the image is downsized, 1 returns the original image
    :param white_background: if True, areas where no image was found are white otherwise black

    WARNING: the image is a very heavy object (2 Go)
    """
    if exp.image_coordinates is None:
        exp.load_tile_information(t)

    image_coodinates = exp.image_coordinates[t]
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    # Find the maximum dimension
    m_x = np.max([c[0] for c in image_coodinates])
    m_y = np.max([c[1] for c in image_coodinates])

    # Dimensions
    d_x = int(m_x + DIM_X)
    d_y = int(m_y + DIM_Y)

    # Create the general image frame
    if white_background:
        background_value = 255
    else:
        background_value = 0
    full_im = np.full((d_x, d_y), fill_value=background_value, dtype=np.uint8)

    # Copy each image into the frame
    for i, im_coord in enumerate(image_coodinates):
        im = exp.get_image(t, i)
        im_coord = [
            int(im_coord[0]),
            int(im_coord[1]),
        ]  # original im coordinates are float
        full_im[
            im_coord[0] : im_coord[0] + DIM_X, im_coord[1] : im_coord[1] + DIM_Y
        ] = im

    # WARNING: cv2 inverses the shape compared to numpy
    full_im = cv.resize(
        full_im, (full_im.shape[1] // downsizing, full_im.shape[0] // downsizing)
    )

    return full_im


def reconstruct_image(
    exp: Experiment,
    t: int,
    region=None,
    downsizing=5,
    prettify=False,
    white_background=True,
) -> Tuple[List[np.array], Callable[[float, float], float]]:
    """
    This function reconstructs the full size image or a part of it given by `region` at
    timestep `t` and return it as an np array. It also returns a function mapping coordinates
    in the TIMESTEP referential to coordinates in the reconstructed image referential.

    :param region: [[a, b], [c, d]] defining a zone in the TIMESTEP ref that
                    we want to extract. Can be np.array or lists, int or floats
    :param downsizing: factor by which the image is downsized, 1 returns the original image
    :param prettify: add transformation operation to make the rendering better (but costly)
    :param white_background: if True, areas where no images were found are white, otherwise black
    :param dimx: x dimension of images

    WARNING: without downsizing, the full image is heavy (2 Go)
    NB: returned image shape is ((int(a)-int(c))//downsizing, (int(b)-int(d))//downsizing)
    NB: the typical full region of a full image is [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    NB: output image format is np.uint8
    """
    # Load information from the stiching
    if exp.image_coordinates is None:
        exp.load_tile_information(
            t
        )  # TODO (FK): make this independent of plate analysis
    image_coodinates = exp.image_coordinates[t]

    # Define canvas dimension
    dim_x, dim_y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        region = get_bounding_box(image_coodinates)
        region[1][0] += dim_x
        region[1][1] += dim_y

    region = format_region(region)
    region = [[int(e) for e in l] for l in region]  # only to avoid errors
    d_x = (region[1][0] - region[0][0]) // downsizing
    d_y = (region[1][1] - region[0][1]) // downsizing

    # Mapping from TIMESTEP referential to downsized image referential
    f = lambda c: (np.array(c) - np.array(region[0])) / downsizing
    f_int = lambda c: f(c).astype(int)
    region_new = [f_int(region[0]), f_int(region[1])]  # should be [[0, 0],[d_x, d_y]]

    # Create the general image frame
    if white_background:
        background_value = 255
    else:
        background_value = 0
    full_im = np.full((d_x, d_y), fill_value=background_value, dtype=np.uint8)

    # Copy each image into the frame
    for i, im_coord in enumerate(image_coodinates):
        if intersect_rectangle(
            region[0],
            region[1],
            im_coord,
            [im_coord[0] + dim_x, im_coord[1] + dim_y],
            strict=True,  # we don't care about the last pixel
        ):
            im = exp.get_image(t, i)
            if prettify:
                im = -bowler_hat(-im, 16, [45])
                im = ((1 + im) * 255).astype(np.uint32)
                im = im + 255 - np.percentile(im.flatten(), 20)
                im[np.where(im >= 255)] = 255
                # im = im + 255 - np.max(im.flatten())
                im = im.astype(np.uint8)
                # im = ((im.astype(np.float)/np.max(im))*255).astype(np.uint8)
            length_x = im.shape[0] // downsizing
            length_y = im.shape[1] // downsizing
            im_coord_new = f_int(im_coord)
            if downsizing != 1:
                im = cv.resize(
                    im, (length_y, length_x)
                )  # cv2 has different (x, y) convention

            try:
                overlap = get_overlap(
                    region_new[0],
                    region_new[1],
                    im_coord_new,
                    im_coord_new + np.array([length_x, length_y]),
                    strict=True,
                )
                full_im[
                    overlap[0][0] : overlap[1][0], overlap[0][1] : overlap[1][1]
                ] = im[
                    overlap[0][0] - im_coord_new[0] : overlap[1][0] - im_coord_new[0],
                    overlap[0][1] - im_coord_new[1] : overlap[1][1] - im_coord_new[1],
                ]
            except:
                logging.error("The resized rectangle is no longer overlapping")

    return full_im, f


def reconstruct_skeletton(
    coord_list_list: List[List[coord_int]],
    region=[[0, 0], [20000, 40000]],
    color_seeds: List[int] = None,
    color_list: List[Tuple[int, int, int, int]] = None,
    downsizing=5,
    dilation=2,
) -> Tuple[List[np.array], Callable[[float, float], float]]:
    """
    This function makes an image of `region` downsized by a factor `downsizing`
    where each list of points in `coord_list_list` is drawn with a different color
    specified in `color_seed`.
    It also applies a kernel dilation of size `dilation`.
    All areas without any points painted are set transparent (alpha value of 0).
    It also returns a function f to plot points in the image.

    :param color_seeds: list of ints of same length as coord_list_list, list of points with same int will have same color
    :param color_list: List of colors (to replace color_seeds if given)
    :param region: [[a, b], [c, d]] defining a zone that we want, it can be np.array or lists, int or floats
    :param downsizing: factor by which the image is downsized, 1 returns the original image

    WARNING: without downsizing, the full image is heavy (8 Go)
    WARNING: if region is not specify, it has to be recomputed which is expensive
    WARNING: default region isn't the same one as for reconstruct image

    NB: returned image shape is ((int(a)-int(c))//downsizing, (int(b)-int(d))//downsizing)
    NB: dilation is applied after downsizing the thickness of lines T is transformed in T*dilation + 2*dilation
    """
    # NB: This function is independent of exp object, it could be moved to utils
    if region == None:
        region = expand_bounding_box(
            get_bounding_box([c for sublist in coord_list_list for c in sublist]), 20
        )

    # Define canvas
    region = format_region(region)
    region = [[int(e) for e in l] for l in region]  # only to avoid errors
    d_x = (region[1][0] - region[0][0]) // downsizing
    d_y = (region[1][1] - region[0][1]) // downsizing
    full_im = np.full(shape=(d_x, d_y, 4), fill_value=0, dtype=np.uint8)

    # Mapping from original referential to downsized and cropped image referential
    f = lambda c: (np.array(c) - np.array(region[0])) / downsizing
    f_int = lambda c: f(c).astype(int)

    if not color_seeds:
        color_seeds = [random.randrange(255) for _ in range(len(coord_list_list))]
    # Plot edges
    for i, coord_list in enumerate(coord_list_list):
        if not color_list:
            color = make_random_color(color_seeds[i])
        else:
            color = color_list[i]
        skel = [
            f_int(coordinates) for coordinates in coord_list
        ]  # nb: coordinates are not unique after downsizing
        for c in skel:
            x, y = c[0], c[1]
            if 0 <= x < d_x and 0 <= y < d_y:
                full_im[x][y][:] = color

    # Dilation
    kernel = np.ones((dilation, dilation), np.uint8)
    full_im = cv.dilate(full_im, kernel, iterations=1)

    return full_im, f


def reconstruct_skeletton_unicolor(
    coord_list_list: List[List[coord_int]],
    region=[[0, 0], [20000, 40000]],
    downsizing=5,
    dilation=2,
    foreground=255,
) -> Tuple[List[np.array], Callable[[float, float], float]]:
    """
    This function makes an image of `region` downsized by a factor `downsizing`
    where each list of points in `coord_list_list` is drawn.
    It also applies a kernel dilation of size `dilation`.
    It also returns a function f to plot points in the image.

    :param region: [[a, b], [c, d]] defining a zone that we want, it can be np.array or lists, int or floats
    :param downsizing: factor by which the image is downsized, 1 returns the original image
    :background: value for the background
    :foreground: value for the foreground (the skeletton)

    WARNING: without downsizing, the full image is heavy (2 Go)
    WARNING: if region is not specify, it has to be recomputed which is expensive
    WARNING: default region isn't the same one as for reconstruct image

    NB: returned image shape is ((int(a)-int(c))//downsizing, (int(b)-int(d))//downsizing)
    NB: dilation is applied after downsizing the thickness of lines T is transformed in T*dilation + 2*dilation
    """
    # NB: This function is indepedent of exp object, could be move to util
    if region == None:
        region = expand_bounding_box(
            get_bounding_box([c for sublist in coord_list_list for c in sublist]), 20
        )

    # Define canvas
    region = format_region(region)
    region = [[int(e) for e in l] for l in region]  # only to avoid errors
    d_x = (region[1][0] - region[0][0]) // downsizing
    d_y = (region[1][1] - region[0][1]) // downsizing
    full_im = np.full(shape=(d_x, d_y), fill_value=0, dtype=np.uint8)

    # Mapping from original referential to downsized and cropped image referential
    f = lambda c: (np.array(c) - np.array(region[0])) / downsizing
    f_int = lambda c: f(c).astype(int)

    # Plot edges
    for i, coord_list in enumerate(coord_list_list):
        skel = [
            f_int(coordinates) for coordinates in coord_list
        ]  # nb: coordinates are not unique after downsizing
        for c in skel:
            x, y = c[0], c[1]
            if 0 <= x < d_x and 0 <= y < d_y:
                full_im[x][y] = foreground

    # Dilation
    kernel = np.ones((dilation, dilation), np.uint8)
    full_im = cv.dilate(full_im, kernel, iterations=1)

    return full_im, f


def reconstruct_skeletton_from_edges(
    exp,
    t,
    edges: List[Edge],
    region=[[0, 0], [20000, 40000]],  # add get bounding box
    color_seeds: List[int] = None,
    color_list: List[Tuple[int, int, int, int]] = None,
    downsizing=5,
    dilation=2,
    timestep=True,
) -> Tuple[List[np.array], Callable[[float, float], float]]:
    """
    This is a wrapper function around reconstruct_skeletton, to apply it
    directly to edge objects.
    See reconstruct_skeletton for documentation.
    Region is in the GENERAL referential.
    """
    # Conversion of pixels lists to TIMESTEP referential
    # QUICKFIX
    pixels = [edge.pixel_list(t) for edge in edges]
    pixels_ = []
    for pl in pixels:
        l = []
        for p in pl:
            if timestep:
                l.append(exp.general_to_timestep(p, t))
            else:
                l.append(p)
        pixels_.append(l)

    im, f = reconstruct_skeletton(
        pixels_,
        region=region,
        color_seeds=color_seeds,
        color_list=color_list,
        downsizing=downsizing,
        dilation=dilation,
    )
    return im, f


def get_timedelta_second(exp, t, tp1):
    seconds = (exp.dates[tp1] - exp.dates[t]).total_seconds()
    return seconds / 3600


def plot_edge_color_value(
    exp: Experiment,
    t: int,
    color_fun: Callable,
    region=None,
    intervals=[[1, 4], [4, 6], [6, 10], [10, 20]],
    cmap=cm.get_cmap("Reds", 100),
    plot_cmap=False,
    v_max=10,
    v_min=0,
    nodes: List[Node] = [],
    downsizing=5,
    dilation=5,
    save_path="",
    color_seed=12,
    dpi=None,
    show_background=True,
    label_colorbar="Width ($\mu m)$",
    figsize=(36, 24),
    figax=None,
    alpha=0.5,
) -> None:
    """
    Plot the width for all the edges at a given timestep.

    :param region: choosen region in the full image, such as [[100, 100], [2000,2000]], if None the full image is shown
    :param color_fun: edge -> float a function of edges that needs to be color plotted

    :param nodes: list of nodes to plot
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save_path: full path to the location where the plot will be saved
    :param intervals: different width intervals that will be given different colors
    :param cmap: a colormap to map width to color
    :param plot_cmap: a boolean, whether or not to plot with cmap
    :param v_max: the max width for the colorbar/colormap
    """
    DIM_X, DIM_Y = get_dimX_dimY(exp)

    if region == None:
        # Full image
        image_coodinates = exp.image_coordinates[t]
        region = get_bounding_box(image_coodinates)
        region[1][0] += DIM_X
        region[1][1] += DIM_Y

    edges = get_all_edges(exp, t)
    if figax is None:
        fig = plt.figure(
            figsize=figsize
        )  # width: 30 cm height: 20 cm # TODO(FK): change dpi
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    # Give colors to edges
    default_color = 1000
    colors = []
    widths = []
    for edge in edges:
        width = color_fun(edge)
        widths.append(width)
        if not plot_cmap:
            color = default_color
            for i, interval in enumerate(intervals):
                if interval[0] <= width and width < interval[1]:
                    color = i + color_seed
            colors.append(color)
    if plot_cmap:
        colors = [cmap((width - v_min) / (v_max - v_min)) for width in widths]
    # 0/ Make color legend
    def convert(c):
        c_ = c / 255
        return (c_[0], c_[1], c_[2])

    if not plot_cmap:
        handles = []
        for i, interval in enumerate(intervals):
            handles.append(
                mpatches.Patch(
                    color=convert(make_random_color(i + color_seed)),
                    label=str(interval),
                )
            )
        handles.append(
            mpatches.Patch(color=convert(make_random_color(1000)), label="out of bound")
        )
        # fig.legend(handles=handles)
    else:
        norm = mpl.colors.Normalize(vmin=v_min, vmax=v_max)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        N = 5
        plt.colorbar(sm, ticks=np.linspace(v_min, v_max, N), label=label_colorbar)
    # 1/ Image layer
    if show_background:
        im, f = reconstruct_image_from_general(
            exp,
            t,
            downsizing=downsizing,
            region=region,
            prettify=False,
            white_background=False,
        )
        f_int = lambda c: f(c).astype(int)

    # 2/ Edges layer
    color_list = (
        [(np.array(color) * 255).astype(int) for color in colors] if plot_cmap else None
    )
    from_edges = reconstruct_skeletton_from_edges(
        exp,
        t,
        edges=edges,
        region=region,
        color_seeds=colors,
        color_list=color_list,
        downsizing=downsizing,
        dilation=dilation,
        timestep=False,
    )
    skel_im, _ = from_edges
    if show_background:
        new_region = [
            f_int(region[0]),
            f_int(region[1]),
        ]  # should be [[0, 0], [d_x/downsized, d_y/downsized]]

    # 3/ Fusing layers
    if show_background:
        ax.imshow(im, cmap="gray", interpolation="none")
    ax.imshow(skel_im, alpha=alpha, interpolation="none")

    # 3/ Plotting the Nodes
    size = 5
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nodes:
        c = node.pos(t)
        if is_in_bounding_box(c, region):
            c = f(node.pos(t))
            node_text = ax.text(
                c[1],
                c[0],
                str(node.label),
                ha="center",
                va="center",
                size=size,
                bbox=bbox_props,
            )

    if save_path:
        plt.savefig(save_path, dpi=dpi)
    else:
        plt.show()
    return fig,ax


def reconstruct_image_from_general(
    exp: Experiment,
    t: int,
    region=None,
    downsizing=5,
    prettify=False,
    white_background=True,
) -> Tuple[List[np.array], Callable[[float, float], float]]:
    """
    This function reconstructs the full size image or a part of it given by `region` at
    timestep `t` and return it as an np array. It also returns a function mapping coordinates
    in the GENERAL referential to coordinates in the reconstructed image referential.

    It is the analog of reconstruct_image but for the general referential.

    :param region: [[x1, y1], [x2, y2]] defining a zone in the GENERAL ref that
                    we want to extract. Can be np.array or lists, int or floats
    :param downsizing: factor by which the image is downsized, 1 returns the original image
    :param prettify: add transformation operation to make the rendering better (but costly)
    :param white_background: if True, areas where no images were found are white, otherwise black
    :param dimx: x dimension of images

    WARNING: without downsizing, the full image is heavy (2 Go)
    NB: returned image shape is ((int(a)-int(c))//downsizing, (int(b)-int(d))//downsizing)
    NB: the typical full region of a full image is [[0, 0], [26000, 52000]]
    NB: the interesting region of a full image is typically [[12000, 15000], [26000, 35000]]
    """

    # TODO(FK): specify explicitly the dtype (here it is 32 why?)

    # Step 1: computing a region in the TIMESTEP referential that contains all point from the original region
    region = format_region(region)

    # Final dimension
    # Rotating the image can slightly change dimension, hence we expand a bit the region
    # and reframe it at the end to have a consistent size returned by the function
    l_x = (region[1][0] - region[0][0]) // downsizing
    l_y = (region[1][1] - region[0][1]) // downsizing
    region = expand_bounding_box(region, margin=downsizing)

    point1, point2 = region
    region_vertices = [point1, point2, [point1[0], point2[1]], [point2[0], point1[1]]]
    region_vertices_ts = [
        exp.general_to_timestep(point, t) for point in region_vertices
    ]
    region_ts = get_bounding_box(region_vertices_ts, margin=0)
    # Step 2: fetch this region
    image_ts, f = reconstruct_image(
        exp,
        t,
        region=region_ts,
        downsizing=downsizing,
        prettify=prettify,
        white_background=white_background,
    )

    # Step 3: extract only the region of interest from the TIMESTEP image
    angle = np.degrees(exp.get_rotation(t))
    extracted_image = extract_inscribed_rotated_image(image_ts, angle=angle)

    # Mapping from GENERAL referential to downsized image referential
    f = lambda c: (np.array(c) - np.array(region[0])) / downsizing

    return extracted_image[:l_x, :l_y, ...], f


if __name__ == "__main__":

    # FOR TEST PURPOSES
    from amftrack.util.sys import (
        update_plate_info_local,
        get_current_folders_local,
        storage_path,
    )
    import os

    plate_name = "20220325_1423_Plate907"
    directory_name = "width1"
    directory = os.path.join(storage_path, directory_name, "full_plates") + "/"

    update_plate_info_local(directory)
    folder_df = get_current_folders_local(directory)
    selected_df = folder_df.loc[folder_df["folder"] == plate_name]
    i = 0
    plate = int(list(selected_df["folder"])[i].split("_")[-1][5:])
    folder_list = list(selected_df["folder"])
    directory_name = folder_list[i]
    exp = Experiment(directory)
    exp.load(selected_df.loc[selected_df["folder"] == directory_name], suffix="")
    exp.load_tile_information(0)

    im, f = reconstruct_skeletton_from_edges(exp, 0, dilation=10)


def plot_hulls_skelet(exp, t, hulls, save_path="", close=True):
    if close:
        plt.close("all")
    my_cmap = cm.Greys
    fig, ax = plt.subplots(figsize=(20, 10))
    skels = []
    ims = []
    kernel = np.ones((5, 5), np.uint8)
    itera = 2
    folders = list(exp.folders["folder"])
    folders.sort()
    for folder in folders[t : t + 1]:
        path_snap = os.path.join(exp.directory, folder)
        skel_info = read_mat(path_snap + "/Analysis/skeleton_realigned_compressed.mat")
        skel = skel_info["skeleton"]
        skels.append(cv2.dilate(skel.astype(np.uint8), kernel, iterations=itera))
    ax.imshow(skels[0], cmap=my_cmap, interpolation=None, alpha=0.7)
    for polygon in hulls:
        p = affine_transform(polygon, [0.2, 0, 0, -0.2, 0, 0])
        p = rotate(p, 90, origin=(0, 0))
        p = gpd.GeoSeries(p)
        try:
            _ = p.boundary.plot(ax=ax, alpha=0.9)
        except ValueError:
            print(p)
        # _ = ax.plot(np.array(y)/5,np.array(x)/5)
    if save_path != "":
        plt.savefig(save_path)
