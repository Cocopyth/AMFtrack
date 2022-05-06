from random import choice
import numpy as np
from typing import List, Tuple, Optional
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
from amftrack.util.plot import crop_image, pixel_list_to_matrix
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from amftrack.util.sparse import dilate_coord_list
from random import randrange
from functools import lru_cache

# Prov
from pymatreader import read_mat
import cv2
from amftrack.plotutil import plot_t_tp1


def get_random_edge(exp: Experiment, t=0) -> Edge:
    "Select randomly an edge of Experiment at timestep t"
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


def find_nearest_edge(point: coord, exp: Experiment, t: int) -> Edge:
    """
    Find the nearest edge to `point` in `exp` at timestep `t`.
    The coordonates are given in the GENERAL ref.
    :return: Edge object
    """
    edges = get_all_edges(exp, t)
    l = [edge.pixel_list(t) for edge in edges]
    return edges[get_closest_line_opt(point, l)[0]]


def get_edge_from_node_labels(
    exp: Experiment, t: int, start_node: int, end_node: int
) -> Optional[Edge]:
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    # Verify that the edge exists
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
    :return: np binary image, coordinate of all the points in image
    """
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
    # TODO(FK): merge with plot_edge
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
    save="",
) -> None:
    """
    This is the general purpose function to plot the full image, downsized by a chosen factor
    with additionnal features such as: edges, nodes, points, segments.
    The coordinates for all the objects are provided in the GENERAL referential.

    WARNING: this function computes the full image (2 Go) before downsizing it, it can be expensive

    :param region: [[100, 100], [2000,2000]] a choosen region in the full image (reduces computation also)
    :param edges: list of edges to plot, it is the pixel list that is plotted, not a straight line
    :param nodes: list of nodes to plot
    :param points: points such as [123, 234] to plot with a white cross on the image
    :param segments: plot lines between two points that are provided
    :param downsizing: factor by which we reduce the image resolution (5 -> image 25 times lighter)
    :param dilation: only for edges: thickness of the edges (dilation applied to the pixel list)
    :param save: full path to the location where the plot will be saved
    """
    # TODO(FK): saving image, add nodes, caching the images
    # TODO(FK): move the cropping part and function for changing the coodinates in the reconstruct image function
    # 1/ Image layer
    im = reconstruct_image(exp, t, downsizing=downsizing)

    if region != None:
        for i in range(2):
            for j in range(2):
                region[i][j] = region[i][j] // downsizing
        im = crop_image(im, region)

    if region is None:
        region = [0, 0]
    f = lambda c: list(
        (np.array(exp.general_to_timestep(c, t)) / downsizing).astype(int)
        - np.array(region[0], dtype=int)
    )
    # 2/ Edge layer
    # TODO(FK): use a dilation instead, this is a QUICKFIX
    # kernel = np.ones((dilation, dilation), np.uint8)
    # skel = pixel_list_to_matrix(edge.pixel_list(t), margin=dilation)
    # dilated_skel = cv.dilate(skel.astype(np.uint8), kernel, iterations=1)
    # im.shape, dilation, downsizin, edge, f
    edge_layer = np.zeros(shape=(im.shape[0], im.shape[1], 3), dtype=int)
    size = int(dilation / downsizing)
    for edge in edges:
        skel = [f(coordinates) for coordinates in edge.pixel_list(t)]
        dilated_skel = dilate_coord_list(skel, iteration=size)
        color = np.array([randrange(255) for i in range(3)])
        for c in dilated_skel:
            # TODO check if out of bound here
            x, y = c[0], c[1]
            try:  # TODO(FK): use is_in_image
                edge_layer[y][x][:] = color
            except:
                None

    # Plotting the two layers
    fig = plt.figure(figsize=(12, 8))  # width: 30 cm height: 20 cm
    ax = fig.add_subplot(111)
    ax.imshow(im, cmap="gray", alpha=0.3, interpolation="none")
    ax.imshow(edge_layer, alpha=0.7, interpolation="none")

    # 3/ Plotting the Nodes
    size = 5
    bbox_props = dict(boxstyle="circle", fc="white")
    for node in nodes:
        c = f(list(node.pos(t)))  # TODO(FK): decide betwenn list and np.array
        node_text = ax.text(
            c[0],
            c[1],
            str(node.label),
            ha="center",
            va="center",
            size=size,
            bbox=bbox_props,
        )  # TODO(FK): fix taille inégale des nodes

    # 4/ Plotting coordinates
    points = [f(c) for c in points]
    for c in points:
        # TODO(FK): which order for the coordinates here
        plt.plot(c[1], c[0], marker="x", color="red")

    # 5/ Plotting segments
    segments = [[f(segment[0]), f(segment[1])] for segment in segments]
    for s in segments:
        plt.plot(
            [s[0][1], s[1][1]],
            [s[0][0], s[1][0]],
            color="white",
            linewidth=2,
        )

    plt.show()


# @lru_cache(maxsize=1)  # TODO(FK): make a custom decorator
def reconstruct_image(exp: Experiment, t: int, downsizing=1) -> np.array:
    """
    This function reconstructs the full size image at timestep t and return it as an np array.
    :param downsizing: factor by which the image is downsized, 1 returns the original image
    WARNING: the image is a very heavy object (2 Go)
    NB: To plot objects in this image, the TIMESTEP referential must be used
    """
    # TODO(FK): add window here, return function for plotting
    # TODO(FK): make a decorator to store in TEMP or cache with
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
        full_im, (full_im.shape[1] // downsizing, full_im.shape[0] // downsizing)
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


if __name__ == "__main__":

    # directory = "/data/felix/width1/full_plates/"  # careful: must have the / at the end
    # # update_plate_info(directory)
    # folder_df = get_current_folders(directory)
    # folders = folder_df.loc[folder_df["Plate"] == "907"]
    # time = "3:00:00"
    # threshold = 0.1
    # args = [threshold, directory]
    # run("prune_skel.py", args, folders)
    from amftrack.util.sys import (
        update_plate_info_local,
        get_current_folders_local,
        data_path,
    )
    import os

    # directory_name = "width1"
    # plate_name = "20220325_1423_Plate907"
    # directory = os.path.join(data_path, directory_name, "full_plates") + "/"

    # plate_name = "20220330_2357_Plate19"
    # directory = data_path + "/"

    plate_name = "20220325_1423_Plate907"
    directory_name = "width1"
    directory = os.path.join(data_path, directory_name, "full_plates") + "/"

    update_plate_info_local(directory)
    folder_df = get_current_folders_local(directory)
    selected_df = folder_df.loc[folder_df["folder"] == plate_name]
    i = 0
    plate = int(list(selected_df["folder"])[i].split("_")[-1][5:])
    folder_list = list(selected_df["folder"])
    directory_name = folder_list[i]
    exp = Experiment(plate, directory)
    exp.load(selected_df.loc[selected_df["folder"] == directory_name], labeled=False)
    exp.load_tile_information(0)
    # plot_full_image(exp, 0, downsizing=20)

    # plot_full_image(exp, 0, downsizing=10, region=([1000, 1000], [10000, 10000]))
    # plot_full_image(exp, 0, downsizing=10, region=[[1000, 1000], [10000, 10000]])
    # edges = [get_random_edge(exp, 0) for _ in range(50)]
    # edges = get_all_edges(exp, 0)
    # plot_full_image_with_features(
    #     exp,
    #     edges,
    #     0,
    #     5,
    #     4,
    #     region=[[10000, 10000], [20000, 20000]],
    #     coordinates=[[15000, 15000], [17000, 17000]],
    #     segments=[[[15000, 15000], [17000, 17000]]],
    # )
    # plot_full_image_with_skel(exp, 0)

    # c = [35000, 20000]  # General ref
    # edge = find_nearest_edge(c, exp, 0)
    a = 1

    plot_full_image_with_features(
        exp,
        0,
        points=[[11191, 39042], [11923, 45165]],
        nodes=[Node(10, exp), Node(100, exp), Node(200, exp)],
    )
