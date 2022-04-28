from random import choice
import numpy as np
from typing import List, Tuple

from amftrack.util.aliases import coord_int, coord
import matplotlib.pyplot as plt

from amftrack.util.aliases import coord_int, coord

from amftrack.util.image_analysis import is_in_image
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from amftrack.util.geometry import (
    generate_index_along_sequence,
    distance_point_pixel_line,
    get_closest_line_opt,
)


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
    edge: Edge, t: int, mode=0, points=10
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


def plot_edge(edge: Edge, t: int, mode=2, points=10):
    """
    Plot the Edge in its source images, if one exists.
    :WARNING: If the edge is accross two image, only a part of the edge will be plotted
    :param mode:
    - mode 0: only begin end end
    - mode 1: plot whole edge
    - mode 2: plot only a number of points
    :param points: number of points to plot
    """
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points)
    plt.imshow(im)
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][0], list_of_coord_im[i][1], marker="x", color="red"
        )


def plot_edge_cropped(edge: Edge, t: int, mode=2, points=10, margin=50):
    """
    Same as plot_edge but the image is cropped.
    """
    im, list_of_coord_im = aux_plot_edge(edge, t, mode, points)
    x_min = np.min([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    x_max = np.max([list_of_coord_im[i][0] for i in range(len(list_of_coord_im))])
    y_min = np.min([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])
    y_max = np.max([list_of_coord_im[i][1] for i in range(len(list_of_coord_im))])

    x_min = np.max([0, x_min - margin])
    x_max = np.min([im.shape[1], x_max + margin])  # NB: Careful: inversion (
    y_min = np.max([0, y_min - margin])
    y_max = np.min([im.shape[0], y_max + margin])

    list_of_coord_im = [[p[0] - x_min, p[1] - y_min] for p in list_of_coord_im]
    im = im[int(y_min) : int(y_max), int(x_min) : int(x_max)]
    plt.imshow(im)
    for i in range(len(list_of_coord_im)):
        plt.plot(
            list_of_coord_im[i][0], list_of_coord_im[i][1], marker="x", color="red"
        )


def plot_edge_mask(edge: Edge, t: int):
    """
    Plot the Edge skeletton in its source images.
    TODO(FK): not working for now, we don't see anything
    """
    im, list_of_coord_im = aux_plot_edge(edge, t, mode=1)
    m = np.max(im)
    for coord in list_of_coord_im:
        x, y = coord[0], coord[1]
        if is_in_image(0, 0, x, y):
            im[int(y)][int(x)] = 250  # Careful with the order
    plt.imshow(im)
