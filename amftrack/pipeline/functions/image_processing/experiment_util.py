from random import choice
import numpy as np
from typing import List, Tuple

from amftrack.util.aliases import coord_int, coord
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)
from amftrack.util.geometry import (
    generate_index_along_sequence,
    distance_point_pixel_line,
)


def get_random_edge(exp: Experiment, t=0) -> Edge:
    "Select randomly an edge of Experiment at timestep t"
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edge_coord = choice(list(G.edges))
    edge = Edge(Node(edge_coord[0], exp), Node(edge_coord[1], exp), exp)
    return edge


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
