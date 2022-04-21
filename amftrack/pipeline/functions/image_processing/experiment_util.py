from random import choice
import numpy as np
from typing import List

from amftrack.util.aliases import coord_int
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Node,
    Edge,
)


def get_random_edge(exp: Experiment, t=0) -> Edge:
    "Select randomly an edge of Experiment at timestep t"
    (G, pos) = exp.nx_graph[t], exp.positions[t]
    edge_coord = choice(list(G.edges))
    edge = Edge(Node(edge_coord[0], exp), Node(edge_coord[1], exp), exp)
    return edge
