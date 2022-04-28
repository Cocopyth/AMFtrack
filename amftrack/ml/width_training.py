import os
import random
import json
import numpy as np

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Edge,
)
from amftrack.util.sys import data_path
from typing import Dict, List
from amftrack.util.aliases import coord
from amftrack.util.image_analysis import convert_to_micrometer
from amftrack.pipeline.functions.image_processing.experiment_util import (
    find_nearest_edge,
)


def fetch_labels(directory: str) -> Dict[str, List[coord]]:
    """
    Go fetch the labels in the directory.
    :return: directory with image names as key and list of segments as values
    """

    def is_valid(name):
        return ".json" in name

    d = {}
    for file in os.listdir(directory):
        if is_valid(file):

            points = []
            path = os.path.join(directory, file)
            with open(path) as f:
                json_from_file = json.load(f)

            for shape in json_from_file["shapes"]:
                if shape["label"] == "width":
                    points.append(shape["points"])
            # name = os.path.splitext(file)[0] + ".tiff"
            name = json_from_file["imagePath"]
            d[name] = points

    return d


def label_edges(exp: Experiment, t: int) -> Dict[Edge, float]:
    """
    NB: the label are supposed to be in the same folder as the images
    :return: each labeled edge with its width
    """
    # TODO(FK): fix error, coordinates are weird

    label_directory = os.path.join(
        exp.directory, "20220325_1423_Plate907", "Img"
    )  # QUICKFIX
    segment_labels = fetch_labels(label_directory)

    edges_widths = {}

    for image_name in segment_labels.keys():
        # Identify image
        image_path = os.path.join(label_directory, image_name)
        image_index = exp.image_paths[t].index(image_path)
        for [point1, point2] in segment_labels[image_name]:
            point1 = np.array(point1)  # in image ref
            point2 = np.array(point2)
            middle_point = (point1 + point2) / 2
            # Compute width
            width = convert_to_micrometer(
                np.linalg.norm(point1 - point2), magnification=2
            )
            # Identify corresponding edge
            middle_point_ = exp.image_to_general(middle_point, t, image_index)
            edge = find_nearest_edge(middle_point_, exp, t)
            # Add to the dataset
            if edge in edges_widths:
                # NB: could also keep the point of the section for further use
                edges_widths[edge].append(width)
            else:
                edges_widths[edge] = [width]

    edges_width_mean = {}
    for key, list_of_width in edges_width_mean.items():
        edges_width_mean[key] = np.mean(list_of_width)

    return edges_width_mean
