import os
import random
import json
import numpy as np
import logging
import pandas as pd

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Edge,
)
from amftrack.util.geometry import generate_index_along_sequence
from amftrack.util.sys import data_path
from typing import Dict, List
from amftrack.util.aliases import coord
from amftrack.util.image_analysis import convert_to_micrometer
from amftrack.pipeline.functions.image_processing.experiment_util import (
    find_nearest_edge,
    plot_edge_cropped,
)
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    extract_section_profiles_for_edge,
)
from amftrack.util.sys import data_path
import cv2

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def fetch_labels(directory: str) -> Dict[str, List[coord]]:
    """
    Go fetch the labels of width in the `directory`.
    The labels are set with `line_segment` of `labelme` labelling application.
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
    Fetch the labels, process them and attribute them to their respective edge.
    NB: labels are supposed to be in the same folder as the images
    :return: a dictionnary associating edges with their width
    """

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
    for key, list_of_width in edges_widths.items():
        edges_width_mean[key] = np.mean(list_of_width)

    return edges_width_mean


def make_extended_dataset(exp: Experiment, t: 0, dataset_name="dataset_test"):
    """
    This function proceeds with the following steps:
    - fetching the labels and processing them from segment to a width value
    - assign each label to its respective edge
    - extract slices along each labeled edge and assign them the width of the edge as label
    - create and save the data set
    :dataset_name: name of the dataset folder, it will be placed in the `data_path`
    NB: the images must have been labeled fist
    At the end the dataset contains
    - `Preview` folder contains the original hypha image with the points where slices where extracted
    - `Img` contains the slices, grouped per hypha
    - `data.csv` contains the label and other information
    """
    # Parameters
    resolution = 5
    offset = 5
    edge_length_limit = 30

    # Make the dataset repository structure
    dataset_directory = os.path.join(data_path, dataset_name)
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)
    image_directory = os.path.join(dataset_directory, "Img")
    if not os.path.isdir(image_directory):
        os.mkdir(image_directory)
    preview_directory = os.path.join(dataset_directory, "Preview")
    if not os.path.isdir(preview_directory):
        os.mkdir(preview_directory)

    # Fect edges and labels
    edges_width_mean = label_edges(exp, t)

    data = {"edge": [], "width": []}  # TODO(FK): add (x, y for each slice)

    f = lambda n: generate_index_along_sequence(n, resolution, offset)

    for edge, width in edges_width_mean.items():
        if len(edge.pixel_list(t)) > edge_length_limit:
            edge_name = f"{str(edge.begin)}-{str(edge.end)}"
            # Extracting and saving profiles
            profiles = extract_section_profiles_for_edge(
                exp,
                t,
                edge,
                resolution=resolution,
                offset=offset,
                step=5,
                target_length=120,
            )
            image_name = edge_name + ".png"
            image_path = os.path.join(image_directory, image_name)
            cv2.imwrite(image_path, profiles)
            # Add information to the csv
            data["edge"].append(edge_name)
            data["width"].append(width)
            # Create preview
            plot_edge_cropped(
                edge,
                t,
                mode=3,
                f=f,
                save_path=os.path.join(preview_directory, edge_name),
            )
        else:
            logging.debug("Removing small root..")

    info_df = pd.DataFrame(data)
    info_df.to_csv(os.path.join(dataset_directory, "data.csv"))
