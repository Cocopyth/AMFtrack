import os
import random
import json
import numpy as np
import logging
import pandas as pd
import imageio

from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
    Edge,
)
from skimage.measure import profile_line
from amftrack.util.geometry import (
    compute_factor,
    expand_segment,
    generate_index_along_sequence,
)
from amftrack.util.sys import storage_path
from typing import Dict, List
from amftrack.util.aliases import coord
from amftrack.util.image_analysis import convert_to_micrometer
from amftrack.util.geometry import distance_point_pixel_line
from amftrack.pipeline.functions.image_processing.experiment_util import (
    find_nearest_edge,
    plot_edge_cropped,
    plot_full_image_with_features,
    get_all_edges,
)
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    extract_section_profiles_for_edge_exp,
    compute_section_coordinates,
)
from amftrack.util.sys import storage_path
import cv2
from amftrack.ml.width.config import ORIGINALSIZE

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)


def fetch_labels(directory: str) -> Dict[str, List[coord]]:
    """
    Fetch the labels of width in the `directory`.
    The labels are set with `line_segment` of `labelme` labelling application.
    :param directory: full path to a directory
    :return: directory with image names as key and list of segments as values
    NB: an image usually has several segment of annotation
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


def label_edges(
    exp: Experiment, t: int, folder="20220325_1423_Plate907"
) -> Dict[Edge, float]:
    """
    Fetch the labels, process them and attribute them to their respective edge.
    At a certain timestep t.
    NB: labels are in the same folder as the images
    :return: a dictionnary associating edges with their width
    """
    label_directory = os.path.join(exp.directory, folder, "Img")  # QUICKFIX

    segment_labels = fetch_labels(label_directory)

    edges_widths = {}

    for image_name in segment_labels.keys():
        # Identify image
        image_path = os.path.join(label_directory, image_name)
        # TODO: introduce this function in Experiment
        image_index = exp.image_paths[t].index(image_path)
        for [point1, point2] in segment_labels[image_name]:
            # Inversion of coordinates !
            point1 = [point1[1], point1[0]]
            point2 = [point2[1], point2[0]]
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
            distance = distance_point_pixel_line(
                middle_point_, edge.pixel_list(t), step=1
            )
            # Verification
            # plot_full_image_with_features(
            #     exp, 0, downsizing=10, edges=[edge], points=[middle_point_]
            # )
            if distance > 20:
                logging.warning(
                    f"WARNING: The edge {edge} has a distance of {distance} to its label."
                )
                # TODO(FK): remove edges above 50
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
    :dataset_name: name of the dataset folder, it will be placed in the `storage_path`
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
    dataset_directory = os.path.join(storage_path, dataset_name)
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)
    image_directory = os.path.join(dataset_directory, "Img")
    if not os.path.isdir(image_directory):
        os.mkdir(image_directory)
    preview_directory = os.path.join(dataset_directory, "Preview")
    if not os.path.isdir(preview_directory):
        os.mkdir(preview_directory)
    edge_directory = os.path.join(dataset_directory, "Data")
    if not os.path.isdir(edge_directory):
        os.mkdir(edge_directory)

    # Fect edges and labels
    edges_width_mean = label_edges(exp, t)

    data = {"edge": [], "width": []}  # TODO(FK): add (x, y for each slice)

    f = lambda n: generate_index_along_sequence(n, resolution, offset)

    for edge, width in edges_width_mean.items():
        edge_data = {
            "x1_timestep": [],
            "y1_timestep": [],
            "x2_timestep": [],
            "y2_timestep": [],
            "x1_image": [],
            "y1_image": [],
            "x2_image": [],
            "y2_image": [],
            "width": [],
        }
        if len(edge.pixel_list(t)) > edge_length_limit:
            edge_name = f"{str(edge.begin)}-{str(edge.end)}"
            # Extracting and saving profiles
            (
                profiles,
                list_of_segments,
                new_section_coord_list,
            ) = extract_section_profiles_for_edge_exp(
                exp,
                t,
                edge,
                resolution=resolution,
                offset=offset,
                step=5,
                target_length=ORIGINALSIZE,
            )
            for segment in list_of_segments:
                edge_data["x1_timestep"].append(segment[0][0])
                edge_data["y1_timestep"].append(segment[0][1])
                edge_data["x2_timestep"].append(segment[1][0])
                edge_data["y2_timestep"].append(segment[1][1])
            for segment in new_section_coord_list:
                edge_data["x1_image"].append(segment[0][0])
                edge_data["y1_image"].append(segment[0][1])
                edge_data["x2_image"].append(segment[1][0])
                edge_data["y2_image"].append(segment[1][1])
                edge_data["width"].append(width)

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
            edge_df = pd.DataFrame(edge_data)
            edge_df.to_csv(os.path.join(edge_directory, edge_name) + ".csv")

        else:
            logging.debug("Removing small root..")

    info_df = pd.DataFrame(data)
    info_df.to_csv(os.path.join(dataset_directory, "data.csv"))


def make_precise_dataset(
    directories: List[str], target_length, dataset_name="test_precise_1"
):
    """
    This function proceeds with the following steps:
    - fetching the labels (set with labelme) from the labeled images
    - computes the width value associated to the segment (label)
    - expand the segment to the target feature value and extract the slice to make a feature
    - create and save the data set
    NB: the dataset format is different as before
    :dataset_name: name of the dataset folder, it will be placed in the `storage_path`
    :directories: list of full paths to plate files where the labels are in `Img` subdirectory
    """

    # Make the dataset repository where the data will be stored
    dataset_directory = os.path.join(storage_path, dataset_name)
    if not os.path.isdir(dataset_directory):
        os.mkdir(dataset_directory)

    slices = []
    labels = []
    # Process one plate at the time
    for directory_path in directories:
        label_path = os.path.join(directory_path, "Img")
        segment_labels = fetch_labels(label_path)

        for image_name in segment_labels.keys():
            image_path = os.path.join(label_path, image_name)
            for [point1, point2] in segment_labels[image_name]:
                # Inversion of coordinates !
                point1 = [point1[1], point1[0]]
                point2 = [point2[1], point2[0]]
                point1 = np.array(point1)  # in image ref
                point2 = np.array(point2)
                middle_point = (point1 + point2) / 2
                # Compute width
                width = convert_to_micrometer(
                    np.linalg.norm(point1 - point2), magnification=2
                )
                im = imageio.imread(image_path)
                point1_, point2_ = expand_segment(
                    point1, point2, factor=compute_factor(point1, point2, target_length)
                )
                profile = profile_line(im, point1_, point2_, mode="constant")
                profile = profile[:target_length]
                profile = profile.reshape((1, len(profile)))
                slices.append(profile)
                labels.append(width)
    slice_array = np.concatenate(slices, axis=0)
    label_array = np.array(labels)
    logger.info(f"Slice array: {slice_array.shape} Label array: {label_array.shape}")

    cv2.imwrite(os.path.join(dataset_directory, "slices.png"), slice_array)
    with open(os.path.join(dataset_directory, "labels.npy"), "wb") as f:
        np.save(f, label_array)

    return slice_array, label_array


if __name__ == "__main__":

    ## Useful 1
    # from amftrack.util.sys import (
    #     update_plate_info_local,
    #     get_current_folders_local,
    #     test_path,
    #     storage_path,
    # )

    # directory = os.path.join(storage_path, "width1", "full_plates")
    # plate_name = "20220325_1423_Plate907"
    # update_plate_info_local(directory)
    # folder_df = get_current_folders_local(directory)
    # selected_df = folder_df.loc[folder_df["folder"] == plate_name]
    # i = 0
    # plate = int(list(selected_df["folder"])[i].split("_")[-1][5:])
    # folder_list = list(selected_df["folder"])
    # directory_name = folder_list[i]
    # exp = Experiment(plate, directory)
    # exp.load(selected_df.loc[selected_df["folder"] == directory_name], labeled=False)
    # exp.load_tile_information(0)

    # make_extended_dataset(exp, 0, dataset_name="dataset_test")

    ## Useful 2

    dir = os.path.join(storage_path, "labels_precise")
    directories = [os.path.join(dir, name) for name in os.listdir(dir)]
    make_precise_dataset(directories, 120)
