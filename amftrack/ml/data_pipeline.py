import tensorflow as tf
import numpy as np
import os
import pandas as pd
import pathlib
import logging
import sys
from typing import Tuple

from amftrack.util.sys import storage_path
from amftrack.util.file import chose_file

##### LOGING #####
logger = logging.getLogger("notebook")
logger.setLevel(logging.DEBUG)

##### PARAMETERS #####
# Move elsewhere
BATCHSIZE = 32
BUFFERSIZE = 12
INPUTSIZE = 120
SHUFFLE_BUFFER = 10
dataset_path = os.path.join(storage_path, "width3", "dataset_2")

##### EAGER MODE #####
# tf.config.functions_run_eagerly()
# tf.config.run_functions_eagerly(True)
# tf.data.experimental.enable_debug_mode()  # statement above doesn't apply to tf.data


def get_nodes(ch: str) -> Tuple[str, str]:
    """
    Return the names of the nodes
    Ex: 12-32 -> (12, 32)

    """
    nodes = ch.split("-")
    print(nodes)
    return nodes[0], nodes[1]


def image_path_to_df_path(path: str) -> str:
    """
    Gives the path to the edge dataframe from the path to the edge slices.
    """
    image_name = os.path.basename(str(path))
    image_name_without_ext = os.path.splitext(image_name)[0]
    df_name = image_name_without_ext + ".csv"
    edge_data_full_path = os.path.join(edge_data_path, df_name)
    return edge_data_full_path


def tf_image_path_to_df_path(path):
    """
    Equivalent of image_path_to_df_path, but wrapped as a tf function to be used
    in the pipeline
    """
    im_shape = path.shape
    [
        path,
    ] = tf.py_function(image_path_to_df_path, [path], [tf.string])
    path.set_shape(im_shape)
    return path


# @tf.function
def load_image(filename):
    "From one file name (corresponding to an edge). Loads the data associated with this one file/edge."
    logger.info(f"Type of file: {type(filename)}")
    logger.info(f"File name: {filename}")
    # 1/ Slices from the edge
    raw = tf.io.read_file(filename)  # open the file
    image = tf.image.decode_png(raw, channels=1, dtype=tf.uint8)
    logger.debug(f"Initial shape: {image.shape}")
    # image = image.set_shape([None, 120, 1])
    image = tf.squeeze(image)  # removing the last axis
    # TODO (FK): chose here only part of the array
    logger.debug(f"Final shape: {image.shape}")
    # TODO (FK): why don't I get the shape
    slice_dataset = tf.data.Dataset.from_tensor_slices(image)

    # 2/ Information on the edge
    # TODO (FK): verify order for the edge dataframe
    # edge_name = os.path.splitext(os.path.basename(str(filename)))[0] + ".csv" # PB: not a tensor
    # edge_data_full_path = os.path.join(edge_data_path, edge_name)
    edge_data_full_path = tf_image_path_to_df_path(filename)

    l_dataset = tf.data.experimental.CsvDataset(
        edge_data_full_path,
        [tf.float32],
        select_cols=[9],  # Only parse last three columns
        header=True,
    )
    # TODO(FK): how to select column by name
    # TODO(FK): how to combine columns of features

    # edge_data_full_path = tf.map_fn(filename, image_path_to_df_path)
    # logger.debug(edge_data_full_path)
    # edge_df = pd.read_csv(edge_data_full_path)
    # logger.debug(edge_df.columns)
    # feature_dataset = edge_df[["x1_image", "y2_image"]]

    # 3/ Labels
    # label_dataset = tf.data.Dataset.from_tensor_slices(
    #     tf.convert_to_tensor(edge_df[["width"]])
    # )

    return tf.data.Dataset.zip((slice_dataset, l_dataset))
    # return slice_dataset


def reader_dataset(
    filepaths, repeat=1, n_readers=5, shuffle_buffer_size=1000, batch_size=32
):
    """
    Take as input a list of the paths to the data files.
    And return a tf.Dataset object iterating through the batches.
    """
    path_dataset = tf.data.Dataset.list_files(filepaths)  # yield file names randomly
    # TODO: make buffer size to size of the dataset
    # TODO: get a dataset object from one file
    general_dataset = path_dataset.interleave(load_image, cycle_length=n_readers)
    general_dataset = general_dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return general_dataset.batch(batch_size).prefetch(1)


if __name__ == "__main__":

    section_path = os.path.join(dataset_path, "Img")
    edge_data_path = os.path.join(dataset_path, "Data")
    filepaths = [os.path.join(section_path, file) for file in os.listdir(section_path)]

    # test1 = "/media/kahane/AMFtopology02/storage/width3/dataset_2/Img/1122-1227.png"
    # test2 = "/media/kahane/AMFtopology02/storage/width3/dataset_2/Img/1122.png"

    # test3 = tf.constant([test1, test2], dtype=tf.string)

    # image_path_to_df_path(txt)

    # b = load_image(chose_file(section_path))
    # for elem in b:
    #     print(elem)

    # path_dataset = tf.data.Dataset.list_files(filepaths)

    # for e in path_dataset:
    #     print(e)

    # general_dataset = path_dataset.interleave(load_image, cycle_length=3)

    # general_dataset = general_dataset.shuffle(shuffle_buffer_size).repeat(repeat)

    a = reader_dataset(filepaths)

    a = 1
