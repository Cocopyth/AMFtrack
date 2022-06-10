import os
import logging
from typing import List
import random
import tensorflow as tf

logger = logging.getLogger(os.path.basename(__file__))
logger.setLevel(logging.DEBUG)

tf.config.run_functions_eagerly(False)


def data_path_from_slice_path(path: str) -> str:
    """
    Transform the full path to slices from an edge to the full path to
    its associated dataframe.
    :param path: full path to the slices of one edge
    :return: full path to the dataframe
    :ex: "/some/repository/dataset/Img/12-34.png" -> "/some/repository/dataset/Data/12-34.csv"
    """
    file_name = os.path.basename(path)
    edge_df_directory = os.path.join(os.path.dirname(os.path.dirname(path)), "Data")
    edge_df_name = os.path.splitext(file_name)[0] + ".csv"
    edge_df_path = os.path.join(edge_df_directory, edge_df_name)
    return edge_df_path


@tf.function
def single_slice_dataset(paths) -> tf.data.Dataset:
    """
    From a tf.Tensor of shape (2,) containing the paths to the slice and to the dataframe,
    returns a dataset yielding labeled slices only from this edge.
    """
    slice_path = paths[0]
    df_path = paths[1]

    raw = tf.io.read_file(slice_path)
    image = tf.image.decode_png(raw, channels=1, dtype=tf.uint8)
    image = tf.cast(image, tf.float32)
    # image = tf.squeeze(image)  # TODO(FK): stop removing last axis here
    slice_dataset = tf.data.Dataset.from_tensor_slices(image)

    label_dataset = tf.data.experimental.CsvDataset(
        df_path,
        [tf.float32],
        select_cols=[9],
        header=True,
    )
    # TODO(FK): how to select column by name
    # NB(FK): could add other features here

    return tf.data.Dataset.zip((slice_dataset, label_dataset))


def reader_dataset(
    file_paths: List[str],
    repeat=1,
    n_readers=5,
    shuffle_buffer_size=1000,  # TODO: make buffer size to size of the dataset
    batch_size=32,  # TODO: remove the batch_size from here
):
    """
    Take as input a list of the paths to the data files. (one file per edge)
    And return a tf.Dataset yielding (randomly) couples (slice, label) slice being of
    shape (120, 1)
    """
    # NB: other option with use of patterns instead of file names: tf.data.Dataset.list_files
    file_path_features = [data_path_from_slice_path(path) for path in file_paths]

    both_path = tf.data.Dataset.from_tensor_slices(
        [list(couple) for couple in zip(file_paths, file_path_features)]
    )
    general_dataset = both_path.interleave(single_slice_dataset, cycle_length=n_readers)
    general_dataset = general_dataset.shuffle(shuffle_buffer_size).repeat(repeat)
    return general_dataset.batch(1)  # Output shape: (1, 120, 1)
    # TODO(FK): remove the batching here and the prefetch  .prefetch(1)


def get_sets(dataset_path: str, proportion=[0.6, 0.2, 0.2]):
    """
    Take as input the full path to a width dataset.
    And the respective proportion of training, validation and test sets.
    Returns the 3 datasets (each as tuple (feature, label))
    """
    if sum(proportion) != 1.0:
        raise Exception("Sum of proportion for all set must be equal to 1")
    # TODO(FK): take as input several dataset_paths instead of one
    # Making a list of all the datafiles
    slices_directory_path = os.path.join(dataset_path, "Img")
    slices_paths = [
        os.path.join(slices_directory_path, file)
        for file in os.listdir(slices_directory_path)
    ]
    # Splitting the list of files in 3 sets
    random.shuffle(slices_paths)  # TODO(FK): set seed here

    l = len(slices_paths)
    bound1 = int(l * proportion[0])
    bound2 = int(l * (proportion[1] + proportion[0]))

    slices_paths_train = slices_paths[0:bound1]
    slices_paths_valid = slices_paths[bound1:bound2]
    slices_paths_test = slices_paths[bound2:]

    logger.info(
        f"Dataset has the following shape: \n\rTrain: {bound1} edges \n\rValid: {bound2-bound1} edges \n\rTest: {l - bound2} edges"
    )

    return (
        reader_dataset(slices_paths_train),
        reader_dataset(slices_paths_valid),
        reader_dataset(slices_paths_test),
    )


if __name__ == "__main__":

    from amftrack.util.sys import storage_path

    dataset_path = os.path.join(storage_path, "width3", "dataset_2")
    a, b, c = get_sets(dataset_path)
    feature, label = next(iter(a))
    print(feature)
