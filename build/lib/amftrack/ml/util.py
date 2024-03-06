import tensorflow as tf
from typing import Tuple
import time
import random


def make_directory_name(extension: str) -> str:
    """
    Make a file name of shape:
    DATE-TIME-RANDOM-extension
    The random component is used to prevent having name conflicts
    when the function is called several times in under a second.
    """
    timestr = time.strftime("%Y%m%d-%H%M%S")
    random_component = str(random.randint(0, 99))
    if len(random_component) == 1:
        random_component = "0" + random_component
    return timestr + "-" + random_component + "-" + extension


def get_intel_on_dataset(dataset: tf.data.Dataset) -> None:
    """Count the number of batch in the dataset and give the shapes of features and labels"""
    c = 0
    first = True
    for feature, label in dataset:
        c += 1
        if first:
            shape_feature = feature.shape
            shape_label = label.shape
            first = False

    print(f"Feature batch shape: {shape_feature}")
    print(f"Label batch shape: {shape_label}")
    print(f"Number of batch: {c}")


def count(dataset: tf.data.Dataset):
    c = 0
    for feature, label in dataset:
        c += 1
    return c


def get_nodes(ch: str) -> Tuple[str, str]:
    """
    Return the names of the nodes
    Ex: 12-32 -> (12, 32)
    """
    nodes = ch.split("-")
    print(nodes)
    return nodes[0], nodes[1]


def display(dataset: tf.data.Dataset) -> None:
    """
    Unpack the first element of a tf.Dataset to give intel on the dataset
    """
    feature, label = next(iter(dataset))
    print(feature)
    print(feature.shape)
    try:
        print(label.shape)
    except:
        pass


if __name__ == "__main__":
    a = 0
