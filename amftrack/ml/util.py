import tensorflow as tf
from typing import Tuple


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


def get_nodes(ch: str) -> Tuple[str, str]:
    """
    Return the names of the nodes
    Ex: 12-32 -> (12, 32)
    """
    nodes = ch.split("-")
    print(nodes)
    return nodes[0], nodes[1]


def display(dataset):
    feature, label = next(iter(dataset))
    print(feature)
