import tensorflow as tf


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
