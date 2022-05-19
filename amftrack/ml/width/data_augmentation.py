from optparse import Values
import tensorflow as tf
from typing import List
import numpy as np


@tf.function
def random_invert_slice(x, p=0.5):
    """Inversing blacks with whites"""
    if tf.random.uniform([]) < p:
        x = 255.0 - x
    else:
        x
    return x


def random_invert(p=0.5) -> tf.keras.layers.Layer:
    "Wrapper function to generate inversion lambda layer"
    return tf.keras.layers.Lambda(lambda x: random_invert_slice(x, p))


@tf.function
def random_mirror_slice(x, p=0.5, axis=(-2,)):
    """
    Mirror transformation along the last axis with respect to the centered vertical
    NB: for the axis:
    - (120 , 1) -> axis = (-1,)
    - (120) -> axis = (-2, )
    """
    if tf.random.uniform([]) < p:
        x = tf.reverse(x, axis=axis)
    return x


def random_mirror(p=0.5) -> tf.keras.layers.Layer:
    "Wrapper function to generate mirror lambda layer"
    return tf.keras.layers.Lambda(lambda x: random_mirror_slice(x, p))


def random_brightness(max_delta=50):
    "Upper value of the delta of value that is added to all pixels."
    return tf.keras.layers.Lambda(lambda x: tf.image.random_brightness(x, max_delta))


def random_crop(output_size):
    # TODO(FK): add name
    # TODO(FK): handle size
    size = [1, output_size, 1]
    return tf.keras.layers.Lambda(lambda x: tf.image.random_crop(x, size=size))


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(120, 1)),  # TODO(FK): change here the shape
        random_crop(80),
        random_invert(p=0.5),  # TODO(FK): keep?
        random_mirror(p=0.5),
        random_brightness(50),
    ]
)


if __name__ == "__main__":

    a = tf.constant([[34, 35, 50], [12, 11, 10], [89, 66, 12]], dtype=tf.float32)
    b = tf.constant(
        [
            [[1, 1, 34, 35, 50], [1, 1, 12, 11, 10], [1, 1, 89, 66, 12]],
            [[1, 1, 3, 3, 5], [1, 1, 1, 1, 1], [1, 1, 89, 66, 12]],
        ],
        dtype=tf.float32,
    )
    c = tf.constant(np.ones((5, 5, 5, 5)))
    d = tf.constant(np.ones((1, 120, 1)))

    layer1 = random_crop(80)
    layer2 = random_invert(1)

    stop = True
