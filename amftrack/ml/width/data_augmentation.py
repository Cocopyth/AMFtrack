from optparse import Values
import tensorflow as tf
from typing import List
import numpy as np

from amftrack.ml.width.config import INPUTSIZE, ORIGINALSIZE


# NB: the data transformations are applied to the whole batch
# They are thus meant to be applied before batching


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
    return tf.keras.layers.Lambda(
        lambda x: random_invert_slice(x, p), name="random_inversion"
    )


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
    return tf.keras.layers.Lambda(
        lambda x: random_mirror_slice(x, p), name=f"random_mirror"
    )


def random_brightness(max_delta=20):
    """
    Wrapper function to generate a random brightness variation layer
    :param max_delta: upper value of the delta of value that is added to all pixels
    """
    return tf.keras.layers.Lambda(
        lambda x: tf.image.random_brightness(x, max_delta),
        name=f"random_brightness_with_delta_{max_delta}",
    )


@tf.function
def random_crop_slice(x, original_size, input_size, offset=0):
    """
    Randomly crop the -2 axis from `original_size` to `input_size`.
    :param offset: used to reduce the range of random crop, offset = input_size/2 is equivalent to center_crop
    :WARNING: must have original_size > input_size + 2*offset
    """
    # TODO(FK): test the crop with varying offset
    start_index = tf.experimental.numpy.random.randint(
        0 + offset,
        high=original_size - input_size - offset,
        dtype=tf.experimental.numpy.int64,
    )
    return x[..., start_index : start_index + input_size, :]


def random_crop(original_size=120, input_size=80, offset=0):
    "Wrapper function to generate a random croping layer"
    return tf.keras.layers.Lambda(
        lambda x: random_crop_slice(x, original_size, input_size, offset),
        name=f"random_crop_with_offset_{offset}",
    )


@tf.function
def center_crop_slice(x, margin):
    "Crops a margin `margin` on both sides of the -2 axis"
    return x[..., margin:-margin, :]


def center_crop(input_size=120, output_size=80):
    """
    Wrapper function to generate a center crop layer
    """
    return tf.keras.layers.Lambda(
        lambda x: center_crop_slice(x, (input_size - output_size) // 2),
        name="center_crop",
    )


data_augmentation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(ORIGINALSIZE, 1)),
        random_crop(ORIGINALSIZE, INPUTSIZE, offset=0),
        # random_invert(p=0.5),
        random_mirror(p=0.5),
        # random_brightness(40),
    ]
)

data_preparation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(ORIGINALSIZE, 1)),
        center_crop(ORIGINALSIZE, INPUTSIZE),
    ]
)


if __name__ == "__main__":

    tf.random.set_seed(11)

    a = tf.constant([[34, 35, 50], [12, 11, 10], [89, 66, 12]], dtype=tf.float32)
    b = tf.constant(
        [
            [[1, 1, 34, 35, 50], [1, 1, 12, 11, 10], [1, 1, 89, 66, 12]],
            [[1, 1, 3, 3, 5], [1, 1, 1, 1, 1], [1, 1, 89, 66, 12]],
        ],
        dtype=tf.float32,
    )
    c = tf.constant(np.ones((5, 5, 5, 5)))
    base = np.ones((1, ORIGINALSIZE, 1))
    d = tf.constant(base)
    base = np.ones((1, ORIGINALSIZE, 1))
    base[:, 0:30, :] = 5.0
    e = tf.constant(base)
    base = np.ones((3, ORIGINALSIZE, 1))
    base[:, 0:30, :] = 5.0
    f = tf.constant(base)

    layer0 = center_crop()
    layer1 = random_crop(80)
    layer2 = random_invert(1)

    stop = True
