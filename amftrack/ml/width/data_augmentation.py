import tensorflow as tf


@tf.function
def random_invert_slice(x, p=0.5):
    """Inversing blacks with whites"""
    if tf.random.uniform([]) < p:
        x = 255.0 - x
    else:
        x
    return x


def random_invert(p=0.5) -> tf.keras.layers.Layer:
    return tf.keras.layers.Lambda(lambda x: random_invert_slice(x, p))


@tf.function
def random_mirror_slice(x, p=0.5):
    """
    Mirror transformation along the last axis with respect to the centered vertical
    WARNING: Careful, this transformation doesn't work if the input shape is (None, 120, 1) (wrong axis)
    """
    if tf.random.uniform([]) < p:
        x = tf.reverse(x, axis=(-1,))
    return x


def random_mirror(p=0.5) -> tf.keras.layers.Layer:
    return tf.keras.layers.Lambda(lambda x: random_mirror_slice(x, p))



data_augmentation = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(120,)),  # TODO(FK): change here the shape
        random_invert(p=0.5),
        random_mirror(p=0.5),
    ]
)


if __name__ == "__main__":

    a = tf.constant([[34, 35, 50], [12, 11, 10], [89, 66, 12]], dtype=tf.float32)
    b = tf.constant(
        [
            [[34, 35, 50], [12, 11, 10], [89, 66, 12]],
            [[3, 3, 5], [1, 1, 1], [89, 66, 12]],
        ],
        dtype=tf.float32,
    )

    layer1 = random_invert(1)

    stop = True
