import numpy as np
import tensorflow as tf
from tensorflow import keras
from amftrack.ml.width.build_features import get_sets
import os
from amftrack.util.sys import storage_path

SLICE_LENGTH = 120
BATCHSIZE = 32


def dummy_model():
    a = 0


def first_model():
    # input = keras.Input(shape=(BATCHSIZE, SLICE_LENGTH))
    input = keras.Input(shape=(SLICE_LENGTH,))
    reshaped = keras.layers.Reshape((120, 1), input_shape=(120,))(input)

    # x = keras.layers.Dense(64, activation="relu")(x)
    conv1 = keras.layers.Conv1D(
        filters=64, kernel_size=8, strides=3, activation="relu", name="conv1"
    )(reshaped)
    conv2 = keras.layers.Conv1D(
        filters=32,
        kernel_size=3,
        strides=3,
        activation="relu",
        name="conv2",
    )(conv1)
    flatten = tf.keras.layers.Flatten()(conv2)
    dense1 = keras.layers.Dense(64, activation="relu", name="dense1")(flatten)
    dense2 = keras.layers.Dense(32, activation="relu", name="dense2")(dense1)
    output = keras.layers.Dense(1, activation=None)(dense2)

    model = keras.Model(inputs=input, outputs=output)
    return model


if __name__ == "__main__":

    # Get the model
    model = first_model()
    # model.summary()

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.mean_squared_error],
    )

    train, valid, test = get_sets(os.path.join(storage_path, "width3", "dataset_2"))

    history = model.fit(
        train,
        batch_size=32,
        epochs=10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=valid,
    )

    test_loss, test_acc = model.evaluate(test, verbose=2)
