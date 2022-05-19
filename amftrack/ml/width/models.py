import numpy as np
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from amftrack.ml.width.build_features import get_sets
import os
from amftrack.util.sys import storage_path

SLICE_LENGTH = 120
BATCHSIZE = 32


class MeanLearningModel:
    """
    This model learns the mean of the labels.
    And use it to predict the width.
    It is the baseline.
    """

    def __init__(self):
        self.values = []
        self.mean = 0

    def fit(self, train_dataset):
        "Learn the mean of the dataset labels"
        self.values = []
        for _, label in train_dataset:
            new_labels = np.ndarray.flatten(np.array(label))
            for new in new_labels:
                self.values.append(new)
        self.mean = np.sum(self.values) / len(self.values)

    def evaluate(self, test_dataset, metric=tf.metrics.mean_squared_error):
        "Evaluate the chosen error on the test dataset"
        values = []
        for _, label in test_dataset:
            labels = np.ndarray.flatten(np.array(label))
            for new in labels:
                values.append(new)
        return metric(np.ones(len(values)) * self.mean, np.array(values))


def first_model() -> keras.Model:
    # input = keras.Input(shape=(BATCHSIZE, SLICE_LENGTH))
    input = keras.Input(shape=(SLICE_LENGTH,))

    scaling = keras.layers.Rescaling(1.0 / 255)  # TODO(FK): center the data
    # preprocess_layer = keras.layers.Normalization()

    # reshaped = keras.layers.Reshape((120, 1), input_shape=(120,))(scaling(input))

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

    dummy_model = MeanLearningModel()

    ### Baseline
    train, valid, test = get_sets(os.path.join(storage_path, "width3", "dataset_2"))

    model.compile(
        optimizer=keras.optimizers.RMSprop(),  # Optimizer
        # Loss function to minimize
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        # List of metrics to monitor
        metrics=[tf.keras.metrics.mean_absolute_error],
    )

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
    train_loss, train_acc = model.evaluate(train, verbose=2)

    dummy_model.fit(train)
    test_acc_dummy = dummy_model.evaluate(test, tf.keras.metrics.mean_absolute_error)

    print(f"First model (test): Loss {test_loss} Acc: {test_acc}")
    print(f"First model (train): Loss {train_loss} Acc: {train_acc}")
    print(f"Baseline: Acc: {test_acc_dummy}")
