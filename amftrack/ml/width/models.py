import numpy as np
import os
from sklearn import preprocessing
import tensorflow as tf
from tensorflow import keras
from amftrack.ml.width.build_features import get_sets
from amftrack.util.sys import storage_path
from amftrack.ml.width.config import INPUTSIZE


class MeanLearningModel:
    """
    This model learns the mean of the labels.
    And use it to predict the width.
    It is the baseline.
    """

    def __init__(self):
        self.values = []
        self.mean = 0

    def fit(self, train_dataset, **keyargs):
        "Learn the mean of the dataset labels"
        self.values = []
        for _, label in train_dataset:
            new_labels = np.ndarray.flatten(np.array(label))
            for new in new_labels:
                self.values.append(new)
        self.mean = np.sum(self.values) / len(self.values)

    def evaluate(self, test_dataset, metric=tf.metrics.mean_absolute_error, **kargs):
        "Evaluate the chosen error on the test dataset"
        values = []
        for _, label in test_dataset:
            labels = np.ndarray.flatten(np.array(label))
            for new in labels:
                values.append(new)
        return 0, metric(np.ones(len(values)) * self.mean, np.array(values))

    def compile(self, *args, **kargs):
        pass

    def save(self, *args, **kargs):
        pass

    def summary(self, *args, **kargs):
        pass


def first_model() -> keras.Model:
    "Returns a simple keras model"
    input = keras.Input(shape=(INPUTSIZE, 1))
    scaling = keras.layers.Rescaling(1.0 / 255)(input)  # TODO(FK): center the data
    conv1 = keras.layers.Conv1D(
        filters=64, kernel_size=8, strides=3, activation="relu", name="conv1"
    )(scaling)
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


def hyper_model_builder_simple(parameters):
    def model_builder(hp):
        conv_factor = hp.Choice("conv", values=parameters["conv"])
        dense_factor = hp.Choice("dense", values=parameters["dense"])

        input = keras.Input(shape=(INPUTSIZE, 1))
        scaling = keras.layers.Rescaling(1.0 / 255)(input)  # TODO(FK): center the data
        conv1 = keras.layers.Conv1D(
            filters=64 * conv_factor,
            kernel_size=8,
            strides=3,
            activation="relu",
            name="conv1",
        )(scaling)
        conv2 = keras.layers.Conv1D(
            filters=32 * conv_factor,
            kernel_size=3,
            strides=3,
            activation="relu",
            name="conv2",
        )(conv1)
        flatten = tf.keras.layers.Flatten()(conv2)
        dense1 = keras.layers.Dense(
            16 * dense_factor, activation="relu", name="dense1"
        )(flatten)
        dense2 = keras.layers.Dense(4 * dense_factor, activation="relu", name="dense2")(
            dense1
        )
        output = keras.layers.Dense(1, activation=None)(dense2)
        model = keras.Model(inputs=input, outputs=output)
        hp_learning_rate = hp.Choice(
            "learning_rate", values=parameters["learning_rate"]
        )
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
            metrics=[tf.keras.metrics.mean_absolute_error],
        )
        return model

    return model_builder


def build_model_dense(hp):
    """
    This function generates models of type MLP of different size.
    It is a small base line.
    """
    input_size = hp.Fixed("input_size", value=80)
    inputs = tf.keras.Input(shape=(input_size, 1))
    reshaping = tf.keras.layers.Reshape((input_size,))(inputs)
    scaling = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)(reshaping)
    x = scaling
    hidden_size = hp.Int("hidden_size", 10, 100, step=32, default=32)
    regul = hp.Float("regul", 1e-5, 1e-1, sampling="log", default=1e-4)
    for i in range(hp.Int("dense_blocks", 1, 10, default=3)):
        x = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1(regul),
        )(x)
        x = tf.keras.layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.5))(
            x
        )
    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", 1e-5, 1e-1, sampling="log", default=1e-3
            )
        ),
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        metrics=[tf.keras.metrics.mean_absolute_error],
    )
    return model


def build_model_conv(hp):
    """
    This function builds a model made of 1D convolutions followed by a dense network
    """
    input_size = hp.Fixed("input_size", value=80)
    inputs = tf.keras.Input(shape=(input_size, 1))
    scaling = keras.layers.Rescaling(scale=1.0 / 127.5, offset=-1)(inputs)
    x = scaling
    regul = hp.Float("regul", 1e-5, 1e-1, sampling="log", default=1e-4)
    batch_normalization = hp.Boolean("batch_normalization", default=False)
    for i in range(hp.Int("conv_blocks", 1, 3, default=2)):
        filters = hp.Int("filters_" + str(i), 32, 256, step=32)
        kernel_size = hp.Int("kernel_size" + str(i), 5, 20)
        x = tf.keras.layers.Convolution1D(
            filters, kernel_size=10, kernel_regularizer=tf.keras.regularizers.L1(regul)
        )(x)
        if batch_normalization:
            x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    pooling = hp.Choice("pooling_" + str(i), ["avg", "max", "none"])
    if pooling == "max":
        x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    elif pooling == "avg":
        x = tf.keras.layers.AveragePooling1D(pool_size=2)(x)
    x = tf.keras.layers.Flatten()(x)
    hidden_size = hp.Int("hidden_size", 10, 100, step=32, default=32)
    for i in range(hp.Int("dense_blocks", 1, 3, default=1)):
        x = tf.keras.layers.Dense(
            hidden_size,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.L1(regul),
        )(x)
    x = tf.keras.layers.Dropout(hp.Float("dropout", 0, 0.5, step=0.1, default=0.5))(x)

    outputs = tf.keras.layers.Dense(1)(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=hp.Float(
                "learning_rate", 1e-5, 1e-1, sampling="log", default=1e-3
            )
        ),
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        metrics=[tf.keras.metrics.mean_absolute_error],
    )
    return model


def build_model_main(hp):
    """
    This is the main function for generating models with different architechtures and hyper parameters
    """
    model_type = hp.Choice("model_type", ["dense", "conv"], default="type1")
    if model_type == "dense":
        with hp.conditional_scope("model_type", "dense"):
            model = build_model_dense(hp)
    if model_type == "conv":
        with hp.conditional_scope("model_type", "conv"):
            model = build_model_conv(hp)
    return model


if __name__ == "__main__":

    # Get the model
    model = first_model(80)
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
