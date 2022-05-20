from logging.config import valid_ident
from multiprocessing.spawn import get_preparation_data
import os

import numpy as np
import tensorflow as tf
import logging
from amftrack.ml.callbacks import SavePlots
from amftrack.ml.util import get_intel_on_dataset, make_directory_name
from amftrack.ml.width.build_features import get_sets
from amftrack.ml.width.models import (
    MeanLearningModel,
    first_model,
    hyper_model_builder_simple,
)
from amftrack.ml.width.data_augmentation import data_augmentation, data_preparation
from amftrack.util.sys import storage_path, ml_path
from tensorflow import keras
import keras_tuner as kt


def get_callbacks(model_path: str):
    """
    Creates the callbacks for training, from the path to the acquisition directory of the model
    """
    tensorboard_path = os.path.join(os.path.dirname(model_path), "tensorboard")
    tb_callback = tf.keras.callbacks.TensorBoard(tensorboard_path, update_freq=5)
    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        verbose=0,
        restore_best_weights=True,
    )
    reduce_lr_callback = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=5,
        verbose=0,
        min_delta=0.001,
        cooldown=0,
    )
    plot_callback = SavePlots(model_path)
    csv_logger = tf.keras.callbacks.CSVLogger(os.path.join(model_path, "training.log"))

    callbacks = [
        csv_logger,
        reduce_lr_callback,
        early_stopping_callback,
        tb_callback,
        plot_callback,
    ]

    return callbacks


def train_model(model: tf.keras.Model, name="test"):
    """
    Training of a single model.
    """
    BATCHSIZE = 16
    model_repository = os.path.join(ml_path, make_directory_name(name))
    proportions = [0.5, 0.1, 0.4]
    lr = 0.001

    os.mkdir(model_repository)

    f_handler = logging.FileHandler(os.path.join(model_repository, "logs.log"))
    f_handler.setLevel(logging.DEBUG)
    f_logger = logging.getLogger("file_logger")
    f_logger.addHandler(f_handler)

    f_logger.info(f"Batchsize: {BATCHSIZE}")
    f_logger.info(f"Split proportions: {proportions}")
    f_logger.info(f"Learning rate: {lr}")

    # 0/ Datasets
    train, valid, test = get_sets(
        os.path.join(storage_path, "width3", "dataset_2"), proportion=proportions
    )

    train = (
        train.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid = (
        valid.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test = (
        test.map(lambda x, y: (data_preparation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # 2/ Set the model

    # dummy_model = MeanLearningModel()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=lr),
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        metrics=[tf.keras.metrics.mean_absolute_error],
    )

    # 3/ Training

    history = model.fit(
        train,
        epochs=10,
        validation_data=valid,
        callbacks=get_callbacks(model_repository),
    )

    # 4/ Saving

    model.save(os.path.join(model_repository, "saved_model.h5"))

    with open(os.path.join(model_repository, "model_summary.txt"), "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # 5/ Evaluate performance
    test_loss, test_acc = model.evaluate(test, verbose=2)
    train_loss, train_acc = model.evaluate(train, verbose=2)

    f_logger.info(f"First model (test): Loss {test_loss} Acc: {test_acc}")
    f_logger.info(f"First model (train): Loss {train_loss} Acc: {train_acc}")


def train_hyper_model():
    BATCHSIZE = 32

    # 0/ Datasets
    train, valid, test = get_sets(
        os.path.join(storage_path, "width3", "dataset_2"), proportion=[0.6, 0.35, 0.05]
    )

    train = (
        train.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    valid = (
        valid.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    test = (
        test.map(lambda x, y: (data_preparation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(tf.data.AUTOTUNE)
    )

    # tuner = kt.Hyperband(
    #     model_builder,
    #     objective="val_mean_absolute_error",
    #     max_epochs=30,
    #     factor=3,
    #     directory=os.path.join(storage_path, "test_test"),
    #     project_name="intro_to_kt",
    # )

    parameters = {
        "learning_rate": [0.05, 0.07, 1e-2, 0.015, 1e-3, 1e-4, 1e-5],
        "conv": [1, 2, 4, 8, 16],
        "dense": [1, 2, 4, 8, 16],
        # TODO(FK): add regularization parameter here
    }

    tuner = kt.RandomSearch(
        hyper_model_builder_simple(parameters),
        objective="val_mean_absolute_error",
        max_trials=45,
        directory=os.path.join(storage_path, "test_test"),
        project_name="intro_to_kt",
        overwrite=True,
    )

    # early_stopping_callback = tf.keras.callbacks.EarlyStopping(
    #     monitor="val_loss",
    #     patience=6,
    #     verbose=0,
    #     restore_best_weights=True,
    # )
    # callbacks=[early_stopping_callback]
    callbacks = []

    tuner.search(train, epochs=30, validation_data=valid, callbacks=callbacks)

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal learning rate is {best_hps.get('learning_rate')}.
    """
    )


if __name__ == "__main__":

    from amftrack.ml.util import count, display

    # train_model(MeanLearningModel())
    # train_model(first_model())
    train_hyper_model()
