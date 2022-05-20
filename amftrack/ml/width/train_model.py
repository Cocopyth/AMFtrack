from logging.config import valid_ident
from multiprocessing.spawn import get_preparation_data
import os

import numpy as np
import tensorflow as tf
import logging
from amftrack.ml.callbacks import SavePlots
from amftrack.ml.util import get_intel_on_dataset, make_directory_name
from amftrack.ml.width.build_features import get_sets
from amftrack.ml.width.models import MeanLearningModel, first_model, model_builder
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


def main():
    """
    Training of a single model.
    """
    BATCHSIZE = 32
    model_repository = os.path.join(ml_path, make_directory_name("test"))
    os.mkdir(model_repository)

    f_handler = logging.FileHandler(os.path.join(model_repository, "logs.log"))
    f_handler.setLevel(logging.DEBUG)
    f_logger = logging.getLogger("file_logger")
    f_logger.addHandler(f_handler)

    # 0/ Datasets
    train, valid, test = get_sets(
        os.path.join(storage_path, "width3", "dataset_2"), proportion=[0.4, 0.2, 0.4]
    )

    train = (
        train.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    valid = (
        valid.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    test = (
        test.map(lambda x, y: (data_preparation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    # 2/ Set the model
    model = first_model()

    # dummy_model = MeanLearningModel()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        metrics=[tf.keras.metrics.mean_absolute_error],
    )

    # 3/ Training

    history = model.fit(
        train,
        batch_size=BATCHSIZE,  # TODO(FK): DOUBLE BATCHING?
        epochs=10,
        validation_data=valid,
        callbacks=get_callbacks(model_repository),
    )

    # 4/ Visualize performance
    test_loss, test_acc = model.evaluate(test, verbose=2)
    train_loss, train_acc = model.evaluate(train, verbose=2)

    model.save(os.path.join(model_repository, "saved_model.h5"))

    with open(os.path.join(model_repository, "model_summary.txt"), "w") as fh:
        model.summary(print_fn=lambda x: fh.write(x + "\n"))

    # dummy_model.fit(train)
    # test_acc_dummy = dummy_model.evaluate(test, tf.keras.metrics.mean_absolute_error)
    # print(f"Baseline: Acc: {test_acc_dummy}")

    print(f"First model (test): Loss {test_loss} Acc: {test_acc}")
    print(f"First model (train): Loss {train_loss} Acc: {train_acc}")


def main_hp():
    BATCHSIZE = 32

    # 0/ Datasets
    train, valid, test = get_sets(
        os.path.join(storage_path, "width3", "dataset_2"), proportion=[0.4, 0.2, 0.4]
    )

    train = (
        train.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    valid = (
        valid.map(lambda x, y: (data_augmentation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    test = (
        test.map(lambda x, y: (data_preparation(x, training=True), y))
        .unbatch()
        .batch(BATCHSIZE)
        .prefetch(1)
    )

    # tuner = kt.Hyperband(
    #     model_builder,
    #     objective="val_mean_absolute_error",
    #     max_epochs=30,
    #     factor=3,
    #     directory=os.path.join(storage_path, "test_test"),
    #     project_name="intro_to_kt",
    # )

    tuner = kt.RandomSearch(
        model_builder,
        objective="val_mean_absolute_error",
        max_trials=50,
        # directory=os.path.join(storage_path, "test_test"),
        # project_name="intro_to_kt",
    )

    early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=4,
        verbose=0,
        restore_best_weights=True,
    )

    tuner.search(
        train, epochs=50, validation_data=valid, callbacks=[early_stopping_callback]
    )

    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]

    print(
        f"""
    The hyperparameter search is complete. The optimal learning rate is {best_hps.get('learning_rate')}.
    """
    )


if __name__ == "__main__":
    # main()
    # train, valid, test = get_sets(os.path.join(storage_path, "width3", "dataset_2"))
    # # get_intel_on_dataset(train)
    # # get_intel_on_dataset(valid)
    # # get_intel_on_dataset(test)
    # from amftrack.ml.width.data_augmentation import data_augmentation
    # from amftrack.ml.util import display
    # from amftrack.ml.width.data_augmentation import random_mirror

    # stop = True

    # aug_train = train.map(
    #     lambda x, y: (data_augmentation(x, training=True), y)
    # )  # TODO(FK) training = True?

    # stop = True

    # layer = random_mirror()
    # new = train.map(lambda x, y: (layer(x), y))

    # a = tf.constant([[2.3], [3.2], [4.5], [12.2], [0], [0.0], [1.0]])
    # b = tf.constant([2.0, 3.0, 4.3, 1.0, 1.0, 1.0, 1.0, 1.0])

    # c = 0

    a = 0
    # main()
    main_hp()
