from multiprocessing.spawn import get_preparation_data
import os

import numpy as np
import tensorflow as tf
from amftrack.ml.callbacks import SavePlots
from amftrack.ml.util import get_intel_on_dataset
from amftrack.ml.width.build_features import get_sets
from amftrack.ml.width.config import BATCHSIZE
from amftrack.ml.width.models import MeanLearningModel, first_model
from amftrack.ml.width.data_augmentation import data_augmentation, data_preparation
from amftrack.util.sys import storage_path
from tensorflow import keras

### PARAMETERS ### TODO(FK)
tensorboard_path = os.path.join(storage_path, "test", "model_test", "logs")
other_path = os.path.join(storage_path, "test", "model_test", "training.log")
last_path = os.path.join(storage_path, "test", "model_test", "directory")

### CALLBACKS ###
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
plot_callback = SavePlots(last_path)
csv_logger = tf.keras.callbacks.CSVLogger(other_path)

callbacks = [
    csv_logger,
    reduce_lr_callback,
    early_stopping_callback,
    tb_callback,
    plot_callback,
]


def main():
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

    # 2/ Set the model
    model = first_model(80)
    # dummy_model = MeanLearningModel()
    model.compile(
        optimizer=keras.optimizers.RMSprop(learning_rate=0.0001),
        loss=keras.losses.MeanSquaredError(name="mean_squared_error"),
        metrics=[tf.keras.metrics.mean_absolute_error],
    )

    # 3/ Training
    history = model.fit(
        train,
        batch_size=BATCHSIZE,
        epochs=10,
        # We pass some validation for
        # monitoring validation loss and metrics
        # at the end of each epoch
        validation_data=valid,
        callbacks=callbacks,
    )

    # 4/ Visualize performance
    test_loss, test_acc = model.evaluate(test, verbose=2)
    train_loss, train_acc = model.evaluate(train, verbose=2)

    # dummy_model.fit(train)
    # test_acc_dummy = dummy_model.evaluate(test, tf.keras.metrics.mean_absolute_error)
    # print(f"Baseline: Acc: {test_acc_dummy}")

    print(f"First model (test): Loss {test_loss} Acc: {test_acc}")
    print(f"First model (train): Loss {train_loss} Acc: {train_acc}")


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
    main()
