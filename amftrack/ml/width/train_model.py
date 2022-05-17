import os

import numpy as np
import tensorflow as tf
from amftrack.ml.callbacks import SavePlots
from amftrack.ml.util import get_intel_on_dataset
from amftrack.ml.width.build_features import get_sets
from amftrack.ml.width.models import MeanLearningModel, first_model
from amftrack.util.sys import storage_path
from tensorflow import keras

### PARAMETERS ###
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
        callbacks=callbacks,
    )

    test_loss, test_acc = model.evaluate(test, verbose=2)
    train_loss, train_acc = model.evaluate(train, verbose=2)

    dummy_model.fit(train)
    test_acc_dummy = dummy_model.evaluate(test, tf.keras.metrics.mean_absolute_error)

    print(f"First model (test): Loss {test_loss} Acc: {test_acc}")
    print(f"First model (train): Loss {train_loss} Acc: {train_acc}")
    print(f"Baseline: Acc: {test_acc_dummy}")


if __name__ == "__main__":
    # main()
    train, valid, test = get_sets(os.path.join(storage_path, "width3", "dataset_2"))
    get_intel_on_dataset(train)
    get_intel_on_dataset(valid)
    get_intel_on_dataset(test)
