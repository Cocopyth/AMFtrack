import numpy as np
import os
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow import keras
from amftrack.ml.width.build_features import get_sets
from amftrack.util.sys import storage_path


class SavePlots(keras.callbacks.Callback):
    """This callback plots the curve for the training and saves them in the chosen directory"""

    def __init__(self, directory_path):
        super(SavePlots, self).__init__()
        self.directory_path = directory_path

    def on_train_end(self, logs=None):
        history = self.model.history
        if not os.path.isdir(self.directory_path):
            os.mkdir(self.directory_path)
        for key in history.history.keys():
            plt.plot(history.history[key])
            plt.title(key)
            plt.xlabel("Epochs")
            plt.ylabel(key)
            plt.savefig(os.path.join(self.directory_path, key))
            plt.close()
