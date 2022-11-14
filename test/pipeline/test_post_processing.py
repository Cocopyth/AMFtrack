import os
import numpy as np
import random
import unittest
import matplotlib as mpl
from test import helper
import matplotlib.pyplot as plt
import amftrack.pipeline.functions.post_processing.exp_plot as exp_plot
import amftrack.pipeline.functions.post_processing.time_plate as time_plate
import amftrack.pipeline.functions.post_processing.time_hypha as time_hypha
from random import choice
import matplotlib as mpl
import json
from amftrack.util.sys import test_path

mpl.use("AGG")


class TestExperiment(unittest.TestCase):
    """Tests that need only a static plate with one timestep"""

    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object_analysis()

    def tearDown(self):
        "Runs after each test"
        plt.close("all")

    def test_exp_plot_f(self):
        fs = dir(exp_plot)
        plot_fs = [f for f in fs if f.split("_")[0] == "plot"]
        for f in plot_fs:
            getattr(exp_plot, f)(self.exp)

    def test_time_plate_f(self):
        fs = dir(time_plate)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        for f in plot_fs:
            print(f,getattr(time_plate, f)(self.exp, 0))

    def test_time_hypha_f(self):
        fs = dir(time_hypha)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        hypha = choice(self.exp.hyphaes)
        data_hypha = {}
        for f in plot_fs:
            column, result = getattr(time_hypha, f)(hypha, 0, 1)
        data_hypha[column] = result
        path_hyph_info = os.path.join(test_path, "time_hypha.json")
        with open(path_hyph_info, "w") as jsonf:
            json.dump(data_hypha, jsonf, indent=4)
