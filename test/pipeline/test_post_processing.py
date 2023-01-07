import os
import unittest
from test.util import helper
import matplotlib.pyplot as plt
import amftrack.pipeline.functions.post_processing.exp_plot as exp_plot
import amftrack.pipeline.functions.post_processing.time_plate as time_plate
import amftrack.pipeline.functions.post_processing.time_hypha as time_hypha
import amftrack.pipeline.functions.post_processing.time_edge as time_edge
import amftrack.pipeline.functions.post_processing.area_hulls as area_hulls

from random import choice
import matplotlib as mpl
import json
from amftrack.util.sys import test_path
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Edge,
    Node,
)
import numpy as np

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
            print(f, getattr(time_plate, f)(self.exp, 0))

    def test_area_hulls_f(self):
        fs = [area_hulls.get_biovolume_density_in_ring]
        args = {'incr':100,'i':0}
        for f in fs:
            print(f, f(self.exp, 2,args))

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

    def test_time_edge_f(self):
        fs = dir(time_edge)
        plot_fs = [f for f in fs if f.split("_")[0] == "get"]
        edge = choice(list(self.exp.nx_graph[0].edges))
        edge = Edge(
            Node(np.min(edge), self.exp), Node(np.max(edge), self.exp), self.exp
        )
        data_hypha = {}
        for f in plot_fs:
            column, result = getattr(time_edge, f)(edge, 0)
        data_hypha[column] = result
        path_hyph_info = os.path.join(test_path, "time_edge.json")
        with open(path_hyph_info, "w") as jsonf:
            json.dump(data_hypha, jsonf, indent=4)
