import os
import numpy as np
import unittest
import matplotlib.pyplot as plt
from amftrack.util.sys import (
    update_plate_info_local,
    get_current_folders_local,
    test_path,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
    distance_point_edge,
    plot_edge,
    plot_edge_cropped,
    find_nearest_edge,
)
from amftrack.util.sys import test_path
from test import helper


@unittest.skipUnless(helper.has_test_plate(), "No plate to run the tests..")
class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object()

    def test_get_random_edge(self):
        get_random_edge(self.exp)
        get_random_edge(self.exp)

    def test_plot_edge(self):
        edge = get_random_edge(self.exp)
        plot_edge(edge, 0)
        plt.close()

    def test_plot_edge_save(self):
        edge = get_random_edge(self.exp)
        plot_edge(edge, 0, save_path=os.path.join(test_path, "plot_edge_1"))

    def test_plot_edge_cropped(self):
        edge = get_random_edge(self.exp)
        plot_edge_cropped(edge, 0, save_path=os.path.join(test_path, "plot_edge_2"))


class TestExperimentLight(unittest.TestCase):
    def test_distance_point_edge(self):

        edge = helper.EdgeMock([[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]])
        self.assertEqual(
            distance_point_edge([2, 3], edge, 0, step=1),
            0,
        )
