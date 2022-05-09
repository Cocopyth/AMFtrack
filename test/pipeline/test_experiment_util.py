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
    Node,
    Edge,
)
from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
    distance_point_edge,
    plot_edge,
    plot_edge_cropped,
    find_nearest_edge,
    get_edge_from_node_labels,
    plot_full_image_with_features,
    get_all_edges,
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

    def test_get_edge_from_node_labels(self):
        # Valid edge
        edge = get_random_edge(self.exp)
        self.assertIsNotNone(
            get_edge_from_node_labels(self.exp, 0, edge.begin.label, edge.end.label)
        )
        self.assertIsNotNone(
            get_edge_from_node_labels(self.exp, 0, edge.end.label, edge.begin.label)
        )
        # Invalid case
        self.assertIsNone(get_edge_from_node_labels(self.exp, 0, 100, 100))

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

    def test_plot_full_image_with_features(self):
        plot_full_image_with_features(
            self.exp,
            0,
            downsizing=10,
            points=[[11191, 39042], [11923, 45165]],
            segments=[[[11191, 39042], [11923, 45165]]],
            nodes=[Node(10, self.exp), Node(100, self.exp), Node(200, self.exp)],
            edges=[get_random_edge(self.exp), get_random_edge(self.exp)],
            dilation=1,
            save_path=os.path.join(test_path, "plot_full"),
        )

    def test_get_all_edges(self):
        get_all_edges(self.exp, t=0)


class TestExperimentLight(unittest.TestCase):
    def test_distance_point_edge(self):

        edge = helper.EdgeMock([[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]])
        self.assertEqual(
            distance_point_edge([2, 3], edge, 0, step=1),
            0,
        )
