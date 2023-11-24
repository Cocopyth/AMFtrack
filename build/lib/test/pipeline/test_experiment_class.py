import unittest
import numpy as np
import random

from test.util import helper

from amftrack.pipeline.functions.image_processing.experiment_util import (
    get_random_edge,
)


@unittest.skipUnless(helper.has_test_plate(), "No plate to run the tests..")
class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object()

    def test_get_image_coords(self):
        image_coord = self.exp.get_image_coords(0)
        self.assertIsNotNone(image_coord)

    def test_general_to_timestep_coords(self):
        self.exp.general_to_timestep([145, 345], 0)
        self.exp.timestep_to_general([12, 23], 0)
        a = np.array([12, 12.3])
        b = self.exp.timestep_to_general(self.exp.general_to_timestep(a, 0), 0)
        c = self.exp.general_to_timestep(self.exp.timestep_to_general(a, 0), 0)
        np.testing.assert_array_almost_equal(a, b, 2)
        np.testing.assert_array_almost_equal(a, c, 2)

    def test_general_to_image_coords(self):
        self.exp.image_to_general([145, 345], 0, 12)
        self.exp.general_to_image([12, 23], 0, 15)
        a = np.array([12, 12.3])
        b = self.exp.image_to_general(self.exp.general_to_image(a, 0, 12), 0, 12)
        c = self.exp.general_to_image(self.exp.image_to_general(a, 0, 15), 0, 15)
        np.testing.assert_array_almost_equal(a, b, 2)
        np.testing.assert_array_almost_equal(a, c, 2)

    def test_get_image(self):
        self.exp.get_image(0, 4)

    # @unittest.skip("To fix later")
    def test_show_source_images(self):
        random.seed(10)
        edge = get_random_edge(self.exp)
        edge.begin.show_source_image(0)

    # @unittest.skip("To fix later")
    def test_find_image_pos(self):
        self.exp.find_image_pos(21980, 30916, 0)

    def test_find_im_indexes_from_general(self):
        self.exp.find_im_indexes_from_general(21980, 30916, 0)

    def test_get_rotation(self):
        self.exp.load_image_transformation(0)
        self.assertTrue(self.exp.get_rotation(0) == 0)
