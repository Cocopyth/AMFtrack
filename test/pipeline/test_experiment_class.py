import os
import unittest
import numpy as np

from amftrack.util.sys import (
    update_plate_info_local,
    get_current_folders_local,
    test_path,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)

from test import helper

# @unittest.skipUnless(helper.has_test_plate(), "No plate to run the tests..")
class TestExperiment(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = test_path + "/"  # TODO(FK): fix this error
        plate_name = "20220330_2357_Plate19"
        update_plate_info_local(directory)
        folder_df = get_current_folders_local(directory)
        selected_df = folder_df.loc[folder_df["folder"] == plate_name]
        i = 0
        plate = int(list(selected_df["folder"])[i].split("_")[-1][5:])
        folder_list = list(selected_df["folder"])
        directory_name = folder_list[i]
        cls.exp = Experiment(plate, directory)
        cls.exp.load(
            selected_df.loc[selected_df["folder"] == directory_name], labeled=False
        )

    def test_find_pos(self):
        r = self.exp.find_image_pos(10000, 5000, t=0)

    def test_get_image_coords(self):
        image_coord = self.exp.get_image_coords(0)
        self.assertIsNotNone(image_coord)

    def test_general_to_image_coords(self):
        self.exp.general_to_image_coords([145, 345], 0)
        self.exp.image_to_general_coords([12, 23], 0)
        a = np.array([12, 12.3])
        b = self.exp.image_to_general_coords(self.exp.image_to_general_coords(a, 0), 0)
        np.testing.assert_array_almost_equal(a, b, 2)

    def test_get_image(self):
        self.exp.get_image(0, 4)
