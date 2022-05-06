import unittest
import os
import numpy as np

from amftrack.util.sys import data_path
from amftrack.ml.width_training import fetch_labels, label_edges, make_extended_dataset
from amftrack.util.sys import (
    update_plate_info_local,
    get_current_folders_local,
    test_path,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from test import helper


@unittest.skip("In progress")
class TestWidthTrainingLight(unittest.TestCase):
    # TODO(FK): fix paths
    def setUpClass(cls):
        path = "/media/kahane/KINGSTON/20220325_1423_Plate907/Img"
        fetch_labels(path)


@unittest.skip("In progress")
class TestWidthTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = "/media/kahane/KINGSTON/"
        plate_name = "20220325_1423_Plate907"
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
        cls.exp.load_tile_information(0)

    def test_label_egdes(self):
        label_edges(self.exp, 0)

    def test_make_extended_dataset(self):
        make_extended_dataset(self.exp, 0)
