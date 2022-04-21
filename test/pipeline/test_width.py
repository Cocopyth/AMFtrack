import os
import unittest
from amftrack.util.sys import (
    update_plate_info_local,
    get_current_folders_local,
    test_path,
)
from amftrack.pipeline.functions.image_processing.experiment_class_surf import (
    Experiment,
)
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    chose_pivots,
    generate_pivot_indexes,
)

from test import helper


class TestWidth(unittest.TestCase):
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

    def test_generate_pivot_indexes(self):
        self.assertEqual(
            generate_pivot_indexes(20, resolution=3, offset=4), [4, 7, 10, 13]
        )
        self.assertEqual(
            generate_pivot_indexes(21, resolution=3, offset=4), [4, 7, 10, 13, 16]
        )
        self.assertEqual(generate_pivot_indexes(6, resolution=1, offset=2), [2, 3])
        self.assertEqual(generate_pivot_indexes(6, resolution=2, offset=10), [3])
