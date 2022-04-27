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
    generate_pivot_indexes,
    compute_section_coordinates,
    find_source_images,
    extract_section_profiles_for_edge,
)
from amftrack.pipeline.functions.image_processing.experiment_util import get_random_edge

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

    def test_compute_section_coordinates(self):
        pixel_list = [
            [1, 2],
            [1, 3],
            [3, 3],
            [11, 2],
            [11, 4],
            [15, 15],
            [16, 16],
            [22, 4],
        ]
        compute_section_coordinates(pixel_list, pivot_indexes=[3, 4], step=2)

    def test_find_source_images(self):
        edge = get_random_edge(self.exp, 0)
        pixel_list = edge.pixel_list(0)
        pixel_indexes = [len(pixel_list) // 2]
        find_source_images(
            compute_section_coordinates(pixel_list, pixel_indexes, step=2), self.exp, 0
        )

    def test_extract_section_profiles_for_edge(self):
        import random

        random.seed(13)
        edge = get_random_edge(self.exp, 0)
        extract_section_profiles_for_edge(self.exp, 0, edge)
