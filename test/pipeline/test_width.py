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
    extract_section_profiles_for_edge,
    find_source_images_filtered,
)
from amftrack.pipeline.functions.image_processing.experiment_util import get_random_edge

from test import helper


class TestWidthLight(unittest.TestCase):
    def test_find_source_images_filtered(self):
        sec1 = [[1500, 1500], [1500, 1600]]  # in image 1
        sec2 = [[1500, 1500], [4000, 1500]]  # in image 1 and partially in 3
        sec3 = [[4000, 1500], [4500, 1500]]  # in image 3 and partially in 1
        sec4 = [[5000, 1500], [0, 1500]]  # in no image
        image1 = [0, 0]
        image2 = [10000, 10000]
        image3 = [3900, 0]

        im_indexes, sections = find_source_images_filtered(
            [sec1, sec2, sec3, sec4], [image1, image2, image3]
        )
        self.assertListEqual(im_indexes, [0, 0, 2])
        self.assertListEqual(sections[2][0], [100, 1500])

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

    def test_extract_section_profiles_for_edge(self):
        import random

        random.seed(13)
        edge = get_random_edge(self.exp, 0)
        extract_section_profiles_for_edge(self.exp, 0, edge)
