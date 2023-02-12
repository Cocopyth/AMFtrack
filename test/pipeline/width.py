import unittest
from amftrack.pipeline.functions.image_processing.extract_width_fun import (
    compute_section_coordinates,
    extract_section_profiles_for_edge_exp,
    find_source_images_filtered,
    get_width_info_new,
)
from amftrack.pipeline.functions.image_processing.experiment_util import get_random_edge

from test.util import helper


class TestWidthLight(unittest.TestCase):
    def test_find_source_images_filtered(self):
        sec1 = [[1500, 1500], [1600, 1500]]  # in image 1
        sec2 = [[1500, 1500], [1500, 4000]]  # in image 1 and partially in 3
        sec3 = [[1500, 4000], [1500, 4500]]  # in image 3 and partially in 1
        sec4 = [[1500, 5000], [1500, 0]]  # in no image
        image1 = [0, 0]
        image2 = [10000, 10000]
        image3 = [0, 3900]

        im_indexes, sections = find_source_images_filtered(
            [sec1, sec2, sec3, sec4], [image1, image2, image3], 3000, 4096
        )
        self.assertListEqual(im_indexes, [0, 0, 2])
        self.assertListEqual(sections[2][0], [1500, 100])

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
        cls.exp = helper.make_experiment_object()

    def test_extract_section_profiles_for_edge(self):
        import random

        random.seed(13)
        edge = get_random_edge(self.exp, 0)
        extract_section_profiles_for_edge_exp(self.exp, 0, edge)

    def test_extract_width_exp(self):
        exp = self.exp
        get_width_info_new(exp, 2, resolution=50)
