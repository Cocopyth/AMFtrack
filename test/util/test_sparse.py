import unittest
import numpy as np
from numpy.testing import assert_array_equal
from amftrack.util.plot import pixel_list_to_matrix, crop_image
from amftrack.util.sparse import dilate_coord_list


class TestSparse(unittest.TestCase):
    def test_pixel_list_to_matrix(self):
        pixel_list = [[3, 0]]
        m1 = dilate_coord_list(pixel_list)
        m2 = dilate_coord_list(pixel_list, iteration=2)
