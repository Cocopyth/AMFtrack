import unittest
import numpy as np
from amftrack.util.plot import pixel_list_to_matrix


class TestPlot(unittest.TestCase):
    def test_pixel_list_to_matrix(self):
        pixel_list = [[0, 0], [3, 0], [2, 1], [1, 3], [3, 5]]
        M = np.array(
            [
                [1, 0, 0, 0, 0, 0],
                [0, 0, 0, 1, 0, 0],
                [0, 1, 0, 0, 0, 0],
                [1, 0, 0, 0, 0, 1],
            ]
        )
        M_ = pixel_list_to_matrix(pixel_list, 0)

        self.assertSequenceEqual(M_.shape, (4, 6))
        np.testing.assert_array_almost_equal(M, M_)
        self.assertTrue((M == M_).all())
        self.assertSequenceEqual(
            pixel_list_to_matrix(pixel_list, margin=1).shape, (6, 8)
        )
