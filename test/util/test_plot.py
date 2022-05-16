import unittest
import numpy as np
from numpy.testing import assert_array_equal
from amftrack.util.plot import pixel_list_to_matrix, crop_image


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

    def test_crop_image(self):
        a = np.array(
            [
                [1.0, 2.3, 12.0, 1.0],
                [4.5, 2.3, 0.0, 12.0],
                [2.3, 0.3, 5.5, 8.9],
                [2.3, 0.3, 5.5, 8.9],
                [2.3, 0.3, 5.5, 8.9],
            ]
        )

        assert_array_equal(
            crop_image(a, [[0, 0], [2, 3]]),
            np.array([[1.0, 2.3, 12.0], [4.5, 2.3, 0.0]]),
        )

        assert_array_equal(
            crop_image(a, [[0, 0], [10, 10]]),
            a,
        )

        assert_array_equal(
            crop_image(a, [[-1, -1], [2, 2]]),
            np.array([[1.0, 2.3], [4.5, 2.3]]),
        )
