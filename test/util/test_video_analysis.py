import unittest
import numpy as np
from test.util import helper
from amftrack.util.video_analysis import extract_kymograph
from numpy.testing import assert_array_equal


class TestVideoAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not helper.has_video():
            helper.make_video()

    def test_extract_kymograph_coordinates(self):

        # The goal of this test is mostly to test that the system coordinate is the right one

        k = extract_kymograph(helper.video_path, 1, 5, 1, 9, validation_fun=None)

        result = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ]
        )
        assert_array_equal(k, result)
