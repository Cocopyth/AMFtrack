import unittest
import numpy as np
import os
from amftrack.util.sys import test_path
from amftrack.util.video_analysis import extract_kymograph
from test.helper import has_video
from PIL import Image
import cv2
from numpy.testing import assert_array_equal


class TestVideoAnalysis(unittest.TestCase):
    # @unittest.skipUnless(has_video(), "No video is present to run tests on video")
    def test_extract_kymograph_coordinates(self):
        # The goal of this test is mostly to test that the system coordinate is the right one
        im1 = np.triu(np.ones((10, 10)) * 100)
        im2 = np.triu(np.ones((10, 10)) * 10)

        # Move to util
        video_path = os.path.join(test_path, "video")
        if not os.path.isdir(video_path):
            os.mkdir(video_path)
        for i in range(2):
            cv2.imwrite(os.path.join(video_path, f"image{i}.png"), im1)
        cv2.imwrite(os.path.join(video_path, f"image{2}.png"), im2)

        k = extract_kymograph(video_path, 1, 5, 1, 9, validation_fun=None)

        result = np.array(
            [
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [100.0, 100.0, 100.0, 100.0, 100.0],
                [10.0, 10.0, 10.0, 10.0, 10.0],
            ]
        )
        assert_array_equal(k, result)
