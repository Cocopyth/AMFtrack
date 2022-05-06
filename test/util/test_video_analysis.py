import unittest
import os
from amftrack.util.sys import test_path
from amftrack.util.video_analysis import extract_kymograph
from test.helper import has_video


@unittest.skipUnless(has_video(), "No video is present to run tests on video")
class TestVideoAnalysis(unittest.TestCase):
    def test_extract_kymograph(self):
        path = os.path.join(test_path, "video")

        def f(name):
            return name != "Felix_03_Image__2022-03-24__11-54-57.tiff"

        k = extract_kymograph(path, 10, 10, 30, 10, validation_fun=f)
        self.assertSequenceEqual(k.shape, (5, 21))
