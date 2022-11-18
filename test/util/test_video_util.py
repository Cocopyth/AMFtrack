import os
import unittest
from test.util import helper
from amftrack.util.sys import (
    test_path,
)
from amftrack.util.video_util import make_video_targeted


class TestExperimentHeavy(unittest.TestCase):
    """Tests that need a plate with multiple timesteps"""

    @classmethod
    def setUpClass(cls):
        cls.exp = helper.make_experiment_object_multi()

    def test_make_video_targeted(self):

        try:
            os.mkdir(os.path.join(test_path, "targeted"))
        except:
            pass

        make_video_targeted(
            self.exp,
            coordinate=[20667, 7079],
            directory_path=os.path.join(test_path, "targeted"),
            size=500,
        )
