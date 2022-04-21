import os
import unittest

from test import helper


class TestNew(unittest.TestCase):
    @unittest.skipUnless(
        helper.has_test_plate(), "No plate to run construct experiment"
    )
    def test_nnn(self):
        point1 = [0, 0]
        point2 = [2, 4]
        assert 1 == 1
