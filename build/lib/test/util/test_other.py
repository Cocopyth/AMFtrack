import unittest
import numpy as np
from amftrack.util.other import is_in


class TestOther(unittest.TestCase):
    def test_is_in(self):
        l = [np.array([1.2, 3.2]), np.array([2.1, 0.98]), np.array([2.0, 0.8])]
        arr1 = np.array([1.2, 3.2])
        arr2 = np.array([1.2, 10.2])

        self.assertTrue(is_in(arr1, l))
        self.assertFalse(is_in(arr2, l))
        self.assertTrue(is_in(list(arr1), l))
        self.assertFalse(is_in(list(arr2), l))
