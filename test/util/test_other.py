import unittest
from amftrack.util.other import expand_segment, get_section_segment, compute_factor


class TestSegment(unittest.TestCase):
    def test_expand_segment(self):
        point1 = [0, 0]
        point2 = [2, 4]
        self.assertEqual(expand_segment(point1, point2, 0.5), ([0.5, 1.0], [1.5, 3.0]))
        assert expand_segment(point1, point2, 2) == ([-1.0, -2.0], [3.0, 6.0])
        assert expand_segment(point2, point1, 2) == ([3.0, 6.0], [-1.0, -2.0])

    def test_compute_factor(self):
        assert compute_factor([0, 0], [0, 3], 6) == 2
        assert compute_factor([1, 0], [10, 0], 4.5) == 0.5

    def test_get_section_segment(self):
        get_section_segment([-2, 2], [1, 1], 2.82)
