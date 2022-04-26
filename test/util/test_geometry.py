import numpy as np
import unittest
from amftrack.util.geometry import (
    expand_segment,
    get_section_segment,
    compute_factor,
    generate_index_along_sequence,
    distance_point_pixel_line,
    get_closest_lines,
    get_closest_line_opt,
)


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

    def test_generate_index_along_sequence(self):
        self.assertEqual(
            generate_index_along_sequence(20, resolution=3, offset=4), [4, 7, 10, 13]
        )
        self.assertEqual(
            generate_index_along_sequence(21, resolution=3, offset=4),
            [4, 7, 10, 13, 16],
        )
        self.assertEqual(
            generate_index_along_sequence(6, resolution=1, offset=2), [2, 3]
        )
        self.assertEqual(generate_index_along_sequence(6, resolution=2, offset=10), [3])
        self.assertEqual(
            generate_index_along_sequence(7, resolution=1, offset=0),
            [0, 1, 2, 3, 4, 5, 6],
        )
        self.assertEqual(generate_index_along_sequence(7, resolution=10, offset=0), [0])
        self.assertEqual(
            generate_index_along_sequence(7, resolution=6, offset=0), [0, 6]
        )

    def test_distance_point_pixel_line(self):

        line = [[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]]
        self.assertEqual(
            distance_point_pixel_line([2, 3], line, step=1),
            0,
        )

        self.assertEqual(
            distance_point_pixel_line([2, 1], line, step=1),
            2.0,
        )

        self.assertEqual(
            distance_point_pixel_line([0, 3], line, step=1),
            2.0,
        )

        self.assertAlmostEqual(
            distance_point_pixel_line([2, 5], line, step=1), np.sqrt(2), 4
        )

        # distance increases with increassing step
        self.assertTrue(
            distance_point_pixel_line([0, 9], line, step=2)
            > distance_point_pixel_line([0, 9], line, step=1)
        )  # when the closest point isn't taken

    def test_get_closest_lines(self):
        line1 = [[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]]
        line2 = [[4, 4], [3, 4], [3, 3], [4, 3], [5, 3], [5, 4], [6, 4]]
        line3 = [
            [4, 4],
            [3, 4],
            [3, 3],
            [4, 3],
            [5, 3],
            [5, 4],
            [6, 4],
            [7, 5],
            [7, 6],
            [7, 7],
        ]
        ind, d = get_closest_lines(
            [7, 10], lines=[line1, line2, line3], step=3, n_nearest=2
        )
        self.assertSequenceEqual(ind, [0, 2])

        ind, d = get_closest_lines(
            [7, 10], lines=[line1, line2, line3], step=3, n_nearest=3
        )
        self.assertSequenceEqual(ind, [0, 2, 1])
        self.assertSequenceEqual(
            d, [3.0, 3.0, np.linalg.norm(np.array([6, 4]) - np.array([7, 10]))]
        )

        # More n_nearest than lines
        ind, d = get_closest_lines(
            [7, 10], lines=[line1, line2, line3], step=1, n_nearest=4
        )
        self.assertSequenceEqual(ind, [0, 2, 1])
        self.assertSequenceEqual(
            d, [3.0, 3.0, np.linalg.norm(np.array([6, 4]) - np.array([7, 10]))]
        )

        # A very big step
        ind, d = get_closest_lines(
            [7, 10], lines=[line1, line2, line3], step=10, n_nearest=4
        )

    def test_get_closest_line_opt(self):

        line1 = [[2, 3], [3, 3], [3, 4], [4, 5], [5, 5], [6, 6], [7, 7]]
        line2 = [[4, 4], [3, 4], [3, 3], [4, 3], [5, 3], [5, 4], [6, 4]]
        line3 = [
            [4, 4],
            [3, 4],
            [3, 3],
            [4, 3],
            [5, 3],
            [5, 4],
            [6, 4],
            [7, 5],
            [7, 6],
            [8, 7],
        ]
        ind, d = get_closest_line_opt([7, 10], lines=[line1, line2, line3], step=1000)
        self.assertEqual(ind, 0)
        self.assertEqual(d, float(np.linalg.norm(np.array([7, 7]) - np.array([7, 10]))))

        ind, d = get_closest_line_opt([7, 10], lines=[line1, line2, line3], step=1)
        self.assertEqual(ind, 0)
        self.assertEqual(d, float(np.linalg.norm(np.array([7, 7]) - np.array([7, 10]))))
