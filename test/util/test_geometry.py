import numpy as np
import unittest
from helper import is_equal_seq
from amftrack.util.geometry import (
    expand_segment,
    get_section_segment,
    compute_factor,
    generate_index_along_sequence,
    distance_point_pixel_line,
    get_closest_lines,
    get_closest_line_opt,
    intersect_rectangle,
    is_overlapping,
    get_overlap,
    format_region,
    is_in_bounding_box,
    get_bounding_box,
    centered_bounding_box,
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

    def test_is_overlapping(self):
        # Overlapping
        self.assertTrue(is_overlapping([1, 10], [5, 15]))
        self.assertTrue(is_overlapping([5, 15], [1, 10]))
        self.assertTrue(is_overlapping([1, 20], [5, 10]))
        self.assertTrue(is_overlapping([5, 10], [1, 20]))

        # Not overlapping
        self.assertFalse(is_overlapping([1, 10], [11, 20]))
        self.assertFalse(is_overlapping([11, 20], [1, 10]))

        # Adjacent segment
        self.assertTrue(is_overlapping([1, 2], [2, 3], strict=False))
        self.assertTrue(is_overlapping([2, 3], [1, 2], strict=False))
        self.assertFalse(is_overlapping([1, 2], [2, 3], strict=True))
        self.assertFalse(is_overlapping([2, 3], [1, 2], strict=True))

    def test_intersect_rectangle(self):
        # Interecting rectangles
        self.assertTrue(
            intersect_rectangle([5, 5], [10, 30], [7, 7], [9, 9])
        )  # contained
        self.assertTrue(
            intersect_rectangle([7, 7], [9, 9], [5, 5], [10, 30])
        )  # contained
        self.assertTrue(
            intersect_rectangle([7, 7], [40, 40], [5, 5], [10, 30])
        )  # partly intersecting

        # Not intersecting rectangles
        self.assertFalse(intersect_rectangle([31, 31], [40, 40], [5, 5], [10, 30]))
        self.assertFalse(intersect_rectangle([5, 5], [10, 30], [31, 31], [40, 40]))
        self.assertFalse(intersect_rectangle([5, 5], [10, 30], [31, 0], [40, 4]))

        # Limit cases
        self.assertFalse(
            intersect_rectangle([10, 30], [40, 40], [5, 5], [10, 30], strict=True)
        )  # touching only a corner
        self.assertFalse(
            intersect_rectangle([5, 5], [15, 15], [15, 7], [20, 10], strict=True)
        )  # touching only a side
        self.assertTrue(
            intersect_rectangle([10, 30], [40, 40], [5, 5], [10, 30], strict=False)
        )  # touching only a corner
        self.assertTrue(
            intersect_rectangle([5, 5], [15, 15], [15, 7], [20, 10], strict=False)
        )  # touching only a side

    def test_get_overlap(self):

        # Interecting rectangles
        self.assertTrue(
            is_equal_seq(
                get_overlap([5, 5], [10, 30], [7, 7], [9, 9]), [[7, 7], [9, 9]]
            )
        )  # contained
        self.assertTrue(
            is_equal_seq(
                get_overlap([7, 7], [9, 9], [5, 5], [10, 30]), [[7, 7], [9, 9]]
            )
        )  # contained
        self.assertTrue(
            is_equal_seq(
                get_overlap([7, 7], [40, 40], [5, 5], [10, 30]),
                [[7, 7], [10, 30]],
            )
        )  # partly intersecting

        self.assertTrue(
            is_equal_seq(
                get_overlap([10, 30], [40, 40], [5, 5], [10, 30], strict=False),
                [[10, 30], [10, 30]],
            )
        )  # touching only a corner
        self.assertTrue(
            is_equal_seq(
                get_overlap([5, 5], [15, 15], [15, 7], [20, 10], strict=False),
                [[15, 7], [15, 10]],
            )
        )  # touching only a side

    def test_format_region(self):
        self.assertTrue(is_equal_seq(format_region([[1, 2], [0, 3]]), [[0, 2], [1, 3]]))
        self.assertTrue(is_equal_seq(format_region([[1, 2], [2, 3]]), [[1, 2], [2, 3]]))

    def test_is_in_bounding_box(self):
        self.assertTrue(is_in_bounding_box([1, 2], [[0, 0], [10, 10]]))
        self.assertFalse(is_in_bounding_box([10, 20], [[0, 0], [10, 10]]))

    def test_get_bounding_box_1(self):
        self.assertTrue(
            is_equal_seq(get_bounding_box([[1, 2]], margin=30), [[-29, -28], [31, 32]])
        )

    def test_get_bounding_box_2(self):
        get_bounding_box([[1.3, 4.1], [1.3, 2.1]])

    def test_centered_bounding_box(self):
        region = centered_bounding_box([1, 2], size=100)
        self.assertTrue(is_equal_seq(region, [[-49, -48], [51, 52]]))
