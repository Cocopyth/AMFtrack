import unittest
import os
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from amftrack.util.image_analysis import (
    find_transformation,
    get_transformation,
    reverse_transformation,
    convert_to_micrometer,
    find_image_index,
    is_in_image,
    extract_inscribed_rotated_image,
    is_negative_similarity,
    find_similarity,
    find_scaling_factor,
)
from amftrack.util.sys import test_path
from helper import make_image
import numpy as np


class TestTransformations(unittest.TestCase):
    def test_find_transformation(self):
        old_coordinates = [
            [16420, 6260],
            [1120, 28480],
            [15760, 26500],
            [16420, 2780],
            [100, 1200],
            [-560, -10000],
        ]
        t = np.array([1200, -230])
        R = np.array(
            [
                [np.cos(np.pi / 15), -np.sin(np.pi / 15)],
                [np.sin(np.pi / 15), np.cos(np.pi / 15)],
            ]
        )
        f = get_transformation(R, t)
        new_coordinates = [f(c) for c in old_coordinates]
        R_, t_ = find_transformation(old_coordinates, new_coordinates)
        f = get_transformation(R_, t_)
        result_coordinates = [f(c) for c in old_coordinates]

        self.assertTrue(np.all((np.round(R - R_) == 0)))
        self.assertTrue(np.all((np.round(t - t_) == 0)))
        self.assertTrue(
            np.all(
                (
                    np.round(np.array(result_coordinates) - np.array(new_coordinates))
                    == 0
                )
            )
        )

    def test_reverse_transformation(self):
        R, t = find_transformation(
            [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
        )
        R_, t_ = reverse_transformation(*reverse_transformation(R, t))
        self.assertAlmostEqual(float(t[0]), float(t_[0]), 5)
        self.assertAlmostEqual(R[0][1], R_[0][1], 5)

    def test_convet(self):
        assert convert_to_micrometer(10) == 17.25

    def test_is_in_image(self):
        self.assertTrue(is_in_image(0, 0, 100, 100, 3000, 4096))
        self.assertFalse(is_in_image(0, 100, -1000, -1000, 3000, 4096))

    def test_find_image_index(self):
        self.assertEqual(
            find_image_index(
                [[0, 0], [10_000, 10_000], [9500, 9500], [0, 0]],
                11000,
                9900,
                3000,
                4096,
            ),
            2,
        )

    def test_extract_inscribed_rotated_image_1(self):

        image = make_image()

        for angle in [0, 1, 3, 5, 10, 87]:
            new_image = extract_inscribed_rotated_image(image, angle)
            im_pil = Image.fromarray(new_image, mode="RGB")
            im_pil.save(os.path.join(test_path, f"rotated{angle}.png"))

    def test_extract_inscribed_rotated_image_2(self):

        image = make_image()

        image = image[:200, :200, :]
        for angle in [30, 40, 50, 60, 80]:
            new_image = extract_inscribed_rotated_image(image, angle)
            im_pil = Image.fromarray(new_image, mode="RGB")
            im_pil.save(os.path.join(test_path, f"rotated{angle}_square.png"))

    def test_extract_inscribed_rotated_image_3(self):

        image = make_image()

        image = image[:300, :100, :]
        with self.assertRaises(Exception):
            new_image = extract_inscribed_rotated_image(image, 40)

    def test_extract_inscribed_rotated_image_4(self):
        # negative values
        image = make_image()

        for angle in [0, -1, -3, -5]:
            new_image = extract_inscribed_rotated_image(image, angle)
            im_pil = Image.fromarray(new_image, mode="RGB")
            im_pil.save(os.path.join(test_path, f"rotated{angle}.png"))

    def test_is_negative_similarity(self):
        old_coordinates = [
            [16420, 26260],
            [17120, 28480],
            [1420, 2620],
            [100, 10000],
        ]
        new_coordinates_1 = [[15760, 26500], [16420, 28780], [1330, 2530], [-165, 9880]]
        new_coordinates_2 = [[x[1], x[0]] for x in old_coordinates]
        new_coordinates_3 = [[x[1], x[0]] for x in new_coordinates_1]

        self.assertFalse(is_negative_similarity(old_coordinates, new_coordinates_1))
        self.assertTrue(is_negative_similarity(old_coordinates, new_coordinates_2))
        self.assertTrue(is_negative_similarity(old_coordinates, new_coordinates_3))

    def test_find_similarity(self):
        old_coordinates = [
            [16420, 26260],
            [17120, 28480],
            [1420, 2620],
            [100, 10000],
        ]
        t = np.array([1200, -230])
        R = np.array(
            [
                [np.cos(np.pi / 15), -np.sin(np.pi / 15)],
                [np.sin(np.pi / 15), np.cos(np.pi / 15)],
            ]
        )
        f = get_transformation(R, t)
        new_coordinates_1 = [f(np.array(c)) for c in old_coordinates]
        new_coordinates_2 = [[x[1], x[0]] for x in new_coordinates_1]
        new_coordinates_3 = [np.array(c) / 1.6 for c in new_coordinates_2]

        f1 = find_similarity(old_coordinates, new_coordinates_1)
        f2 = find_similarity(old_coordinates, new_coordinates_2)
        f3 = find_similarity(old_coordinates, new_coordinates_3)

        result1 = np.array([np.round(f1(c)) for c in old_coordinates])
        result2 = np.array([np.round(f2(c)) for c in old_coordinates])
        result3 = np.array([np.round(f3(c)) for c in old_coordinates])

        self.assertTrue(
            np.all((np.round(result1 - np.array(new_coordinates_1), decimals=-1) == 0))
        )
        self.assertTrue(
            np.all((np.round(result2 - np.array(new_coordinates_2), decimals=-1) == 0))
        )
        self.assertTrue(
            np.all((np.round(result3 - np.array(new_coordinates_3), decimals=-1) == 0))
        )
