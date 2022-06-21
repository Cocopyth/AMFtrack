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
)
from amftrack.util.sys import test_path
from helper import make_image


class TestTransformations(unittest.TestCase):
    def test_find_transformation(self):
        R, t = find_transformation(
            [[16420, 26260], [17120, 28480]], [[15760, 26500], [16420, 28780]]
        )
        f = get_transformation(R, t)
        f([16420, 26260])

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
        self.assertTrue(is_in_image(0, 0, 100, 100))
        self.assertFalse(is_in_image(0, 100, -1000, -1000))

    def test_find_image_index(self):
        self.assertEqual(
            find_image_index(
                [[0, 0], [10_000, 10_000], [9500, 9500], [0, 0]], 11000, 9900
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
