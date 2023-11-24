from amftrack.ml.width.data_augmentation import *
import tensorflow as tf
import numpy as np
import unittest
from numpy.testing import assert_array_equal, assert_raises


class TestDataAugmentation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):

        cls.a = tf.constant([[34, 35, 50], [12, 11, 10]], dtype=tf.float32)
        cls.b = tf.constant(
            [
                [[34, 35, 50], [12, 11, 10]],
                [[3, 3, 5], [1, 1, 1]],
            ],
            dtype=tf.float32,
        )
        cls.c = tf.constant(
            [[[34.0], [35.0], [50.0]], [[12.0], [11.0], [10.0]]], dtype=tf.float32
        )
        data_point = np.ones((2, 120, 1))
        data_point[0, 50, 0] = 2.0
        cls.d = tf.constant(data_point, dtype=tf.float32)

    def test_random_invert(self):
        layer = random_invert(1)
        a_ = layer(self.a)

        assert_array_equal(
            np.array(a_), np.array([[221.0, 220.0, 205.0], [243.0, 244.0, 245.0]])
        )

    def test_random_mirror(self):
        layer = random_mirror(1)
        c_ = layer(self.c)

        assert_array_equal(
            np.array(c_),
            np.array([[[50.0], [35.0], [34.0]], [[10.0], [11.0], [12.0]]]),
        )

    def test_random_brightness(self):
        # checking that a different seed is used at each call
        layer = random_brightness(10)
        c_1 = layer(self.c)
        c_2 = layer(self.c)
        assert_raises(AssertionError, assert_array_equal, np.array(c_1), np.array(c_2))

    def test_random_crop(self):
        # test that a random seed is used each time
        layer = random_crop(original_size=120, input_size=80)
        c_1 = layer(self.d)
        c_2 = layer(self.d)
        assert_raises(AssertionError, assert_array_equal, np.array(c_1), np.array(c_2))

    def test_augmentation_pipeline(self):
        data_augmentation(self.d)


# class TestDataAugmentationOnDataset(unittest.TestCase):
#     a = 1
