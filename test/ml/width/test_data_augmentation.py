from amftrack.ml.width.data_augmentation import *
import tensorflow as tf
import numpy as np
import unittest
from numpy.testing import assert_array_equal


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

    def test_random_invert(self):
        layer = random_invert(1)
        a_ = layer(self.a)

        assert_array_equal(
            np.array(a_), np.array([[221.0, 220.0, 205.0], [243.0, 244.0, 245.0]])
        )

    def test_random_mirror(self):
        layer = random_mirror(1)
        a_ = layer(self.a)
        b_ = layer(self.b)

        assert_array_equal(
            np.array(a_),
            np.array([[50.0, 35, 34], [10, 11, 12]]),
        )

        assert_array_equal(
            np.array(b_),
            np.array(
                [
                    [[50, 35, 34], [10, 11, 12]],
                    [[5, 3, 3], [1, 1, 1]],
                ]
            ),
        )
