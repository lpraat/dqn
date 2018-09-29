import unittest
import numpy as np

from src.utils import one_hot


class TestUtils(unittest.TestCase):
    def test_one_hot(self):
        a = np.array([[0], [3], [2], [4]])
        np.testing.assert_array_equal(one_hot(a, 5), np.array([[1, 0, 0, 0, 0],
                                                               [0, 0, 0, 1, 0],
                                                               [0, 0, 1, 0, 0],
                                                               [0, 0, 0, 0, 1]]))
