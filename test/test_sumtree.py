import unittest
import numpy as np

from src.dqn.sumtree import SumTree


class TestSumTree(unittest.TestCase):

    def test_sum_tree(self):
        t = SumTree(4)
        print(t.nodes)
        print(t.data)

        t.add(4, "A")
        t.add(44, "B")
        t.add(52, "C")
        t.add(2, "D")

        np.testing.assert_array_equal(t.data, ["A", "B", "C", "D"])
        np.testing.assert_array_almost_equal(t.nodes, [102, 48, 54, 4, 44, 52, 2])

        t.update(0, 12)
        np.testing.assert_array_almost_equal(t.nodes, [110, 56, 54, 12, 44, 52, 2])

        print(t.nodes)
        print(t.data)

