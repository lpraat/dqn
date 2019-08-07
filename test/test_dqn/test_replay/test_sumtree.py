import unittest

import numpy as np

from src.dqn.replay.sum_tree import SumTree


class TestSumTree(unittest.TestCase):

    def test_sum_tree(self):
        t = SumTree(4)

        t.add(4, "A")
        t.add(44, "B")
        t.add(52, "C")
        t.add(2, "D")

        np.testing.assert_array_equal(t.data, ["A", "B", "C", "D"])
        np.testing.assert_array_almost_equal(t.nodes, [102, 48, 54, 4, 44, 52, 2])
        self.assertAlmostEqual(t.get_total_sum(), 102)
        self.assertAlmostEqual(t.min_value, 2)
        self.assertAlmostEqual(max(t.nodes[t.size // 2 + 1:]), 52)

        t.update(0, 12)
        np.testing.assert_array_almost_equal(t.nodes, [110, 56, 54, 12, 44, 52, 2])
        self.assertAlmostEqual(t.get_total_sum(), 110)

        t.add(4, "A")
        np.testing.assert_array_almost_equal(t.nodes, [102, 48, 54, 4, 44, 52, 2])

        data_index, value, data = t.get(50)
        self.assertEqual(data_index, 2)
        self.assertAlmostEqual(value, 52)
        self.assertEqual(data, "C")

        data_index, value, data = t.get(101)
        self.assertEqual(data_index, 3)
        self.assertAlmostEqual(value, 2)
        self.assertEqual(data, "D")

        t.add(1, "B")
        t.add(100, "C")
        self.assertAlmostEqual(t.min_value, 1)
        self.assertAlmostEqual(max(t.nodes[t.size // 2 + 1:]), 100)
