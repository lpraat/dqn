import unittest
import tensorflow as tf
import numpy as np

from src.dqn.graph import new_targets_graph


class TestTargetGraphs(unittest.TestCase):

    def test_new_targets(self):

        actions = np.array([
            [0],
            [0],
            [1]]
        )

        preds_next = np.array([
            [89.307625, 79.39642],
            [47.669586, 43.58093],
            [74.52588, 66.5371]
        ])

        preds_t = np.array([
            [88.30711,  76.472015],
            [45.066994, 40.439713],
            [72.97738, 63.52341]
        ])

        rewards = np.array([
            [1.],
            [1.],
            [1.]
        ])

        ends = np.array([
            [0],
            [0],
            [0]
        ])

        gamma = 0.999

        new_targets = np.array([
            [89.21881,   0.],
            [46.021927,  0.],
            [0., 73.9044],
        ], dtype=np.float32)

        g_targets = new_targets_graph(new_targets.shape[0], new_targets.shape[1])

        with tf.Session() as sess:
            built_targets = sess.run(g_targets.targets, feed_dict={
                g_targets.actions: actions,
                g_targets.preds_next: preds_next,
                g_targets.preds_t: preds_t,
                g_targets.rewards: rewards,
                g_targets.ends: ends,
                g_targets.gamma: gamma
            })
            np.testing.assert_array_equal(new_targets, built_targets)



