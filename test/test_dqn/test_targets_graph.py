import unittest

import numpy as np


from src.dqn.dqn import DQN
import gym


class TestTargetGraphs(unittest.TestCase):

    def test_new_targets(self):
        mock_env = gym.make('CartPole-v0')
        dqn = DQN(env=mock_env, mini_batch_size=3, gamma=0.999)
        dqn.num_actions = 2

        actions = np.array([
            [0],
            [0],
            [1]
        ], dtype=np.int)

        preds_next = np.array([
            [89.307625, 79.39642],
            [47.669586, 43.58093],
            [74.52588, 66.5371]
        ], dtype=np.float32)

        preds_t = np.array([
            [88.30711,  76.472015],
            [45.066994, 40.439713],
            [72.97738, 63.52341]
        ], dtype=np.float32)

        rewards = np.array([
            [1.],
            [1.],
            [1.]
        ], dtype=np.float32)

        ends = np.array([
            [0],
            [0],
            [0]
        ], dtype=np.float32)

        new_targets = np.array([
            [89.21881,   0.],
            [46.021927,  0.],
            [0., 73.9044],
        ], dtype=np.float32)

        built_targets = dqn.get_targets(actions, preds_next, preds_t, rewards, ends)
        np.testing.assert_array_equal(new_targets, built_targets)



