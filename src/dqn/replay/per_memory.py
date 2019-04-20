import numpy as np

from src.dqn.replay.memory import Memory
from src.dqn.replay.sum_tree import SumTree


class PERMemory(Memory):
    def __init__(self, size, state_size, alpha, beta, epsilon, beta_grow):
        super().__init__(size, state_size)
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_grow = beta_grow
        self.max_priority = 1.0
        self.sum_tree = SumTree(self.size)
        self.train_step = 0

    def add_sample(self, sample):
        # New transitions arrive without a known TD-error
        # They are added with maximal priority to guarantee that they are seen and their TD-error(priority) is updated
        self.sum_tree.add(self.max_priority ** self.alpha, sample)
        self.added_samples += 1

    def sample_batch(self, batch_size):
        states = np.empty((batch_size, self.state_size))
        actions = np.empty((batch_size, 1), dtype=np.int32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, self.state_size), dtype=np.float32)
        ends = np.empty((batch_size, 1), dtype=np.int32)
        is_weights = np.empty((batch_size, 1), dtype=np.float32)
        node_indices = np.empty((batch_size,), dtype=np.int32)

        tree_total_sum = self.sum_tree.get_total_sum()
        range_size = tree_total_sum / batch_size
        max_is_weight = np.power(self.size * (self.sum_tree.min_value / tree_total_sum), -self.beta)

        for i in range(batch_size):
            sample_value = np.random.uniform(range_size * i, range_size * (i + 1))
            data_index, value, data = self.sum_tree.get(sample_value)

            states[i] = data[0]
            actions[i] = data[1]
            rewards[i] = data[2]
            next_states[i] = data[3]
            ends[i] = data[4]
            is_weights[i] = (np.power(self.size * (value / tree_total_sum), -self.beta)) / max_is_weight
            node_indices[i] = data_index

        self.train_step += 1
        self.beta = min(1, self.beta_grow(self.beta, self.train_step))

        return states, actions, rewards, next_states, ends, is_weights, node_indices

    def update_priorities(self, node_indices, td_errors):
        updated_priorities = np.power(np.minimum(np.abs(td_errors) + self.epsilon, self.max_priority), self.alpha)

        for i in range(len(updated_priorities)):
            self.sum_tree.update(node_indices[i], updated_priorities[i])
