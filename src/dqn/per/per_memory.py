import numpy as np

from src.dqn.per.sum_tree import SumTree


class PERMemory:
    def __init__(self,
                 size,
                 state_size,
                 alpha=0.4,
                 beta=0.7,
                 epsilon=0.001,
                 beta_grow=lambda beta, train_step: beta + 1 / 100000,
                 max_priority=1.):
        self.alpha = alpha
        self.beta = beta
        self.epsilon = epsilon
        self.beta_grow = beta_grow
        self.max_priority = max_priority
        self.size = size
        self.added_samples = 0
        self.sum_tree = SumTree(size)
        self.state_size = state_size
        self.train_step = 0

    def add_sample(self, sample):
        self.sum_tree.add(self.sum_tree.max_value if self.sum_tree.max_value is not None else self.max_priority, sample)
        self.added_samples += 1

    def sample_batch(self, batch_size):
        states = np.empty((batch_size, self.state_size))
        actions = np.empty((batch_size, 1), dtype=np.int32)
        rewards = np.empty((batch_size, 1))
        next_states = np.empty((batch_size, self.state_size))
        ends = np.empty((batch_size, 1), dtype=np.int32)
        is_weights = np.empty((batch_size, 1))
        node_indices = np.empty((batch_size,), dtype=np.int32)

        range_size = self.sum_tree.get_total_sum() / batch_size
        max_is_weight = (self.size * (self.sum_tree.min_value / self.sum_tree.get_total_sum())) ** -self.beta

        for i in range(batch_size):
            sample_value = np.random.uniform(range_size * i, range_size * (i + 1))
            data_index, value, data = self.sum_tree.get(sample_value)

            states[i] = data[0]
            actions[i] = data[1]
            rewards[i] = data[2]
            next_states[i] = data[3]
            ends[i] = data[4]
            is_weights[i] = ((self.size * (value / self.sum_tree.get_total_sum())) ** -self.beta) / max_is_weight
            node_indices[i] = data_index

        self.train_step += 1
        self.beta = min(1, self.beta_grow(self.beta, self.train_step))
        if (self.train_step % 1000) == 0:
            print(self.beta)

        return states, actions, rewards, next_states, ends, is_weights, node_indices

    def update_priorities(self, node_indices, abs_td_errors):
        abs_td_errors = np.minimum(abs_td_errors + self.epsilon, self.max_priority)
        updated_priorities = np.power(abs_td_errors, self.alpha)

        for i in range(len(updated_priorities)):
            self.sum_tree.update(node_indices[i], updated_priorities[i])

