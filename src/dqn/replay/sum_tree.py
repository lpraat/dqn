import numpy as np


class SumTree:
    def __init__(self, size):
        self.size = size
        self.nodes = np.zeros(2 * size - 1)
        self.data = np.empty(size, dtype=np.object)
        self.curr_index = 0
        self.min_value = None
        self.max_value = None

    def update_min_max(self, new_value):
        self.min_value = new_value if self.min_value is None else min(self.min_value, new_value)

    def add(self, value, data):
        self.data[self.curr_index] = data
        self.update(self.curr_index, value)
        self.curr_index = (self.curr_index + 1) % self.size

    def update(self, data_index, value):
        node_index = data_index + self.size - 1
        diff = self.nodes[node_index] - value
        self.nodes[node_index] = value

        self.update_min_max(value)

        while node_index != 0:
            node_index = (node_index - 1) // 2
            new_value = self.nodes[node_index] - diff
            self.nodes[node_index] = new_value

    def get(self, sample_value):
        node_index = 0

        while node_index < self.size - 1:
            left_child_index = node_index * 2 + 1
            right_child_index = left_child_index + 1

            if sample_value <= self.nodes[left_child_index]:
                node_index = left_child_index
            else:
                node_index = right_child_index
                sample_value = sample_value - self.nodes[left_child_index]

        data_index = node_index - self.size + 1

        return data_index, self.nodes[node_index], self.data[data_index]

    def get_total_sum(self):
        return self.nodes[0]

