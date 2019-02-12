import numpy as np


class SumTree:
    def __init__(self, size):
        self.size = size
        self.nodes = np.zeros(2 * size - 1)
        self.data = np.empty(size, dtype=np.object)
        self.curr_index = 0

    def add(self, value, data):
        self.data[self.curr_index] = data
        self.update(self.curr_index, value)
        self.curr_index = (self.curr_index + 1) % self.size

    def update(self, data_index, value):
        node_index = data_index + self.size - 1
        diff = self.nodes[node_index] - value
        self.nodes[node_index] = value

        while node_index != 0:
            node_index = (node_index - 1) // 2
            self.nodes[node_index] -= diff






