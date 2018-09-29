import numpy as np


class ReplayMemory:
    def __init__(self, size, dims):
        self.max_size = size
        self.curr_index = 0
        self.dims = dims
        self.buffer = []

    def add_sample(self, sample):

        if len(self.buffer) < self.max_size:
            self.buffer.append(sample)
        else:
            self.buffer[self.curr_index] = sample
            self.curr_index = (self.curr_index + 1) % self.max_size

    def sample_batch(self, batch_size):
        assert batch_size <= len(self.buffer), "Batch size is greater than buffer length"
        curr_len = len(self.buffer)

        states = np.empty((batch_size, self.dims[0]))
        actions = np.empty((batch_size, self.dims[1]), dtype=np.int32)
        rewards = np.empty((batch_size, self.dims[2]))
        next_states = np.empty((batch_size, self.dims[3]))
        ends = np.empty((batch_size, self.dims[4]), dtype=np.int32)

        for i in range(batch_size):
            sample = self.buffer[np.random.randint(curr_len)].reshape(5)
            states[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_states[i] = sample[3]
            ends[i] = sample[4]

        return states, actions, rewards, next_states, ends