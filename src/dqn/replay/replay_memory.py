import numpy as np

from src.dqn.replay.memory import Memory


class ReplayMemory(Memory):
    def __init__(self, size, state_size):
        super().__init__(size, state_size)
        self.curr_index = 0
        self.buffer = []

    def add_sample(self, sample):
        if len(self.buffer) < self.size:
            self.buffer.append(sample)
        else:
            self.buffer[self.curr_index] = sample
            self.curr_index = (self.curr_index + 1) % self.size
        self.added_samples += 1

    def sample_batch(self, batch_size):
        curr_len = len(self.buffer)

        states = np.empty((batch_size, self.state_size), dtype=np.float32)
        actions = np.empty((batch_size, 1), dtype=np.int32)
        rewards = np.empty((batch_size, 1), dtype=np.float32)
        next_states = np.empty((batch_size, self.state_size), dtype=np.float32)
        ends = np.empty((batch_size, 1), dtype=np.float32)

        for i in range(batch_size):
            sample = self.buffer[np.random.randint(curr_len)]
            states[i] = sample[0]
            actions[i] = sample[1]
            rewards[i] = sample[2]
            next_states[i] = sample[3]
            ends[i] = sample[4]

        return states, actions, rewards, next_states, ends
