class Memory:

    def __init__(self, size, state_size):
        self.size = size
        self.state_size = state_size
        self.added_samples = 0

    def add_sample(self, sample):
        raise NotImplementedError

    def sample_batch(self, batch_size):
        raise NotImplementedError
