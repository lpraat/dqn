import numpy as np


def one_hot(x, num_labels):
    return np.squeeze(np.eye(num_labels)[x.reshape(-1)])
