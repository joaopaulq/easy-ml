import numpy as np


def sigmoid(z):
    """Computes the sigmoid function on a 1D NumPy array."""
    return 1.0 / (1 + np.exp(-z))