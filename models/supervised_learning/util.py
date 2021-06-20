import numpy as np


def sigmoid(z):
    """Computes the sigmoid function on a 1D NumPy array."""
    return 1.0 / (1 + np.exp(-z))


def add_intercept(x):
    """Add intercept to matrix x.
    
    Args:
        x: 2D NumPy array.
    
    Returns:
        new_x: New matrix, same as x, with 1's in the 0th column.
    """
    m, n = x.shape
    new_x = np.ones((m, n+1), dtype=x.dtype)
    new_x[:, 1:] = x

    return new_x
