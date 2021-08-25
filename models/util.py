import numpy as np


def sigmoid(z):
    """Computes the sigmoid function on a 1D NumPy array."""
    return 1.0 / (1 + np.exp(-z))


def add_intercept(X):
    """Add intercept to a 2D NumPy array.
    
    Args:
        X: 2D NumPy array.
    
    Returns:
        new_x: New 2D NumPy array, same as X, with 1's in the 0th column.
    """
    m, n = X.shape
    new_x = np.ones((m, n+1), dtype=X.dtype)
    new_x[:, 1:] = X

    return new_x