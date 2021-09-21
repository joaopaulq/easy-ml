import numpy as np


def sigmoid(z):
    """Computes the sigmoid function on a NumPy array."""
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


def dist(x, y, measure='e'):
    """Computes the distance between two NumPy arrays.

    Args:
        x: A NumPy array.
        y: A NumPy array.
        measure: Distance measure.

    Returns:
        Distance between x and y using the given measure. Float.
    """
    if measure == 'e':
        return np.linalg.norm(x - y) # Euclidian distance.
    elif measure == 'm':
        return np.sum(np.abs(x - y)) # Manhattan distance.
    else:
        raise NotImplementedError(f'Measure {measure} not implemented')