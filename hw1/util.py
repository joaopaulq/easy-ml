import numpy as np


def sigmoid(z):
    """Computa a função sigmoid de um array NumPy 1D."""
    return 1 / (1 + np.exp(-z))
    

def add_intercept(x):
    """Adiciona o termo de interseção a matriz x.

    Args:
        x: Um array NumPy 2D. Dim (m, n).
    
    Returns:
        new_x: Uma nova matriz, semelhante a x, com 1's na col 0. Dim (m, n+1).
    """
    m, n = x.shape
    new_x = np.ones((m, n+1), dtype=x.dtype)
    new_x[:, 1:] = x

    return new_x
