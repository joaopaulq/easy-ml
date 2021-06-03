import numpy as np


def sigmoid(z):
    """Computa a função sigmoid de um array NumPy 1D."""
    # *** START CODE HERE ***
    return 1 / (1 + np.exp(-z))
    # *** END CODE HERE ***
    

def add_intercept(x):
    """Adiciona o termo de interseção a matriz x.

    Args:
        x: Um array NumPy 2D.
    
    Returns:
        new_x: Uma nova matriz, semelhante a x, com 1's na coluna 0.
    """
    m, n = x.shape
    new_x = np.zeros((m, n+1), dtype=x.dtype)
    new_x[:, 0] = 1
    new_x[:, 1:] = x

    return new_x