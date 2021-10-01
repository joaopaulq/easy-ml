import numpy as np


def least_squared(y, h_x):
    return np.sum((y - h_x)**2)


def cross_entropy(y, h_x):
    return np.mean(y*np.log(h_x) + (1-y)*np.log(1-h_x))


def zero_one(y, h_x):
    return np.sum(y != h_x)
