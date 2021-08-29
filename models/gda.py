import numpy as np

from scipy.stats import multivariate_normal


class GDA:
    """Class for the Gaussian Discriminant Analysis model.
    
    Attributes:
        phi: Proportion of examples from class 1. Float.
        mu_0: Mean for class 0. NumPy array.
        mu_1: Mean for class 1. NumPy array.
        sigma: Covariance matrix. NumPy array.

    Example of usage:
        > clf = GDA()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """
    
    def __init__(self):
        self.phi = 0
        self.mu_0 = None
        self.mu_1 = None
        self.sigma = None
    
    
    def fit(self, X, y):
        self.phi = np.mean(y == 1)
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        t0 = X[y == 0] - self.mu_0
        t1 = X[y == 1] - self.mu_1
        self.sigma = ((t0.T @ t0) + (t1.T @ t1)) / X.shape[0]
    
    
    def predict(self, X):
        z = 
        return sigmoid(z)
    
    
    def pdf(self, X):
        py_0 = multivariate_normal.pdf(data, self.mu_0, self.sigma)
        py_1 = multivariate_normal.pdf(data, self.mu_1, self.sigma)
        y[i] = np.argmax([p0 * py_0, p1 * py_1])
        
        return y
    