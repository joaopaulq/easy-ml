import numpy as np

from scipy.stats import multivariate_normal


class GDA:
    """Class for the Gaussian Discriminant Analysis model.

    Attributes:
        phi: Proportion of examples from class 0. Float.
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
        """Find the maximum likelihood estimate of the parameters.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
        """
        self.phi = np.mean(y == 0)
        self.mu_0 = np.mean(X[y == 0], axis=0)
        self.mu_1 = np.mean(X[y == 1], axis=0)
        t0 = X[y == 0] - self.mu_0
        t1 = X[y == 1] - self.mu_1
        self.sigma = ((t0.T @ t0) + (t1.T @ t1)) / X.shape[0]


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        # Probability of each class being true. (Prior).
        p0, p1 = self.phi, 1 - self.phi

        # Models the distribution of each class. (Likelihood).
        px_0 = multivariate_normal.pdf(X, mean=self.mu_0, cov=self.sigma)
        px_1 = multivariate_normal.pdf(X, mean=self.mu_1, cov=self.sigma)

        # Use Bayes rule to derive the distribution on y given x. (Posterior).
        p0_x = p0 * px_0
        p1_x = p1 * px_1

        return np.argmax([p0_x, p1_x], axis=0)
