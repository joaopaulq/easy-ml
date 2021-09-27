import numpy as np

from util import add_intercept


class LinearRegression:
    """Class for the Linear Regression model.
    
    Attributes:
        w: The weights. NumPy array.
        b: The intercept term. Float.

    Example of usage:
        > clf = LinearRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self):
        self.w = None
        self.b = 0


    def fit(self, X, y):
        """Run the least squares method.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
        """
        X = add_intercept(X)
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w = theta[1:]
        self.b = theta[0]


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        return X @ self.w + self.b
