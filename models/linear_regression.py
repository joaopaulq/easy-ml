import numpy as np

from util import add_intercept


class LinearRegression(object):
    """Class for the Linear Regression model.
    
    Attributes:
        w: The weights. An array.
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
            X: Training examples of shape (m, n).
            y: Training examples labels of shape (m,).
        """
        X = add_intercept(X)
        theta = np.linalg.pinv(X.T @ X) @ X.T @ y
        self.w, self.b = theta[1:], theta[0]


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            h_x: Predictions of shape (m,).
        """
        return X @ self.w + self.b


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).

        Returns:
            J: How close the h_x are to the corresponding y. Float.
        """
        return 0.5 * np.sum(np.square(h_x - y)) # Mean squared error (MSE).
