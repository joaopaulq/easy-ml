import numpy as np


class LinearRegression(object):
    """Class for the Linear Regression model.

    Example usage:
        > clf = LinearRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self):
        self.theta = None


    def fit(self, X, y):
        """Run the least squares method.

        Args:
            X: Training examples of shape (m, n).
            y: Training examples labels of shape (m,).
        """
        inv = np.linalg.pinv(X.T @ X)
        self.theta = inv @ X.T @ y


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            h_x: Predictions of shape (m,).
        """
        return X @ self.theta


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).

        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        return 0.5 * np.sum(np.square(h_x - y)) # Mean squared error (MSE).
