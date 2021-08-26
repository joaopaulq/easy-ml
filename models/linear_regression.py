import numpy as np


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
        X = self._add_intercept(X)
        inv = np.linalg.pinv(X.T @ X)
        theta = inv @ X.T @ y
        self.w, self.b = theta[1:], theta[0]


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        return X @ self.w + self.b


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,). NumPy array.
            h_x: Predict values of shape (m,). NumPy array.

        Returns:
            J: How close the h_x are to the corresponding y. Float.
        """
        return 0.5 * np.sum(np.square(h_x - y)) # Mean squared error (MSE).


    def _add_intercept(X):
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
