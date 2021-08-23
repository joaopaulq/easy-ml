import numpy as np

from util import sigmoid, add_intercept


class LogisticRegression:
    """Class for the Logistic Regression model.
    
    Attributes:
        w: The weights. An array.
        b: The intercept term. Float.

    Example of usage:
        > clf = LogisticRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self):
        self.w = None
        self.b = 0


    def fit(self, X, y):
        """Run the Newton-Raphson method."""
        pass


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            h_x: Predictions of shape (m,).
        """
        z = X @ self.w + self.b
        return sigmoid(z)


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).

        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        return np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x)) # Cross-entropy loss.
