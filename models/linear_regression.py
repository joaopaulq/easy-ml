import numpy as np


class LinearRegression(object):
    """Class for the linear regression model.
   
    Example usage:
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y):
        """Run the least squares method.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        inv = np.linalg.pinv(x.T @ x)
        self.theta = inv @ x.T @ y
        

    def predict(self, x):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).
        
        Returns:
            h_x: Predictions of shape (m,).
        """
        return x @ self.theta
    

    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).
        
        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        # Mean squared error (MSE).
        return 0.5 * np.sum(np.square(h_x - y))