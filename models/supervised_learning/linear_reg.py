import numpy as np

from linear_model import LinearModel


class LinearRegression(LinearModel):
    """Class for the linear regression model.
   
    Example usage:
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y, *_):
        """Run the least squares method."""
        self.theta = np.zeros(x.shape[1])
        inv = np.linalg.pinv(x.T @ x)
        self.theta = inv @ x.T @ y
        

    def predict(self, x):
        """Make a prediction given new inputs x."""
        return x @ self.theta
    

    def loss(self, y, h_x):
        """Calculate the mean squared error (MSE)."""
        return 0.5 * np.sum(np.square(h_x - y))