import numpy as np

from linear_model import LinearModel


class LinearRegression(LinearModel):
    """Class for the linear regression model.
   
    Example usage:
        > clf = LinearRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y):
        """Run solver to fit linear model."""
        self.theta = np.zeros(x.shape[1])
 
        if self.solver is None or self.solver == 'lstsq':
            self.least_squares(x, y)
        else:
            raise NotImplementedError(f"Method {self.solver} not implemented.")

    
    def least_squares(self, x, y):
        """Run the least squares method.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        inv = np.linalg.pinv(x.T @ x)
        self.theta = inv @ x.T @ y
        

    def predict(self, x):
        """Make a prediction given new inputs x."""
        return x @ self.theta
    

    def loss(self, y, h_x):
        """Function that measures the quality of the model."""
        # Mean squared error (MSE).
        return 0.5 * np.sum(np.square(h_x - y))