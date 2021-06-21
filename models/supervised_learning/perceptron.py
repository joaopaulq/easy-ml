import numpy as np 

from linear_model import LinearModel


class Perceptron(LinearModel):
    """Class for the perceptron model.
       
    Example usage:
        > clf = Perceptron()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y, lr=0.2, max_iter=100, eps=1e-5, verbose=False):
        """Run the gradient ascent algorithm."""
        self.theta = np.zeros(x.shape[1])
        
        for i in range(max_iter):
            h_x = self.predict(x)
            J = self.loss(y, h_x)
            dJ = x.T @ (y - h_x)
            theta_prev = self.theta
            self.theta = self.theta + lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=eps):
                break

            if verbose and i % 10:
                print(f"Loss on iteration {i}: {J}")


    def predict(self, x):
        """Make a prediction given new inputs x."""
        h_x = x @ self.theta
        return np.where(h_x >= 0, 1, 0)


    def loss(self, y, h_x):
        """Calculate the 0-1 loss."""
        return np.sum(y == h_x)
