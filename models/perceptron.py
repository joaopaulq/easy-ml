import numpy as np

from util import add_intercept


class Perceptron:
    """Class for the Perceptron model.
    
    Attributes:
        w: The weights. An array.
        b: The intercept term. Float.

    Example of usage:
        > clf = Perceptron()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self):
        self.w = None
        self.b = 0


    def fit(self, X, y, lr=0.2, max_iter=100, eps=1e-5, verbose=False):
        """Run the gradient ascent algorithm.

        Args:
            X: Training examples of shape (m, n).
            y: Training examples labels of shape (m,).
            lr: The learning rate. Float.
            max_iter: Maximum number of iterations. Integer.
            eps: Threshold for determining convergence. Float.
            verbose: Print loss values during training. Boolean.
        """
        X = add_intercept(X)
        # Start theta with the zero vector.
        theta = np.zeros(X.shape[1])
        self.w = theta[1:]

        for i in range(max_iter):
            # Make a prediction.
            h_x = self.predict(X[:, 1:])
            # Compute the gradient.
            dJ = X.T @ (y - h_x)   
            # Update rule.
            prev_theta = theta
            theta = theta + lr*dJ
            
            self.w, self.b = theta[1:], theta[0]
            
            # Stop if converges.
            if np.allclose(theta, prev_theta, atol=eps):
                break

            if verbose and i % 10 == 0:
                # Print the loss.
                J = self.loss(y, h_x)
                print(f"Loss on iteration {i}: {J}")
        


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n).

        Returns:
            h_x: Predictions of shape (m,).
        """
        h_x = X @ self.w + self.b
        # Apply the threshold function.
        return np.where(h_x >= 0, 1, 0)


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).

        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        return np.sum(y != h_x) # 0-1 loss.
