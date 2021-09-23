import numpy as np


class Perceptron:
    """Class for the Perceptron model.
    
    Attributes:
        w: The weights. NumPy array.
        b: The intercept term (bias). Float.

    Example of usage:
        > clf = Perceptron()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self):
        self.w = None
        self.b = 0


    def fit(self, X, y, lr=0.1, max_iter=500, verbose=False):
        """Run the gradient ascent algorithm.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
            lr: The learning rate. Float.
            max_iter: Maximum number of iterations. Integer.
            verbose: Print loss values during training. Boolean.
        """
        m, n = X.shape
        # Start the weights with the zero vector.
        self.w = np.zeros(n)

        for i in range(max_iter):
            # Make a prediction.
            h_x = self.predict(X)
            
            # Compute the gradient.
            dJw = (X.T @ (y - h_x)) / m # Derivative of loss wrt weights.
            dJb = np.sum(y - h_x) / m # Derivative of loss wrt bias.
            
            # Update rule.
            prev_w, prev_b = self.w, self.b
            self.w = self.w + lr*dJw
            self.b = self.b + lr*dJb
            
            # Stop if converges.
            if np.allclose(prev_w, self.w) and np.isclose(prev_b, self.b):
                break
            
            if verbose and i % 10 == 0:
                J = self._loss(y, h_x)
                print(f"Loss on iteration {i}: {J}")
        

    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        z = X @ self.w + self.b
        # Apply the threshold function.
        return np.where(z >= 0, 1, 0)


    def _loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,). NumPy array.
            h_x: Predict values of shape (m,). NumPy array.

        Returns:
            J: How close the h_x are to the corresponding y. Float.
        """
        return np.sum(y != h_x) # 0-1 loss.