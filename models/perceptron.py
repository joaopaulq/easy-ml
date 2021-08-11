import numpy as np 


class Perceptron(object):
    """Class for the Perceptron model.
       
    Example usage:
        > clf = Perceptron()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """
    
    def __init__(self):
        self.theta = None 


    def fit(self, X, y, lr=0.2, max_iter=100, eps=1e-5, verbose=False):
        """Run the gradient ascent algorithm.
        
        Args:
            X: Training examples of shape (m, n).
            y: Training examples labels of shape (m,).
            lr: The learning rate.
            max_iter: Maximum number of iterations.
            eps: Threshold for determining convergence.
            verbose: Print loss values during training.
        """
        self.theta = np.zeros(X.shape[1])
        
        for i in range(max_iter):
            h_x = self.predict(X)
            dJ = X.T @ (y - h_x)
            theta_prev = self.theta
            self.theta = self.theta + lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=eps):
                break

            if verbose and i % 10 == 0:
                J = self.loss(y, h_x)
                print(f"Loss on iteration {i}: {J}")


    def predict(self, X):
        """Make a prediction given new inputs X.
        
        Args:
            X: Inputs of shape (m, n).
        
        Returns:
            h_x: Predictions of shape (m,).
        """
        h_x = X @ self.theta
        return np.where(h_x >= 0, 1, 0)


    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).
        
        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        # 0-1 loss.
        return np.sum(y == h_x)
