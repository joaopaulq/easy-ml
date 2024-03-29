import numpy as np


class LogisticRegression:
    """Class for the Logistic Regression model.
    
    Attributes:
        w: The weights. NumPy array.
        b: The intercept term. Float.

    Example of usage:
        > clf = LogisticRegression()
        > clf.fit(X_train, y_train)
        > clf.predict(X_test)
    """

    def __init__(self):
        self.w = None
        self.b = 0


    def fit(self, X, y, max_iter=1000, verbose=False):
        """Run the Newton-Raphson method.
        
        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
            max_iter: Maximum number of iterations. Integer. Default=1000.
            verbose: Print loss during training. Boolean. Default=False.
        """
        m, n = X.shape
        # Start the weights with the zero vector.
        self.w = np.zeros(n) 
        
        for i in range(max_iter):
            # Make a prediction.
            h_x = self.predict(X)
            
            # Compute the gradient.
            dJw = (X.T @ (y - h_x)) / m
            dJb = np.sum(y - h_x) / m
            d2Jb = -m
            
            # Compute the hessian and its inverse.
            D = np.diag(h_x * (1 - h_x))
            H = (X.T @ D @ X) / m 
            H_inv = np.linalg.pinv(H)
            
            # Update rule.
            prev_w, prev_b = self.w, self.b
            self.w = self.w - H_inv @ dJw
            self.b = self.b - (dJb / d2Jb)
            
            if verbose and i % 100 == 0:
                J = self.loss(y, h_x)
                print(f"Loss on iteration {i}: {J}")
            
            # Stop if converges.
            if np.allclose(prev_w, self.w) and np.isclose(prev_b, self.b):
                break
        

    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        z = X @ self.w + self.b
        # Apply the sigmoid function.
        h_x = self.sigmoid(z)

        return h_x
    
    
    def loss(self, y, h_x):
        """Function that measures the quality of the model.
        
        Args:
            y: Targets values of shape (m,). NumPy array.
            h_x: Predict values of shape (m,). NumPy array.
        
        Returns:
            J: How close the h_x are to the corresponding y. Float.
        """
        # Cross-entropy loss.
        J = np.mean(y * np.log(h_x) + (1-y) * np.log(1-h_x))

        return J
    

    def sigmoid(self, z):
        """Computes the sigmoid function on a NumPy array."""
        return 1.0 / (1 + np.exp(-z))