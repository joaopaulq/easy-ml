import numpy as np 

from util import sigmoid 


class LogisticRegression(object):
    """Class for the logistic regression model.
           
    Example usage:
        > clf = LogisticRegression()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y, lr=0.2, max_iter=100, eps=1e-5, verbose=False):
        """Run the Newton-Raphson method."""
        pass 


    def predict(self, x):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).
        
        Returns:
            h_x: Predictions of shape (m,).
        """
        z = x @ self.theta
        return sigmoid(z)
    

    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).
        
        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        # Cross-Entropy loss.
        return np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
