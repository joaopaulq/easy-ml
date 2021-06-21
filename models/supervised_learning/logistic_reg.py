import numpy as np 

from util import sigmoid 
from linear_model import LinearModel 


class LogisticRegression(LinearModel):
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
        """Make a prediction given new inputs x."""
        z = x @ self.theta
        return sigmoid(z)
    

    def loss(self, y, h_x):
        """Calculate the cross-entropy loss."""
        return np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
