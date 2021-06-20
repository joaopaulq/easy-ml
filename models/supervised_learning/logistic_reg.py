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

    def fit(self, x, y):
        """Run solver to fit linear model."""
        self.theta = np.zeros(x.shape[1])

        if self.solver is None or self.solver == 'newton':
            self.newtons_method(x, y) 
        elif self.solver == 'gradient':
            self.gradient_ascent(x, y)
        else:
            raise NotImplementedError(f"Método {self.solver} não implementado.")


    def newtons_method(self, x, y):
        """Run the Newton-Raphson method.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        pass 


    def gradient_ascent(self, x, y):
        """Run the gradient ascent algorithm.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
        """
        assert self.max_iter >= 0 and self.lr > 0

        for i in range(self.max_iter):
            h_x = self.predict(x)
            J = self.loss(y, h_x)
            dJ = x.T @ (y - h_x)
            theta_prev = self.theta
            self.theta = self.theta + self.lr*dJ
            
            if np.allclose(self.theta, theta_prev, atol=self.eps):
                break

            if self.verbose and i % 10:
                print(f"Loss on iteration {i}: {J}")
    

    def predict(self, x):
        """Make a prediction given new inputs x. """
        z = x @ self.theta
        return sigmoid(z)
    

    def loss(self, y, h_x):
        """Function that measures the quality of the model."""
        # Cross-entropy loss.
        return np.sum(y*np.log(h_x) + (1-y)*np.log(1-h_x))
