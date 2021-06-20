import numpy as np 

from linear_model import LinearModel


class Perceptron(LinearModel):
    """Class for the perceptron model.
    
    Example usage:
        > clf = Perceptron()
        > clf.fit(x_train, y_train)
        > clf.predict(x_valid)
    """

    def fit(self, x, y):
        """Run solver to fit linear model."""
        self.theta = np.zeros(x.shape[1])

        if self.solver is None or self.solver == 'gradient':
            self.gradient_ascent(x, y)
        else:
            raise NotImplementedError(f"Método {self.solver} não implementado.")            


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
                print(f"Perda na iteração {i}: {J}")


    def predict(self, x):
        """Make a prediction given new inputs x."""
        h_x = x @ self.theta
        return np.where(h_x >= 0, 1, 0)


    def loss(self, y, h_x):
        """Function that measures the quality of the model."""
        # 0-1 loss.
        return np.sum(y == h_x)
