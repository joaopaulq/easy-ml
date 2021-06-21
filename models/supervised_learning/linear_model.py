class LinearModel(object):
    """Base class for linear models."""

    def fit(self, x, y, lr=0.2, max_iter=100, eps=1e-5, verbose=False):
        """Run solver to fit linear model.

        Args:
            x: Training example inputs. Shape (m, n).
            y: Training example labels. Shape (m,).
            lr: The learning rate.
            max_iter: Maximum number of iterations.
            eps: Threshold for determining convergence.
            verbose: Print loss values during training.
        """
        raise NotImplementedError(
            'Subclass of LinearModel must implement fit method.')


    def predict(self, x):
        """Make a prediction given new inputs x.
        
        Args:
            x: Inputs of shape (m, n).
        
        Returns:
            h_x: Predictions of shape (m,).
        """
        raise NotImplementedError(
            'Subclass of LinearModel must implement predict method.')
    

    def loss(self, y, h_x):
        """Function that measures the quality of the model.

        Args:
            y: Targets values of shape (m,).
            h_x: Predict values of shape (m,).
        
        Returns:
            J: How close the h_x are to the corresponding y. Scalar.
        """
        raise NotImplementedError(
            'Subclass of LinearModel must implement loss method.')
