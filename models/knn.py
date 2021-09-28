import numpy as np

from scipy.stats import mode
from models.util import dist


class KNN:
    """Class for the K-Nearest Neighbors (Classification) model.
    
    Attributes:
        k: Number of neighbors. Integer. Default=1.
        X: Training examples of shape (m, n). NumPy array.
        y: Training examples labels of shape (m,). NumPy array.

    Example of usage:
        > clf = KNN()
        > clf.fit(X_train, y_train)
        > clf.predict(X_valid)
    """

    def __init__(self, k=1):
        self.k = k
        self.X = None 
        self.y = None


    def fit(self, X, y):
        """Memorize all the training data.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
        """
        self.X, self.y = X, y


    def predict(self, X):
        """Make a prediction given new inputs.
        
        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Predictions of shape (m,). NumPy array.
        """
        h_x = np.zeros((X.shape[0], self.k))

        for i, data in enumerate(X):
            # Compute the distance to each training example.
            D = [dist(data, x) for x in self.X]
            # Take the k nearest neighbors.
            neighbors = np.argsort(D)[:self.k]
            h_x[i] = self.y[neighbors]
        
        # Assign each object to the class most common among its neighbors.
        return mode(h_x, axis=1)[0]
    
