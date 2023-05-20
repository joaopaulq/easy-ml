import numpy as np

from scipy.stats import mode


class KNN:
    """Class for the K-Nearest Neighbors (Classification) model.
    
    Attributes:
        k: Number of neighbors. Integer. Default=2.

    Example of usage:
        > clf = KNN()
        > clf.fit(X_train, y_train)
        > clf.predict(X_test)
    """

    def __init__(self, k=2):
        self.k = k


    def fit(self, X, y):
        """Memorize all the training data.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            y: Training examples labels of shape (m,). NumPy array.
        """
        self.X = X
        self.y = y


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
            distances = [np.linalg.norm(data - x) for x in self.X]
            # Take the top k nearest neighbors.
            neighbors = np.argsort(distances)
            top_k = neighbors[:self.k]
            h_x[i] = self.y[top_k]
        
        # Assign each object to the most common class among its neighbors.
        h_x = mode(h_x, axis=1, keepdims=False)[0]

        return h_x 
    
