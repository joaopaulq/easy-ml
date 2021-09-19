import numpy as np

from scipy import stats


class KNN:
    """Class for the K-Nearest Neighbors model.
    
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
            D = [self._dist(data, x) for x in self.X]
            neighbors = np.argsort(D)[:self.k]
            h_x[i] = self.y[neighbors]

        return stats.mode(h_x, axis=1)[0]

    
    def _dist(self, x, y, measure='e'):
        """Computes the distance between two NumPy arrays.

        Args:
            x: A NumPy array.
            y: A NumPy array.
            measure: Distance measure.

        Returns:
            Distance between x and y using the given measure. Float.
        """
        if measure == 'e':
            return np.linalg.norm(x - y) # Euclidian distance.
        elif measure == 'm':
            return np.sum(np.abs(x - y)) # Manhattan distance.
        else:
            raise NotImplementedError(f'Measure {measure} not implemented')
    
