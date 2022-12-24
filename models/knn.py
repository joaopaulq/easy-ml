import numpy as np

from scipy.stats import mode


class KNN:
    """Class for the K-Nearest Neighbors (Classification) model.
    
    Attributes:
        k: Number of neighbors. Integer. Default=1.
        X: Training examples of shape (m, n). NumPy array.
        y: Training examples labels of shape (m,). NumPy array.

    Example of usage:
        > clf = KNN()
        > clf.fit(X_train, y_train)
        > clf.predict(X_test)
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
            distances = [self._dist(data, x) for x in self.X]
            # Take the top k nearest neighbors.
            neighbors = np.argsort(distances)
            top_k = neighbors[:self.k]
            h_x[i] = self.y[top_k]
        
        # Assign each object to the most common class among its neighbors.
        return mode(h_x, axis=1)[0]
    

    def _dist(self, x, y, measure='euclidian'):
        """Computes the distance between two NumPy arrays.

        Args:
            x: A NumPy array of shape (m,).
            y: A NumPy array of shape (m,)
            measure: Distance measure. [euclidian, manhattan]

        Returns:
            Distance between x and y using the given measure. Float.
        """
        if measure == 'euclidian':
            return np.linalg.norm(x - y)
        elif measure == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            raise NotImplementedError(f'Measure {measure} not implemented')
    
