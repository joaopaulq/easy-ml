import numpy as np


class KMeans(object):
    """Class for the K-Means model.
           
    Example usage:
        > clf = KMeans()
        > clf.fit(X_train)
        > clf.predict(X_valid)
    """

    def __init__(self, k=2):
        """
        Args:
            k: The number of clusters.
        """
        self.k = k
        self.centroids = None
        self.labels = None


    def fit(self, X, max_iter=100):
        """Run the K-Means algorithm.
        
        Args:
            X: Training examples of shape (m, n).
            max_iter: Maximum number of iterations.
        """
        lowest_distortion = 10e7
        m, _ = X.shape

        for _ in range(max_iter):
            centroids = X[np.random.choice(m, self.k, replace=False)]
            labels = np.zeros(m, dtype=int)

            while True:
                for i, data in enumerate(X):
                    labels[i] = np.argmin(
                        [self._dist(data, ct) for ct in centroids])
        
                prev_centroids = centroids
                for j in range(self.k):
                    centroids[j] = np.mean(X[np.argwhere(labels == j)], axis=0)

                if np.allclose(centroids, prev_centroids):
                    break
            
            distortion = 0 
            for i, data in enumerate(X):
                distortion += self._dist(data, centroids[labels[i]])
                    
            if distortion < lowest_distortion:
                self.centroids, self.labels = centroids, labels
                lowest_distortion = distortion


    def predict(self, X):
        """Make a prediction given new inputs X.
        
        Args:
            X: Inputs of shape (m, n).
        
        Returns:
            y: Class predictions of shape (m,).
        """
        y = np.zeros(X.shape[0], dtype=int)
        for i, data in enumerate(X):
            y[i] = np.argmin([self._dist(data, ct) for ct in self.centroids])

        return y
    

    def _dist(self, x, y, m='euclidian'):
        """Computes the distance between two NumPy arrays.
        
        Args:
            x: A NumPy array.
            y: A NumPy array.
            m: Distance measure.

        Returns:
            Distance between x and y using measure m. Scalar.
        """
        if m == 'euclidian':
            return np.linalg.norm(x - y)
        elif m == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            raise NotImplementedError(f'Measure {m} not implemented.')
    


