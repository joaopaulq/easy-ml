import numpy as np


class KMeans(object):
    """Class for the K-Means model.
           
    Example usage:
        > clf = KMeans()
        > clf.fit(x_train)
        > clf.predict(x_valid)
    """

    def __init__(self, k=2):
        """
        Args:
            k: The number of clusters.
        """
        self.k = k
        self.centroids = None
        self.C = None


    def fit(self, X, max_iter=100):
        """Run the K-Means algorithm.
        
        Args:
            X: Training example inputs of shape (m, n).
            max_iter: Maximum number of iterations.
        """
        lowest_distortion = 10e7

        for _ in range(max_iter):
            centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]
            C = {}

            while True:
                for idx, data in enumerate(X):
                    C[idx]  = np.argmin(
                        [self._dist(data, c) for c in centroids]
                    )
        
                prev_centroids = centroids
                for j in range(self.k):
                    centroids[j] = np.mean(
                        np.array([X[idx] for idx, u in C.items() if u == j]),
                        axis=0,
                    )

                if np.allclose(centroids, prev_centroids):
                    distortion = 0 
                    for idx, data in enumerate(X):
                        u = C[idx]
                        distortion += self._dist(data, centroids[u])
                    
                    if distortion < lowest_distortion:
                        self.centroids = centroids
                        self.C = C
                        lowest_distortion = distortion

                    break

    
    def labels(self):
        pass


    def predict(self, X):
        """Make a prediction given new inputs X.
        
        Args:
            X: Inputs of shape (m, n).
        
        Returns:
            y: Class predictions of shape (m,).
        """
        y = np.array(X.shape[0])
        for idx, data in enumerate(X):
            y[idx] = np.argmin(
                [self._dist(data, c) for c in self.centroids]
            )
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
    


