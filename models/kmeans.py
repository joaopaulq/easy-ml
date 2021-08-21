import numpy as np


class KMeans(object):
    """Class for the K-Means model.

    Example usage:
        > clf = KMeans()
        > clf.fit(X_train)
        > clf.predict(X_valid)
    """

    def __init__(self, k=1):
        """Args: k: The number of clusters."""
        self.k = k
        self.centroids = None
        self.labels = None


    def fit(self, X, max_iter=100):
        """Run the K-Means algorithm.

        Args:
            X: Training examples of shape (m, n).
            max_iter: Maximum number of iterations.
        """
        lowest_distortion = float("inf")
        m, _ = X.shape

        for _ in range(max_iter):
            # Pick k random examples to be the inital cluster centroids.
            centroids = X[np.random.choice(m, self.k, replace=False)]
            labels = np.zeros(m, dtype=int)

            while True:
                # Assign each example to the closest cluster centroid.
                for i, data in enumerate(X):
                    labels[i] = np.argmin(
                        [self._dist(data, ct) for ct in centroids]
                    )

                # Move each cluster centroid to the mean of the points assigned.
                prev_centroids = centroids
                for j in range(self.k):
                    centroids[j] = np.mean(X[np.argwhere(labels == j)], axis=0)

                # Stop if converges.
                if np.allclose(centroids, prev_centroids):
                    break

            # Pick the clustering that gives the lowest distortion.
            d = self._distortion(X, centroids, labels)
            if d < lowest_distortion:
                self.centroids, self.labels = centroids, labels
                lowest_distortion = d


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


    def _distortion(self, X, c, y):
        """Computes the distortion of a clustering.

        Args:
            X: Training examples of shape (m, n).
            c: The centroids.
            y: Class of each training example.

        Returns: Sum of distances between each example and the centroid to
                 which it has been assigned.
        """
        return sum([self._dist(data, c[y[i]]) for i, data in enumerate(X)])


    def _dist(self, x, y, m='e'):
        """Computes the distance between two NumPy arrays.

        Args:
            x: A NumPy array.
            y: A NumPy array.
            m: Distance measure.

        Returns: Distance between x and y using measure m.
        """
        if m == 'e':
            return np.linalg.norm(x - y) # Euclidian distance.
        elif m == 'm':
            return np.sum(np.abs(x - y)) # Manhattan (cityblock) distance.
        else:
            raise NotImplementedError(f'Measure {m} not implemented.')
