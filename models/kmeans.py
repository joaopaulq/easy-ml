import numpy as np


class KMeans:
    """Class for the K-Means model.

    Attributes:
        k: An integer representing the number of clusters. Default=1.
        measure: Distance measure. Default='e' (Euclidian distance).
        centroids: A NumPy array that stores all the K centroids.
        clusters: A NumPy array that stores the cluster of each training
                  example.

    Example of usage:
        > clf = KMeans()
        > clf.fit(X_train)
        > clf.predict(X_valid)
    """

    def __init__(self, k=1, measure='e'):
        self.k = k
        self.measure = measure
        self.centroids = None
        self.clusters = None


    def fit(self, X, max_iter=100):
        """Run the K-Means algorithm.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            max_iter: Maximum number of iterations. Integer.
        """
        lowest_distortion = float("inf")
        m, _ = X.shape

        for _ in range(max_iter):
            # Pick k random examples to be the initial cluster centroids.
            centroids = X[np.random.choice(m, self.k, replace=False)]
            clusters = np.zeros(m, dtype=int)

            while True:
                # Assign each example to the closest cluster centroid.
                for i, data in enumerate(X):
                    clusters[i] = np.argmin(
                        [self._dist(data, c) for c in centroids]
                    )

                # Move each cluster centroid to the mean of the points assigned.
                prev_centroids = centroids
                for j in range(self.k):
                    centroids[j] = np.mean(X[clusters == j], axis=0)

                # Stop if converges.
                if np.allclose(centroids, prev_centroids):
                    break

            # Pick the clustering that gives the lowest distortion.
            d = self._distortion(X, centroids, clusters)
            if d < lowest_distortion:
                self.centroids, self.clusters = centroids, clusters
                lowest_distortion = d


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            y: Class predictions of shape (m,). NumPy array.
        """
        y = np.zeros(X.shape[0], dtype=int)
        for i, data in enumerate(X):
            # Assign each example to the closest cluster centroid.
            y[i] = np.argmin([self._dist(data, c) for c in self.centroids])

        return y


    def _distortion(self, X, c, y):
        """Computes the distortion of a clustering.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            c: The centroids of shape (k,). NumPy array.
            y: Cluster of each training example of shape (m,). NumPy array.

        Returns: Sum of distances between each example and the cluster
                 centroid to which it has been assigned. Float.
        """
        return sum([self._dist(data, c[y[i]]) for i, data in enumerate(X)])


    def _dist(self, x, y):
        """Computes the distance between two NumPy arrays.

        Args:
            x: A NumPy array.
            y: A NumPy array.

        Returns: Distance between x and y using the given measure. Float.
        """
        if self.measure == 'e':
            return np.linalg.norm(x - y) # Euclidian distance.
        elif self.measure == 'm':
            return np.sum(np.abs(x - y)) # Manhattan distance.
        else:
            raise NotImplementedError(f'Measure {self.measure} not implemented.')
