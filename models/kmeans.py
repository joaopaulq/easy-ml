import numpy as np


class KMeans:
    """Class for the K-Means model.

    Attributes:
        k: An integer representing the number of clusters. Default=1.
        centroids: Stores all the K centroids. NumPy array
        clusters: Stores the cluster of each training example. NumPy array.

    Example of usage:
        > clf = KMeans()
        > clf.fit(X_train)
        > clf.predict(X_test)
    """

    def __init__(self, k=1):
        self.k = k
        self.centroids = None
        self.clusters = None


    def fit(self, X, n_init=100, max_iter=1000):
        """Run the K-Means algorithm.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            n_init: Number of times the K-Means algorithm will be run with
                    different centroid seeds. The final results will be the best
                    output of n_init consecutive runs in terms of distortion.
            max_iter: Maximum number of iterations. Integer.
        """
        lowest_distorion = float("inf")
        m, _ = X.shape

        for _ in range(n_init):
            # Pick k random examples to be the initial cluster centroids.
            centroids = X[np.random.choice(m, self.k, replace=False)]
            clusters = np.zeros(m, dtype=int)

            for _ in range(max_iter):
                # Assign each example to the closest cluster centroid.
                for i, data in enumerate(X):
                    distances = [self._dist(data, C) for C in centroids]
                    clusters[i] = np.argmin(distances)

                # Move each cluster centroid to the mean of the points assigned.
                prev_centroids = centroids
                for j in range(self.k):
                    centroids[j] = np.mean(X[clusters == j], axis=0)

                # Stop if converges.
                if np.allclose(centroids, prev_centroids):
                    break

            # Pick the clustering that gives the lowest distortion.
            D = self._distortion(X, centroids, clusters)
            if D < lowest_distorion:
                self.centroids = clusters
                self.clusters = centroids
                lowest_distorion = D


    def predict(self, X):
        """Make a prediction given new inputs.

        Args:
            X: Inputs of shape (m, n). NumPy array.

        Returns:
            h_x: Class predictions of shape (m,). NumPy array.
        """
        h_x = np.zeros(X.shape[0], dtype=int)
        for i, data in enumerate(X):
            # Assign each example to the closest cluster centroid.
            distances = [self._dist(data, C) for C in self.centroids]
            h_x[i] = np.argmin(distances)

        return h_x


    def _distortion(self, X, C, y):
        """Computes the distortion of a clustering.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            C: The centroids of shape (k,). NumPy array.
            y: Cluster of each training example of shape (m,). NumPy array.

        Returns:
            Sum of distances between each example and the cluster centroid
            to which it has been assigned. Float.
        """
        distances = [self._dist(data, C[y[i]]) for i, data in enumerate(X)]
        return sum(distances)


    def _dist(self, x, y, measure='e'):
        """Computes the distance between two NumPy arrays.

        Args:
            x: A NumPy array of shape (m,).
            y: A NumPy array of shape (m,)
            measure: Distance measure. Float

        Returns:
            Distance between x and y using the given measure. Float.
        """
        if measure == 'euclidian':
            return np.linalg.norm(x - y)
        elif measure == 'manhattan':
            return np.sum(np.abs(x - y))
        else:
            raise NotImplementedError(f'Measure {measure} not implemented')