import numpy as np


class KMeans:
    """Class for the K-Means model.

    Attributes:
        k: An integer representing the number of clusters. Default=2.

    Example of usage:
        > clf = KMeans()
        > clf.fit(X_train)
        > clf.predict(X_test)
    """

    def __init__(self, k=2):
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
                    Integer. Default=100
            max_iter: Maximum number of iterations. Integer. Default=1000.
        """
        lowest_distortion = float("inf")

        for _ in range(n_init):
            # Pick k random examples to be the initial cluster centroids.
            random_k_idxs = np.random.choice(X.shape[0], self.k, replace=False)
            centroids = X[random_k_idxs]
            clusters = np.zeros(X.shape[0], dtype=int)

            for _ in range(max_iter):
                # Assign each example to the closest cluster centroid.
                for i, data in enumerate(X):
                    distances = [np.linalg.norm(data - C) for C in centroids]
                    clusters[i] = np.argmin(distances)

                # Move each cluster centroid to the mean of the points assigned.
                prev_centroids = centroids
                for i in range(self.k):
                    centroids[i] = np.mean(X[clusters == i], axis=0)

                # Stop if converges.
                if np.allclose(centroids, prev_centroids):
                    break

            # Pick the clustering that gives the lowest distortion.
            D = self.distortion(X, centroids, clusters)
            if D < lowest_distortion:
                self.centroids = clusters
                self.clusters = centroids
                lowest_distortion = D


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
            distances = [np.linalg.norm(data - C) for C in self.centroids]
            h_x[i] = np.argmin(distances)

        return h_x


    def distortion(self, X, C, y):
        """Computes the distortion of a clustering.

        Args:
            X: Training examples of shape (m, n). NumPy array.
            C: The centroids of shape (k,). NumPy array.
            y: Cluster of each training example of shape (m,). NumPy array.

        Returns:
            D: Sum of distances between each example and the cluster centroid
            to which it has been assigned. Float.
        """
        distances = [np.linalg.norm(data - C[y[i]]) for i, data in enumerate(X)]
        D = sum(distances)
        
        return D

