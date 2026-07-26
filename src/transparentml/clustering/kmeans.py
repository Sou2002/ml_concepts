"""Implementation of K-Means clustering."""

from typing import Optional

import numpy as np

from src.transparentml._typing import FeatureMatrix, FeatureMatrixArray, ClassLabels


class KMeans:
    """
    K-Means clustering via Lloyd's algorithm.

    Iteratively assigns points to the nearest centroid, then updates
    each centroid to the mean of its assigned points, until centroids
    stop changing or `max_iter` is reached.

    Parameters
    ----------
    n_clusters : int
        Number of clusters to form.
    max_iter : int, default=100
        Maximum number of iterations of the assign/update loop.
    random_state : int, optional
        Seed for reproducible centroid initialization.

    Attributes
    ----------
    cluster_centers_ : ndarray of shape (n_clusters, n_features) or None
        Coordinates of cluster centroids. ``None`` until `fit_predict`
        is called.
    labels_ : ndarray of shape (n_samples,) or None
        Cluster label assigned to each training sample. ``None`` until
        `fit_predict` is called.

    Notes
    -----
    If a cluster receives zero points during an update step, its
    centroid becomes ``nan`` (mean of an empty set). This basic
    implementation does not handle re-seeding empty clusters.
    """

    def __init__(self, n_clusters: int, max_iter: int = 100, random_state: Optional[int] = None) -> None:
        """
        Initialize an unfitted KMeans model.

        Parameters
        ----------
        n_clusters : int
            Number of clusters to form.
        max_iter : int, default=100
            Maximum number of iterations of the assign/update loop.
        random_state : int, optional
            Seed for reproducible centroid initialization.
        """
        self.n_clusters: int = n_clusters
        self.max_iter: int = max_iter
        self.random_state: Optional[int] = random_state
        self.cluster_centers_: Optional[FeatureMatrixArray] = None
        self.labels_: Optional[ClassLabels] = None

    def fit_predict(self, X: FeatureMatrix) -> ClassLabels:
        """
        Compute cluster centroids and predict cluster labels for `X`.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Data to cluster.

        Returns
        -------
        ndarray of shape (n_samples,)
            Cluster label assigned to each sample.
        """
        X = np.asarray(X, dtype=np.float64)

        rng = np.random.default_rng(self.random_state)
        centroid_idx = rng.choice(X.shape[0], size=self.n_clusters, replace=False)
        self.cluster_centers_ = X[centroid_idx]

        cluster_label = None
        for _ in range(self.max_iter):
            cluster_label = self._assign_clusters(X)
            old_centroids = self.cluster_centers_
            self.cluster_centers_ = self._move_centroids(X, cluster_label)

            if np.array_equal(old_centroids, self.cluster_centers_):
                break

        self.labels_ = cluster_label
        return cluster_label

    def _assign_clusters(self, X: FeatureMatrixArray) -> ClassLabels:
        """
        Assign each sample to its nearest centroid.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data to assign.

        Returns
        -------
        ndarray of shape (n_samples,)
            Index of the nearest centroid for each sample.
        """
        cluster_label = [
            np.argmin(np.linalg.norm(self.cluster_centers_ - row, axis=1))
            for row in X
        ]
        return np.array(cluster_label, dtype=np.int64)

    def _move_centroids(self, X: FeatureMatrixArray, cluster_label: ClassLabels) -> FeatureMatrixArray:
        """
        Recompute each centroid as the mean of its assigned points.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Data being clustered.
        cluster_label : ndarray of shape (n_samples,)
            Current cluster assignment for each sample.

        Returns
        -------
        ndarray of shape (n_clusters, n_features)
            Updated centroid coordinates.
        """
        new_centroids = [
            X[cluster_label == label].mean(axis=0)
            for label in np.unique(cluster_label)
        ]
        return np.array(new_centroids)