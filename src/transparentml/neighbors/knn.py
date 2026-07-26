"""Implementation of K-Nearest Neighbours."""

from statistics import mode
from typing import Optional

import numpy as np

from src.transparentml._typing import (
    FeatureMatrix, TargetVector, FeatureMatrixArray, ClassLabels
)


class KNN:
    """
    K-Nearest Neighbours classifier.

    Predicts a sample's class by majority vote among its
    `n_neighbours` closest training points, using Euclidean distance.

    Parameters
    ----------
    n_neighbours : int, default=5
        Number of nearest neighbours to consider when voting.

    Attributes
    ----------
    X_train : ndarray of shape (n_samples, n_features) or None
        Stored training feature matrix. ``None`` until `fit` is called.
    y_train : ndarray of shape (n_samples,) or None
        Stored training class labels. ``None`` until `fit` is called.
    """

    def __init__(self, n_neighbours: int = 5) -> None:
        """
        Initialize an unfitted KNN classifier.

        Parameters
        ----------
        n_neighbours : int, default=5
            Number of nearest neighbours to consider when voting.
        """
        self.n_neighbours: int = n_neighbours
        self.X_train: Optional[FeatureMatrixArray] = None
        self.y_train: Optional[ClassLabels] = None

    def fit(self, X_train: FeatureMatrix, y_train: TargetVector) -> "KNN":
        """
        Store training data for later distance comparisons.

        KNN is a lazy learner — `fit` performs no computation, it
        simply retains the training set for use at prediction time.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training class labels.

        Returns
        -------
        self : KNN
            The fitted estimator.
        """
        self.X_train = np.asarray(X_train, dtype=np.float64)
        self.y_train = np.asarray(y_train, dtype=np.int64)
        return self

    def predict(self, X_test: FeatureMatrix) -> ClassLabels:
        """
        Predict class labels for new samples.

        For each test sample, computes Euclidean distance to every
        training point, then predicts the majority class among the
        `n_neighbours` closest points.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels.
        """
        X_test = np.asarray(X_test, dtype=np.float64)

        pred_labels = []
        for data_point in X_test:
            distances = sorted(
                enumerate(np.linalg.norm(self.X_train - data_point, axis=1)),
                key=lambda x: x[1]
            )[:self.n_neighbours]

            neighbour_indices = [idx for idx, _ in distances]
            pred_labels.append(mode(self.y_train[neighbour_indices]))

        return np.array(pred_labels, dtype=np.int64)