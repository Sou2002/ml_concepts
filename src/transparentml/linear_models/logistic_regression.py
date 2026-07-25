"""Implementation of Logistic Regression via Gradient Descent."""

from typing import Optional

import numpy as np

from src.transparentml._typing import (
    FeatureMatrix, TargetVector, ClassLabels, TargetVectorArray, WeightVector
)


class LogisticRegression:
    """Binary logistic regression fit via batch gradient ascent on
    the log-likelihood.

    Parameters
    ----------
    learning_rate : float, default=0.01
        Step size for gradient updates.
    epochs : int, default=10
        Number of full passes over the training data.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or None
        Estimated coefficients for each feature. ``None`` until `fit`
        is called.
    intercept_ : float or None
        Estimated intercept (bias) term. ``None`` until `fit` is called.
    """

    def __init__(self, learning_rate: float = 0.01, epochs: int = 10) -> None:
        """Initialize an unfitted LogisticGD model.

        Parameters
        ----------
        learning_rate : float, default=0.01
            Step size for gradient updates.
        epochs : int, default=10
            Number of full passes over the training data.
        """
        self.intercept_: Optional[float] = None
        self.coef_: Optional[WeightVector] = None
        self.learning_rate: float = learning_rate
        self.epochs: int = epochs

    def _sigmoid(self, x: WeightVector) -> WeightVector:
        """Compute the elementwise logistic sigmoid.

        Parameters
        ----------
        x : ndarray
            Input values (typically the linear combination ``Xw``).

        Returns
        -------
        ndarray
            Sigmoid-transformed values, in the range (0, 1).
        """
        return 1 / (1 + np.exp(-x))

    def fit(self, X_train: FeatureMatrix, y_train: TargetVector) -> "LogisticRegression":
        """Fit logistic regression via batch gradient ascent.

        Prepends a column of ones to `X_train` so the intercept is
        estimated jointly with the feature weights, then iteratively
        updates weights in the direction of the log-likelihood
        gradient.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Binary training labels, expected to be 0 or 1.

        Returns
        -------
        self : LogisticRegression
            The fitted estimator.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        X_train = np.insert(X_train, 0, 1, axis=1)
        weights = np.ones(X_train.shape[1])

        for _ in range(self.epochs):
            y_pred = self._sigmoid(X_train @ weights)
            gradient = (X_train.T @ (y_train - y_pred)) / X_train.shape[0]
            weights = weights + self.learning_rate * gradient

        self.intercept_, self.coef_ = weights[0], weights[1:]
        return self

    def predict_proba(self, X_test: FeatureMatrix) -> TargetVectorArray:
        """Predict the probability of the positive class (label 1).

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted probabilities in the range (0, 1).
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        return self._sigmoid(X_test @ self.coef_ + self.intercept_)

    def predict(self, X_test: FeatureMatrix) -> ClassLabels:
        """Predict binary class labels.

        Applies a 0.5 threshold to `predict_proba`.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        ndarray of shape (n_samples,)
            Predicted class labels, 0 or 1.
        """
        return np.where(self.predict_proba(X_test) > 0.5, 1, 0)