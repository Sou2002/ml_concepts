"""Implementation of Linear Regression."""

from typing import Optional

import numpy as np

from src.transparentml._typing import (
    FeatureMatrix, TargetVector, TargetVectorArray, WeightVector
)


class LinearRegression:
    """Ordinary Least Squares (OLS) Linear Regression.

    Fits a linear model of the form ``y = X @ coef_ + intercept_`` by
    minimizing the sum of squared residuals between observed and
    predicted targets, solved directly via the normal equation.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or None
        Estimated coefficients for each feature. ``None`` until `fit`
        is called.
    intercept_ : float or None
        Estimated intercept (bias) term. ``None`` until `fit` is called.
    weights : ndarray of shape (n_features + 1,) or None
        Combined weight vector solved by the normal equation, where
        ``weights[0]`` is the intercept and ``weights[1:]`` are the
        feature coefficients. Kept for internal use; prefer `coef_`
        and `intercept_` for inspecting the fitted model.
    """

    def __init__(self) -> None:
        """Initialize an unfitted LinearRegression model."""
        self.coef_: Optional[WeightVector] = None
        self.intercept_: Optional[np.float64] = None
        self.weights: Optional[WeightVector] = None

    def fit(self, X_train: FeatureMatrix, y_train: TargetVector) -> "LinearRegression":
        """Fit the linear regression model using the normal equation.

        Solves ``weights = (XᵀX)⁻¹Xᵀy`` after prepending a column of
        ones to `X_train` so the intercept is estimated jointly with
        the feature coefficients.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
            Training feature matrix.
        y_train : array-like of shape (n_samples,)
            Training target values.

        Returns
        -------
        self : LinearRegression
            The fitted estimator.

        Notes
        -----
        This computes the matrix inverse explicitly for clarity. It is
        not the most numerically stable approach for ill-conditioned
        or highly correlated features — production libraries such as
        scikit-learn instead use ``lstsq`` or SVD-based solvers.
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        # Prepend a column of ones so the intercept is solved for
        # jointly with the feature weights.
        X_train = np.insert(X_train, 0, values=1, axis=1)

        self.weights = np.linalg.inv(X_train.T @ X_train) @ X_train.T @ y_train

        self.intercept_ = self.weights[0]
        self.coef_ = self.weights[1:]

        return self

    def predict(self, X_test: FeatureMatrix) -> TargetVectorArray:
        """Predict target values for new samples.

        Parameters
        ----------
        X_test : array-like of shape (n_samples, n_features)
            Samples to predict on.

        Returns
        -------
        y_pred : ndarray of shape (n_samples,)
            Predicted target values.
        """
        X_test = np.asarray(X_test, dtype=np.float64)
        y_pred = X_test @ self.coef_ + self.intercept_
        return y_pred