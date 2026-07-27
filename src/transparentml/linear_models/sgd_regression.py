"""Implementation of SGD Regression."""


import numpy as np

from transparentml._typing import FeatureMatrix, TargetVector, TargetVectorArray, WeightVector


class SGDRegression:
    """Linear regression fit via mini-batch gradient descent.

    Parameters
    ----------
    batch_size : int, default=100
        Number of samples per mini-batch.
    learning_rate : float, default=0.01
        Step size for gradient updates.
    epochs : int, default=10
        Number of full passes over the training data.

    Attributes
    ----------
    coef_ : ndarray of shape (n_features,) or None
    intercept_ : float or None

    Notes
    -----
    Gradient descent is sensitive to feature scale. Features with large
    magnitudes or widely differing scales can cause slow convergence or
    divergence. Standardizing features (zero mean, unit variance) before
    fitting is recommended.
    """

    def __init__(
        self, batch_size: int = 100, learning_rate: float = 0.01, epochs: int = 10
    ) -> None:
        """Initialize an unfitted SGDRegression model."""
        self.coef_: WeightVector | None = None
        self.intercept_: np.float64 | None = None
        self.__batch_size = batch_size
        self.__learning_rate = learning_rate
        self.__epochs = epochs

    def fit(self, X_train: FeatureMatrix, y_train: TargetVector) -> "SGDRegression":
        """Fit the model using mini-batch gradient descent.

        Parameters
        ----------
        X_train : array-like of shape (n_samples, n_features)
        y_train : array-like of shape (n_samples,)

        Returns
        -------
        self : GDRegressor
        """
        X_train = np.asarray(X_train, dtype=np.float64)
        y_train = np.asarray(y_train, dtype=np.float64)

        # Zero Initialization
        self.intercept_ = 0.0
        self.coef_ = np.ones(X_train.shape[1])

        n_samples = X_train.shape[0]

        for _ in range(self.__epochs):
            # Shuffle X and y together to preserve correspondence
            indices = np.random.permutation(n_samples)
            X_shuffled = X_train[indices]
            y_shuffled = y_train[indices]

            n_minibatches = int(np.ceil(n_samples / self.__batch_size))

            for j in range(n_minibatches):
                # Calculate mini-batch start and end indices
                start = j * self.__batch_size
                end = start + self.__batch_size

                X_mini = X_shuffled[start:end]
                y_mini = y_shuffled[start:end]

                y_pred = X_mini @ self.coef_ + self.intercept_

                intercept_der = -2 * np.mean(y_mini - y_pred)
                self.intercept_ -= self.__learning_rate * intercept_der

                coef_der = -2 * (X_mini.T @ (y_mini - y_pred)) / X_mini.shape[0]
                self.coef_ -= self.__learning_rate * coef_der

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
