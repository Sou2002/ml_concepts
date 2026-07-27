"""Tests for KNN."""

import numpy as np

from transparentml.neighbors.knn import KNN


def test_fit_returns_self(classification_data):
    X, y = classification_data
    model = KNN(n_neighbours=5)
    result = model.fit(X, y)
    assert result is model


def test_predict_output_shape(classification_data):
    X, y = classification_data
    model = KNN(n_neighbours=5).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_accuracy_above_chance(classification_data):
    X, y = classification_data
    model = KNN(n_neighbours=5).fit(X, y)
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy > 0.9


def test_uses_self_x_train_not_global():
    """Regression test: catches the earlier bug where the global X_train was
    referenced instead of self.X_train."""
    X_train_data = np.array([[0.0, 0.0], [10.0, 10.0]])
    y_train_data = np.array([0, 1])

    model = KNN(n_neighbours=1).fit(X_train_data, y_train_data)

    # Predict a point clearly closer to the second training point.
    prediction = model.predict(np.array([[9.0, 9.0]]))
    assert prediction[0] == 1


def test_k_equals_1_matches_nearest_point_label():
    """With n_neighbours=1, prediction should exactly match the single nearest point's label."""
    X_train_data = np.array([[0.0], [5.0], [10.0]])
    y_train_data = np.array([0, 1, 2])

    model = KNN(n_neighbours=1).fit(X_train_data, y_train_data)
    predictions = model.predict(np.array([[0.5], [4.5], [9.9]]))

    assert list(predictions) == [0, 1, 2]