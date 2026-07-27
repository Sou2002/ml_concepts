"""Tests for LogisticRegression."""

import numpy as np

from transparentml.linear_models.logistic_regression import LogisticRegression


def test_fit_returns_self(classification_data):
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.1, epochs=500)
    result = model.fit(X, y)
    assert result is model


def test_predict_proba_range(classification_data):
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.1, epochs=500).fit(X, y)
    probabilities = model.predict_proba(X)
    assert np.all(probabilities >= 0) and np.all(probabilities <= 1)


def test_predict_output_shape_and_labels(classification_data):
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.1, epochs=500).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)
    assert set(np.unique(predictions)).issubset({0, 1})


def test_accuracy_above_chance(classification_data):
    """Correctness test: on well-separated data, accuracy should be high."""
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.1, epochs=1000).fit(X, y)
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy > 0.9


def test_predict_matches_thresholded_proba(classification_data):
    """API contract: predict() should exactly match thresholding predict_proba() at 0.5."""
    X, y = classification_data
    model = LogisticRegression(learning_rate=0.1, epochs=500).fit(X, y)
    proba = model.predict_proba(X)
    expected = np.where(proba > 0.5, 1, 0)
    assert np.array_equal(model.predict(X), expected)