"""Tests for DecisionTree."""

import numpy as np

from transparentml.tree.decision_tree import DecisionTree


def test_fit_returns_self(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=3)
    result = model.fit(X, y)
    assert result is model


def test_predict_output_shape(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=3).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_accuracy_above_chance(classification_data):
    X, y = classification_data
    model = DecisionTree(max_depth=5).fit(X, y)
    predictions = model.predict(X)
    accuracy = (predictions == y).mean()
    assert accuracy > 0.9


def test_deeper_tree_fits_training_data_better():
    """Sanity check: deeper trees should fit training data at least as well
    as shallow trees (classic overfitting-capacity behavior)."""
    from sklearn.datasets import make_classification
    X, y = make_classification(n_samples=200, n_features=5, random_state=1)

    shallow = DecisionTree(max_depth=1).fit(X, y)
    deep = DecisionTree(max_depth=8).fit(X, y)

    shallow_acc = (shallow.predict(X) == y).mean()
    deep_acc = (deep.predict(X) == y).mean()

    assert deep_acc >= shallow_acc


def test_single_class_data_returns_that_class():
    """Edge case: if all training labels are identical, tree should always predict it."""
    X = np.random.rand(20, 2)
    y = np.zeros(20)
    model = DecisionTree(max_depth=3).fit(X, y)
    predictions = model.predict(X)
    assert np.all(predictions == 0)