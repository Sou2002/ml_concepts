"""Shared pytest fixtures for transparentml tests."""

import numpy as np
import pytest
from sklearn.datasets import make_blobs, make_classification, make_regression


@pytest.fixture
def regression_data():
    """Synthetic linear regression data with known ground truth."""
    X, y, true_coef = make_regression(
        n_samples=200,
        n_features=3,
        noise=1.0,
        coef=True,
        random_state=42,
    )
    return X, y, true_coef


@pytest.fixture
def simple_regression_data():
    """Single-feature regression data, easy to reason about."""
    X, y, true_coef = make_regression(
        n_samples=100,
        n_features=1,
        noise=0.5,
        coef=True,
        random_state=0,
    )
    return X, y, true_coef


@pytest.fixture
def classification_data():
    """Well-separated binary classification data."""
    X, y = make_classification(
        n_samples=300,
        n_features=2,
        n_informative=2,
        n_redundant=0,
        n_clusters_per_class=1,
        class_sep=2.5,
        random_state=42,
    )
    return X, y


@pytest.fixture
def blob_data():
    """Well-separated synthetic clusters for testing."""
    X, y_true = make_blobs(
        n_samples=300,
        centers=3,
        n_features=2,
        cluster_std=0.5,
        random_state=0,
    )
    return X, y_true