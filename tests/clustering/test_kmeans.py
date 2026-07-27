"""Tests for KMeans."""

import numpy as np

from transparentml.clustering.kmeans import KMeans


def test_fit_predict_output_shape(blob_data):
    X, _ = blob_data
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    assert labels.shape == (X.shape[0],)


def test_cluster_centers_shape(blob_data):
    X, _ = blob_data
    model = KMeans(n_clusters=3, random_state=42)
    model.fit_predict(X)
    assert model.cluster_centers_.shape == (3, X.shape[1])


def test_labels_attribute_matches_return_value(blob_data):
    X, _ = blob_data
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    assert np.array_equal(labels, model.labels_)


def test_recovers_correct_number_of_clusters(blob_data):
    X, _ = blob_data
    model = KMeans(n_clusters=3, random_state=42)
    labels = model.fit_predict(X)
    assert len(np.unique(labels)) == 3


def test_reproducible_with_same_random_state(blob_data):
    X, _ = blob_data
    labels_a = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    labels_b = KMeans(n_clusters=3, random_state=42).fit_predict(X)
    assert np.array_equal(labels_a, labels_b)


def test_clustering_quality_against_ground_truth(blob_data):
    """Purity check: each predicted cluster should be dominated by one true label,
    since KMeans label numbering is arbitrary relative to ground truth."""
    X, y_true = blob_data
    model = KMeans(n_clusters=3, random_state=0)
    y_pred = model.fit_predict(X)

    for cluster_id in np.unique(y_pred):
        true_labels_in_cluster = y_true[y_pred == cluster_id]
        purity = np.bincount(true_labels_in_cluster).max() / len(true_labels_in_cluster)
        assert purity > 0.9