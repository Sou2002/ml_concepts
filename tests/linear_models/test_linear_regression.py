"""Tests for LinearRegression."""

import numpy as np

from transparentml.linear_models.linear_regression import LinearRegression


def test_fit_returns_self(regression_data):
    X, y, _ = regression_data
    model = LinearRegression()
    result = model.fit(X, y)
    assert result is model


def test_attributes_set_after_fit(regression_data):
    X, y, _ = regression_data
    model = LinearRegression().fit(X, y)
    assert model.coef_ is not None
    assert model.intercept_ is not None
    assert model.coef_.shape == (X.shape[1],)


def test_predict_output_shape(regression_data):
    X, y, _ = regression_data
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_recovers_true_coefficients(regression_data):
    """Correctness test: with low noise, fitted coef_ should be close to true_coef."""
    X, y, true_coef = regression_data
    model = LinearRegression().fit(X, y)
    assert np.allclose(model.coef_, true_coef, atol=0.5)


def test_predictions_close_to_targets(simple_regression_data):
    """Sanity check: predictions on training data should track y reasonably well."""
    X, y, _ = simple_regression_data
    model = LinearRegression().fit(X, y)
    predictions = model.predict(X)
    residuals = y - predictions
    assert np.mean(residuals ** 2) < 5.0  # generous MSE bound given noise=0.5


def test_accepts_list_input():
    """API contract: should accept plain Python lists, not just ndarrays."""
    X = [[1.0], [2.0], [3.0], [4.0]]
    y = [2.0, 4.0, 6.0, 8.0]
    model = LinearRegression().fit(X, y)
    predictions = model.predict([[5.0]])
    assert predictions.shape == (1,)
    assert np.isclose(predictions[0], 10.0, atol=1.0)


def test_single_feature_perfect_fit():
    """Edge case: noiseless, perfectly linear data should be fit almost exactly."""
    X = np.array([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = np.array([3.0, 5.0, 7.0, 9.0, 11.0])  # y = 2x + 1
    model = LinearRegression().fit(X, y)
    assert np.isclose(model.coef_[0], 2.0, atol=1e-6)
    assert np.isclose(model.intercept_, 1.0, atol=1e-6)