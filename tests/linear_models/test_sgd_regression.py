"""Tests for SGDRegression (mini-batch gradient descent)."""

import numpy as np

from transparentml.linear_models.sgd_regression import SGDRegression


def test_fit_returns_self(regression_data):
    X, y, _ = regression_data
    model = SGDRegression(batch_size=16, learning_rate=0.01, epochs=200)
    result = model.fit(X, y)
    assert result is model


def test_predict_output_shape(regression_data):
    X, y, _ = regression_data
    model = SGDRegression(batch_size=16, learning_rate=0.01, epochs=200).fit(X, y)
    predictions = model.predict(X)
    assert predictions.shape == (X.shape[0],)


def test_converges_close_to_true_coefficients(regression_data):
    """Correctness test: after enough epochs, should approximate true_coef."""
    X, y, true_coef = regression_data
    model = SGDRegression(batch_size=32, learning_rate=0.05, epochs=500).fit(X, y)
    assert np.allclose(model.coef_, true_coef, atol=2.0)


def test_loss_decreases_over_training(simple_regression_data):
    """Sanity check: MSE after training should be much lower than an untrained baseline."""
    X, y, _ = simple_regression_data

    model = SGDRegression(batch_size=10, learning_rate=0.01, epochs=300).fit(X, y)
    trained_mse = np.mean((y - model.predict(X)) ** 2)

    baseline_mse = np.mean((y - np.mean(y)) ** 2)  # predicting the mean every time

    assert trained_mse < baseline_mse


def test_x_and_y_correspondence_preserved():
    X_raw = np.arange(20).reshape(-1, 1).astype(np.float64)
    X = (X_raw - X_raw.mean()) / X_raw.std()  # standardize
    y = (2 * X_raw.flatten() + 1).astype(np.float64)

    model = SGDRegression(batch_size=4, learning_rate=0.01, epochs=300).fit(X, y)
    predictions = model.predict(X)

    assert np.mean((y - predictions) ** 2) < 1.0