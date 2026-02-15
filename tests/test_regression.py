import numpy as np

from spongebobs_corpse.stats import confidence_band_linear, fit_linear


def test_linear_regression_recovers_slope_intercept() -> None:
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = 2.5 * x + 1.0
    fit = fit_linear(x, y, method="OLS")
    assert abs(fit["slope"] - 2.5) < 1e-10
    assert abs(fit["intercept"] - 1.0) < 1e-10
    assert fit["r2"] > 0.999999


def test_confidence_band_is_ordered() -> None:
    x = np.array([10, 20, 30, 40, 50], dtype=float)
    y = 0.8 * x + 5.0
    fit = fit_linear(x, y, method="OLS")
    x_fit = np.linspace(10, 50, 100)
    lower, upper = confidence_band_linear(x_fit, fit, level=0.95)
    assert lower.shape == upper.shape
    assert np.all(lower <= upper)
