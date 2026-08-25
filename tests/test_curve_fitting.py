import numpy as np
import pytest
import sympy as sp

from modules.curve_fitting import (
    fit_curve, fit_linear, fit_polynomial, fit_exponential, fit_power,
    fit_logarithmic, fit_custom, parse_xy_csv, best_fit, BUILTIN_FAMILIES,
)


# ---------------------------------------------------------------- CSV parsing


def test_parse_csv_with_header():
    xs, ys, xl, yl = parse_xy_csv("time,distance\n1,2\n2,4\n3,6\n")
    assert xs == [1.0, 2.0, 3.0]
    assert ys == [2.0, 4.0, 6.0]
    assert xl == "time" and yl == "distance"


def test_parse_csv_without_header():
    xs, ys, xl, yl = parse_xy_csv("1,2\n2,4\n3,6\n")
    assert xs == [1.0, 2.0, 3.0]
    assert xl == "x" and yl == "y"


def test_parse_csv_too_few_rows_raises():
    with pytest.raises(ValueError, match="at least 2"):
        parse_xy_csv("x,y\n1,2\n")


def test_parse_csv_non_numeric_row_raises():
    with pytest.raises(ValueError, match="isn't numeric"):
        parse_xy_csv("x,y\nabc,2\n1,2\n")


def test_parse_csv_short_row_raises():
    with pytest.raises(ValueError, match="fewer than 2 columns"):
        parse_xy_csv("x,y\n1\n2,4\n")


# ---------------------------------------------------------------- built-in families


def test_fit_linear_recovers_known_line():
    xs = [0, 1, 2, 3, 4]
    ys = [2 * x + 1 for x in xs]
    r = fit_linear(xs, ys)
    assert r.error is None
    assert r.r_squared == pytest.approx(1.0, abs=1e-9)
    assert r.param_values["c1"] == pytest.approx(2.0, abs=1e-6)
    assert r.param_values["c0"] == pytest.approx(1.0, abs=1e-6)


def test_fit_polynomial_recovers_known_quadratic():
    xs = list(range(-3, 4))
    ys = [3 * x ** 2 - 2 * x + 5 for x in xs]
    r = fit_polynomial(xs, ys, degree=2)
    assert r.r_squared == pytest.approx(1.0, abs=1e-9)
    assert r.param_values["c2"] == pytest.approx(3.0, abs=1e-6)
    assert r.param_values["c1"] == pytest.approx(-2.0, abs=1e-6)
    assert r.param_values["c0"] == pytest.approx(5.0, abs=1e-6)


def test_fit_polynomial_needs_enough_points():
    r = fit_polynomial([1, 2], [1, 2], degree=3)
    assert r.error is not None
    assert "at least 4" in r.error


def test_fit_exponential_recovers_known_curve():
    xs = np.linspace(0, 5, 10)
    ys = 2.0 * np.exp(0.5 * xs)
    r = fit_exponential(list(xs), list(ys))
    assert r.r_squared == pytest.approx(1.0, abs=1e-6)
    assert r.param_values["a"] == pytest.approx(2.0, rel=1e-4)
    assert r.param_values["b"] == pytest.approx(0.5, rel=1e-4)


def test_fit_exponential_rejects_nonpositive_y():
    r = fit_exponential([1, 2, 3], [1, -2, 3])
    assert r.error is not None
    assert "positive" in r.error


def test_fit_power_recovers_known_curve():
    xs = np.linspace(1, 10, 10)
    ys = 3.0 * xs ** 1.5
    r = fit_power(list(xs), list(ys))
    assert r.r_squared == pytest.approx(1.0, abs=1e-6)
    assert r.param_values["a"] == pytest.approx(3.0, rel=1e-4)
    assert r.param_values["b"] == pytest.approx(1.5, rel=1e-4)


def test_fit_power_rejects_nonpositive_x():
    r = fit_power([-1, 2, 3], [1, 2, 3])
    assert r.error is not None
    assert "positive" in r.error


def test_fit_logarithmic_recovers_known_curve():
    xs = np.linspace(1, 20, 15)
    ys = 4.0 * np.log(xs) + 2.0
    r = fit_logarithmic(list(xs), list(ys))
    assert r.r_squared == pytest.approx(1.0, abs=1e-6)
    assert r.param_values["a"] == pytest.approx(4.0, rel=1e-4)
    assert r.param_values["b"] == pytest.approx(2.0, rel=1e-4)


def test_fit_logarithmic_rejects_nonpositive_x():
    r = fit_logarithmic([-1, 2, 3], [1, 2, 3])
    assert r.error is not None
    assert "positive" in r.error


# ---------------------------------------------------------------- custom (linear-in-params) fit


def test_fit_custom_linear_in_params_recovers_known_model():
    xs = list(np.linspace(0, 10, 25))
    ys = [2 * np.sin(x) + 3 * x + 1 for x in xs]
    r = fit_custom(xs, ys, "a*sin(x) + b*x + c", ["a", "b", "c"])
    assert r.error is None
    assert r.r_squared == pytest.approx(1.0, abs=1e-6)
    assert r.param_values["a"] == pytest.approx(2.0, abs=1e-4)
    assert r.param_values["b"] == pytest.approx(3.0, abs=1e-4)
    assert r.param_values["c"] == pytest.approx(1.0, abs=1e-4)


def test_fit_custom_rejects_nonlinear_in_parameter():
    xs = list(np.linspace(0, 10, 25))
    ys = [np.sin(0.5 * x) for x in xs]
    r = fit_custom(xs, ys, "a*sin(b*x)", ["a", "b"])
    assert r.error is not None
    assert "linear" in r.error.lower() or "interact" in r.error.lower()


def test_fit_custom_rejects_unknown_symbol():
    r = fit_custom([1, 2, 3], [1, 2, 3], "a*x + d", ["a"])
    assert r.error is not None
    assert "d" in r.error


def test_fit_custom_rejects_unparseable_expression():
    r = fit_custom([1, 2, 3], [1, 2, 3], "a*x +", ["a"])
    assert r.error is not None


def test_fit_custom_needs_enough_points():
    r = fit_custom([1, 2], [1, 2], "a*x + b*x**2 + c*x**3", ["a", "b", "c"])
    assert r.error is not None
    assert "at least 3" in r.error


# ---------------------------------------------------------------- dispatch + best_fit


def test_fit_curve_dispatches_by_family_name():
    xs = [1, 2, 3, 4]
    ys = [2, 4, 6, 8]
    r = fit_curve(xs, ys, "linear")
    assert r.family == "linear"
    assert r.r_squared == pytest.approx(1.0, abs=1e-9)


def test_fit_curve_unknown_family_errors():
    r = fit_curve([1, 2], [1, 2], "quadratic-ish")
    assert r.error is not None


def test_fit_curve_mismatched_lengths_errors():
    r = fit_curve([1, 2, 3], [1, 2], "linear")
    assert r.error is not None
    assert "different lengths" in r.error


def test_best_fit_ranks_and_skips_invalid_families():
    xs = np.linspace(1, 5, 8)
    ys = 2.0 * np.exp(0.3 * xs)
    results = best_fit(list(xs), list(ys))
    # exponential should be at (or extremely near) the top since it's the true model
    ranked = sorted(results.items(), key=lambda kv: kv[1].r_squared, reverse=True)
    assert ranked[0][0] == "exponential"
    assert set(results.keys()) <= set(BUILTIN_FAMILIES)


def test_predict_uses_fitted_expr():
    r = fit_linear([0, 1, 2], [1, 3, 5])  # y = 2x + 1
    assert r.predict(10) == pytest.approx(21.0, abs=1e-6)
