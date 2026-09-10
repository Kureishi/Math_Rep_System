"""Tests for modules/statistical_inference.py -- the statistical/data
layer on top of curve_fitting.py: parameter inference, residual
diagnostics, Bayesian regression, and nested-model comparison.

Several tests below check against SPECIFIC numeric values that were
cross-validated against statsmodels during development (see this
module's docstring) -- statsmodels itself is not a test dependency;
these are just the reference numbers it produced, hardcoded here so the
cross-check doesn't require installing it."""
import numpy as np
import pytest

from modules.statistical_inference import (
    regression_inference, residual_diagnostics, bayesian_linear_regression,
    compare_polynomial_degrees,
)


@pytest.fixture
def linear_data():
    """y = 2 + 3x + noise, seeded for reproducibility -- the exact
    dataset used for the statsmodels cross-validation during
    development (see the module docstring), so results here should
    match those hardcoded reference numbers exactly."""
    rng = np.random.RandomState(42)
    xs = np.linspace(0, 10, 15)
    ys = 2.0 + 3.0 * xs + rng.normal(0, 2, size=15)
    return list(xs), list(ys)


# --------------------------------------------------------------------- regression_inference
def test_regression_inference_matches_statsmodels_reference(linear_data):
    xs, ys = linear_data
    result = regression_inference(xs, ys, "polynomial", degree=1)
    assert result.error is None
    by_name = {p.name: p for p in result.parameters}
    assert abs(by_name["c1"].estimate - 2.62631725) < 1e-6
    assert abs(by_name["c1"].std_error - 0.1380615) < 1e-5
    assert abs(by_name["c0"].estimate - 3.88911082) < 1e-6
    assert abs(result.f_statistic - 361.867193) < 1e-3
    assert abs(result.adjusted_r_squared - 0.962653440) < 1e-6
    assert result.degrees_of_freedom == 13


def test_regression_inference_confidence_interval_widens_with_confidence_level(linear_data):
    xs, ys = linear_data
    r95 = regression_inference(xs, ys, "polynomial", degree=1, confidence=0.95)
    r99 = regression_inference(xs, ys, "polynomial", degree=1, confidence=0.99)
    p95 = next(p for p in r95.parameters if p.name == "c1")
    p99 = next(p for p in r99.parameters if p.name == "c1")
    width_95 = p95.ci_upper - p95.ci_lower
    width_99 = p99.ci_upper - p99.ci_lower
    assert width_99 > width_95


def test_regression_inference_exponential_ci_back_transform_is_monotonic():
    """The 'a' parameter's CI is computed in log-space and mapped
    through exp() -- since exp is monotonic, the LOWER log-space bound
    must map to the LOWER a-space bound (no accidental swap)."""
    rng = np.random.RandomState(7)
    xs_arr = np.linspace(1, 10, 12)
    ys_arr = 2.0 * np.exp(0.3 * xs_arr) * np.exp(rng.normal(0, 0.1, size=12))
    result = regression_inference(list(xs_arr), list(ys_arr), "exponential")
    assert result.error is None
    a_param = next(p for p in result.parameters if p.name == "a")
    assert a_param.ci_lower < a_param.estimate < a_param.ci_upper
    assert a_param.ci_lower > 0  # exp() of anything is positive -- CI must stay in valid range


def test_regression_inference_not_enough_data_reports_error():
    result = regression_inference([1, 2], [1, 2], "polynomial", degree=5)
    assert result.error is not None


def test_regression_inference_single_parameter_skips_f_test():
    """A model with only one parameter (e.g. y = a*x, no intercept via
    a custom fit) has no non-intercept-vs-flat-mean comparison to make;
    the F-test must be reported as unavailable, not a bogus number."""
    result = regression_inference([1, 2, 3, 4, 5], [2, 4, 6, 8, 10], "custom",
                                    expr_str="a*x", param_names=["a"])
    assert result.error is None
    assert result.f_statistic is None


# --------------------------------------------------------------------- residual_diagnostics
def test_residual_diagnostics_matches_reference_values(linear_data):
    xs, ys = linear_data
    result = residual_diagnostics(xs, ys, "polynomial", degree=1)
    assert result.error is None
    assert result.durbin_watson is not None
    assert result.normality_p_value is not None
    assert result.heteroscedasticity_p_value is not None
    # well-behaved simulated normal-noise data shouldn't trip any of the three checks
    assert result.normality_p_value > 0.05
    assert 1.0 < result.durbin_watson < 3.0
    assert result.heteroscedasticity_p_value > 0.05


def test_residual_diagnostics_not_enough_points():
    result = residual_diagnostics([1, 2, 3], [1, 2, 3], "polynomial", degree=1)
    assert result.error is not None


def test_residual_diagnostics_detects_strong_positive_autocorrelation():
    """Residuals that trend smoothly (rather than bouncing randomly
    around the fit) should show low Durbin-Watson -- this checks the
    statistic actually responds to the pattern it's meant to detect,
    not just that it returns SOME number."""
    xs = list(range(20))
    # a smooth sine wiggle on top of a straight line: residuals will be
    # strongly autocorrelated since nearby points share the same wiggle phase
    ys = [2 * x + 5 * np.sin(x / 2) for x in xs]
    result = residual_diagnostics(xs, ys, "polynomial", degree=1)
    assert result.error is None
    assert result.durbin_watson < 1.0


# --------------------------------------------------------------------- bayesian_linear_regression
def test_bayesian_posterior_converges_to_ols_under_diffuse_prior(linear_data):
    """The central correctness claim of this module's Bayesian layer:
    with a very weak (diffuse) prior, the posterior mean must closely
    track the OLS estimate -- this is a mathematical consequence of the
    conjugate update as prior precision -> 0, not a coincidence, so a
    real implementation bug (wrong formula) would show up as a
    noticeable mismatch here."""
    xs, ys = linear_data
    freq = regression_inference(xs, ys, "polynomial", degree=1)
    bayes = bayesian_linear_regression(xs, ys, "polynomial", degree=1, prior_precision=1e-6)
    assert bayes.error is None
    freq_by_name = {p.name: p for p in freq.parameters}
    bayes_by_name = {p.name: p for p in bayes.parameters}
    for name in ("c0", "c1"):
        assert abs(freq_by_name[name].estimate - bayes_by_name[name].posterior_mean) < 1e-3


def test_bayesian_posterior_predictive_interval_contains_point_estimate(linear_data):
    xs, ys = linear_data
    bayes = bayesian_linear_regression(xs, ys, "polynomial", degree=1)
    assert bayes.error is None
    mean, lo, hi = bayes.posterior_predictive_fn(np.array([5.0, 1.0]))
    assert lo < mean < hi


def test_bayesian_stronger_prior_pulls_estimate_toward_zero():
    """A much larger prior_precision should visibly shrink the
    posterior mean toward the prior mean of zero, relative to the
    diffuse-prior case -- confirms the prior is actually doing
    something, not being silently ignored. Uses a single-parameter,
    no-intercept model (y = a*x) specifically to avoid the well-known
    ridge-regression effect where shrinking a CORRELATED intercept and
    slope together can push one estimate away from zero to compensate
    (confirmed separately during development with an off-center
    two-parameter fit, where shrinkage is real but not simply
    coefficient-by-coefficient monotonic) -- with one orthogonal
    parameter there's nothing to compensate against, so the shrinkage
    must be unambiguous."""
    xs = [1, 2, 3, 4, 5, 6, 7, 8]
    ys = [3.1, 5.9, 9.2, 11.8, 15.3, 17.9, 21.2, 23.8]  # y ~ 3*x
    diffuse = bayesian_linear_regression(xs, ys, "custom", expr_str="a*x", param_names=["a"],
                                           prior_precision=1e-6)
    strong = bayesian_linear_regression(xs, ys, "custom", expr_str="a*x", param_names=["a"],
                                          prior_precision=10.0)
    diffuse_a = next(p for p in diffuse.parameters if p.name == "a").posterior_mean
    strong_a = next(p for p in strong.parameters if p.name == "a").posterior_mean
    assert abs(strong_a) < abs(diffuse_a)


def test_bayesian_not_enough_data_reports_error():
    result = bayesian_linear_regression([1, 2], [1, 2], "polynomial", degree=5)
    assert result.error is not None


# --------------------------------------------------------------------- compare_polynomial_degrees
def test_nested_model_comparison_matches_reference_value():
    rng = np.random.RandomState(1)
    xs = np.linspace(-3, 3, 25)
    ys = 1.0 + 0.5 * xs + 2.0 * xs ** 2 + rng.normal(0, 3, size=25)
    result = compare_polynomial_degrees(list(xs), list(ys), reduced_degree=1, full_degree=2)
    assert result.error is None
    assert abs(result.f_statistic - 103.0965) < 0.01
    assert abs(result.p_value - 9.140346e-10) < 1e-12
    assert result.significant


def test_nested_model_comparison_detects_no_improvement():
    """Genuinely linear data: adding a quadratic term should NOT
    significantly improve the fit -- the test must correctly report
    'not significant' rather than always finding significance just
    because R-squared went up by some tiny amount (which it always
    does with more parameters)."""
    rng = np.random.RandomState(3)
    xs = np.linspace(0, 10, 30)
    ys = 5.0 + 2.0 * xs + rng.normal(0, 1.5, size=30)
    result = compare_polynomial_degrees(list(xs), list(ys), reduced_degree=1, full_degree=2)
    assert result.error is None
    assert not result.significant
    assert result.p_value > 0.05


def test_nested_model_comparison_requires_full_degree_greater():
    result = compare_polynomial_degrees([1, 2, 3, 4, 5], [1, 2, 3, 4, 5],
                                          reduced_degree=2, full_degree=1)
    assert result.error is not None


def test_nested_model_comparison_not_enough_data():
    result = compare_polynomial_degrees([1, 2, 3], [1, 2, 3], reduced_degree=1, full_degree=2)
    assert result.error is not None
