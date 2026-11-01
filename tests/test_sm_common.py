"""Tests for modules/sm_common.py: the seed convention and honest error bars."""
import numpy as np
import pytest

from modules import sm_common as sc


def test_resolve_seed_is_concrete_and_reproducible():
    assert sc.resolve_seed(7) == 7
    a, b = sc.resolve_seed(None), sc.resolve_seed(None)
    assert isinstance(a, int) and 0 <= a < 2 ** 32 and a != b                  # two fresh seeds differ (overwhelmingly)
    with pytest.raises(sc.StatMechError):
        sc.resolve_seed(-1)


def test_spawned_generators_are_independent_and_reproducible():
    r1, r2 = sc.spawn_rngs(5, 3), sc.spawn_rngs(5, 3)
    draws = [g.random() for g in r1]
    assert draws == [g.random() for g in r2] and len(set(draws)) == 3
    # a point's stream does not depend on how many other points there are
    assert sc.spawn_rngs(5, 3)[1].random() == sc.spawn_rngs(5, 8)[1].random()


def test_autocorrelation_time_of_white_noise_is_one_half():
    x = np.random.default_rng(0).normal(size=20000)
    assert sc.integrated_autocorrelation_time(x) == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("phi", [0.5, 0.9])
def test_autocorrelation_time_of_an_ar1_process_matches_theory(phi):
    """tau_int = (1/2)(1 + phi)/(1 - phi) for x_t = phi x_{t-1} + noise."""
    rng = np.random.default_rng(1)
    n = 400000
    x = np.empty(n)
    x[0] = 0.0
    e = rng.normal(size=n)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    want = 0.5 * (1 + phi) / (1 - phi)
    assert sc.integrated_autocorrelation_time(x) == pytest.approx(want, rel=0.15)


def test_short_or_constant_series_return_the_uncorrelated_value():
    assert sc.integrated_autocorrelation_time([1.0] * 100) == 0.5
    assert sc.integrated_autocorrelation_time([1.0, 2.0, 3.0]) == 0.5


def test_error_bar_grows_with_correlation():
    rng = np.random.default_rng(2)
    white = rng.normal(size=20000)
    corr = np.convolve(white, np.ones(20) / 20, mode="same")
    assert sc.mean_with_error(corr)[1] / corr.std() > sc.mean_with_error(white)[1] / white.std() * 2


def test_error_bar_covers_the_true_mean_for_correlated_data():
    """Over many AR(1) series the 1-sigma interval should contain the true mean about 68% of the time (not ~30%, as a naive error would)."""
    rng = np.random.default_rng(3)
    hits = 0
    trials = 200
    for _ in range(trials):
        x = np.empty(3000)
        x[0] = 0.0
        e = rng.normal(size=3000)
        for i in range(1, 3000):
            x[i] = 0.8 * x[i - 1] + e[i]
        m, err, _ = sc.mean_with_error(x)
        hits += abs(m) < err
    assert 0.55 < hits / trials < 0.85


def test_jackknife_reproduces_a_linear_function_and_a_known_error():
    rng = np.random.default_rng(4)
    x = rng.normal(2.0, 1.0, 6400)
    est, err = sc.jackknife({"x": x}, lambda d: d["x"], 32)
    assert est == pytest.approx(x.mean()) and err == pytest.approx(1.0 / 80, rel=0.4)
    est2, _ = sc.jackknife({"x": x, "y": x ** 2}, lambda d: d["y"] - d["x"] ** 2, 32)       # a nonlinear estimator: the variance
    assert est2 == pytest.approx(x.var(), rel=1e-9)


def test_jackknife_needs_data():
    with pytest.raises(sc.StatMechError):
        sc.jackknife({"x": np.array([])}, lambda d: d["x"], 8)
