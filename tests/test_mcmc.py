"""Posterior sampling: right against an exact answer, honest about convergence, careful with priors and with unbounded posteriors."""
import warnings

import numpy as np
import pytest

from modules import mcmc
from modules import mcmc_plots as mp
from modules.mcmc import McmcError, Prior


@pytest.fixture(scope="module")
def line():
    """A straight-line fit with known noise: the posterior is exactly Gaussian, so the sampler can be checked against it."""
    rng = np.random.default_rng(1)
    xs = np.linspace(0, 10, 40)
    sigma = 0.7
    ys = 1.5 * xs + 2.0 + rng.normal(0, sigma, xs.shape)
    A = np.column_stack([xs, np.ones_like(xs)])
    exact_cov = sigma ** 2 * np.linalg.inv(A.T @ A)
    exact_mean = np.linalg.solve(A.T @ A, A.T @ ys)
    result = mcmc.sample_posterior(xs, ys, "a*x + b", ["a", "b"], p0=[1, 1], sigma=sigma, steps=6000, seed=3)
    return result, exact_mean, exact_cov


@pytest.fixture(scope="module")
def demos():
    out = {}
    for name in mcmc.DEMOS:
        d = mcmc.demo_dataset(name, seed=2)
        out[name] = (d, mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], p0=d["p0"], priors=d["priors"], seed=1))
    return out


# ------------------------------------------------------------------------------------------ correctness

def test_the_sampler_reproduces_the_exact_gaussian_posterior(line):
    r, mean, cov = line
    assert np.allclose(r.samples.mean(axis=0), mean, atol=0.05 * np.sqrt(np.diag(cov)))
    assert np.allclose(r.samples.std(axis=0), np.sqrt(np.diag(cov)), rtol=0.05)
    exact_corr = cov[0, 1] / np.sqrt(cov[0, 0] * cov[1, 1])
    assert r.correlation()[0, 1] == pytest.approx(exact_corr, abs=0.03)


def test_the_run_is_reported_converged_and_the_predictive_check_passes(line):
    r, _, _ = line
    assert r.converged and not r.warnings and r.predictive_coverage > 0.85
    assert 0.2 < r.acceptance < 0.8 and np.all(r.rhat < 1.1) and np.all(r.ess > 400)


def test_the_noise_level_is_recovered_when_it_is_sampled():
    rng = np.random.default_rng(5)
    xs = np.linspace(0, 10, 200)
    ys = 1.5 * xs + 2.0 + rng.normal(0, 0.9, xs.shape)
    r = mcmc.sample_posterior(xs, ys, "a*x + b", ["a", "b"], p0=[1, 1], steps=5000, seed=2)
    s = next(x for x in r.summaries if x.name == "sigma")
    assert r.sigma_sampled and r.names[-1] == "sigma" and s.lo95 < 0.9 < s.hi95 and s.median == pytest.approx(0.9, rel=0.1)


def test_the_true_parameters_lie_inside_the_95_percent_intervals_of_every_demo(demos):
    for name, (d, r) in demos.items():
        for summary, truth in zip(r.summaries, d["truth"]):
            assert summary.lo95 <= truth <= summary.hi95, f"{name}: {summary.name}"


def test_every_demo_converges_at_the_default_length_and_covers_its_data(demos):
    for name, (d, r) in demos.items():
        assert r.converged, name
        assert 0.8 <= r.predictive_coverage, name


def test_a_saturating_curve_is_flagged_as_non_gaussian_so_a_plus_minus_would_mislead(demos):
    _, r = demos["Michaelis-Menten (saturating)"]
    assert {s.name for s in r.summaries if s.non_gaussian} >= {"V", "K"}
    assert any("non-Gaussian" in w for w in r.warnings)


def test_the_posterior_is_compared_with_least_squares(demos):
    d, r = demos["Exponential decay with an offset"]
    assert all(s.ls_value is not None and s.ls_se for s in r.summaries[:3])
    assert np.allclose([s.ls_value for s in r.summaries[:3]], r.ls_theta)


# ------------------------------------------------------------------------------------------ honesty about convergence

def test_a_run_that_is_too_short_is_not_called_converged():
    d = mcmc.demo_dataset("Gaussian peak", seed=2)
    r = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], p0=d["p0"], steps=300, seed=1)
    assert not r.converged and any("longer" in w for w in r.warnings)


def _sine():
    xs = np.linspace(0, 6, 60)
    return xs, np.sin(2 * xs) + np.random.default_rng(0).normal(0, 0.05, xs.shape)        # a line cannot describe a sine


def test_a_wrong_model_is_caught_by_the_residual_structure_even_when_sigma_is_sampled():
    xs, ys = _sine()
    r = mcmc.sample_posterior(xs, ys, "a*x + b", ["a", "b"], p0=[0, 0], steps=2000, seed=1)
    assert r.residual_z > 3 and r.residual_lag1 > 0.9
    assert any("correlated along x" in w and "understate" in w for w in r.warnings)


def test_the_predictive_band_cannot_catch_it_when_sigma_is_sampled_but_can_when_sigma_is_given():
    """Documents the limitation: a sampled noise level inflates to absorb the misfit, so the band covers the data anyway."""
    xs, ys = _sine()
    sampled = mcmc.sample_posterior(xs, ys, "a*x + b", ["a", "b"], p0=[0, 0], steps=2000, seed=1)
    given = mcmc.sample_posterior(xs, ys, "a*x + b", ["a", "b"], p0=[0, 0], sigma=0.05, steps=2000, seed=1)
    assert sampled.predictive_coverage > 0.9
    assert given.predictive_coverage < 0.5 and any("does not describe these data" in w for w in given.warnings)


def test_well_specified_models_show_no_residual_structure(demos):
    for name, (d, r) in demos.items():
        assert abs(r.residual_z) < 3, name


def test_an_unbounded_posterior_is_diagnosed_not_reported_as_a_result():
    """Saturating data seen far from saturation fix only V/K; with flat priors the tail is infinite."""
    d = mcmc.demo_dataset("Michaelis-Menten (saturating)", seed=2)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                         # and it must not emit numpy overflow warnings
        r = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], p0=d["p0"], steps=2000, seed=1)
    assert set(r.unbounded) == {"V", "K"} and not r.converged
    assert "do not bound V, K" in r.warnings[0] and "prior bound" in r.warnings[0]


def test_bounding_the_prior_makes_the_same_problem_well_posed():
    d = mcmc.demo_dataset("Michaelis-Menten (saturating)", seed=2)
    r = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], p0=d["p0"], priors=d["priors"], steps=6000, seed=1)
    assert not r.unbounded and r.converged


def test_the_autocorrelation_time_of_a_known_process_is_estimated_correctly():
    phi, n = 0.9, 20000
    rng = np.random.default_rng(0)
    chain = np.zeros((n, 8, 1))
    noise = rng.normal(size=(n, 8))
    for t in range(1, n):
        chain[t, :, 0] = phi * chain[t - 1, :, 0] + noise[t]
    tau, reliable = mcmc.integrated_time(chain)
    assert reliable and tau[0] == pytest.approx((1 + phi) / (1 - phi), rel=0.2)                  # 19


def test_a_chain_too_short_for_its_correlation_is_marked_unreliable_and_not_mixed():
    chain = np.cumsum(np.random.default_rng(0).normal(size=(200, 4, 1)), axis=0)               # a random walk
    assert mcmc.integrated_time(chain)[1] is False and mcmc.split_rhat(chain)[0] > 1.5


def test_split_rhat_is_one_for_independent_chains_and_large_for_disagreeing_ones():
    rng = np.random.default_rng(0)
    good = rng.normal(size=(2000, 8, 1))
    bad = good + np.arange(8)[None, :, None] * 3.0
    assert mcmc.split_rhat(good)[0] == pytest.approx(1.0, abs=0.02) and mcmc.split_rhat(bad)[0] > 1.5


# ------------------------------------------------------------------------------------------ priors

def test_a_uniform_prior_is_never_violated_and_a_normal_prior_pulls_the_estimate():
    rng = np.random.default_rng(2)
    xs = np.linspace(0, 10, 8)
    ys = 1.0 * xs + rng.normal(0, 2.0, xs.shape)
    bounded = mcmc.sample_posterior(xs, ys, "a*x", ["a"], p0=[1.0], priors={"a": Prior("uniform", 0.5, 1.1)}, steps=2500, seed=1)
    assert bounded.samples[:, 0].min() >= 0.5 and bounded.samples[:, 0].max() <= 1.1
    flat = mcmc.sample_posterior(xs, ys, "a*x", ["a"], p0=[1.0], steps=2500, seed=1)
    pulled = mcmc.sample_posterior(xs, ys, "a*x", ["a"], p0=[1.0], priors={"a": Prior("normal", 3.0, 0.05)}, steps=2500, seed=1)
    assert abs(pulled.summaries[0].median - 3.0) < abs(flat.summaries[0].median - 3.0)


def test_a_positive_prior_keeps_a_rate_positive():
    d = mcmc.demo_dataset("Exponential decay with an offset", seed=2)
    r = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], p0=d["p0"], priors={"b": Prior("positive")}, steps=2000, seed=1)
    assert r.samples[:, 1].min() > 0


def test_reproducible_given_the_seed_and_different_for_another(line):
    d = mcmc.demo_dataset("Gaussian peak", seed=2)
    kw = dict(p0=d["p0"], steps=800)
    a = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], seed=4, **kw)
    b = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], seed=4, **kw)
    c = mcmc.sample_posterior(d["xs"], d["ys"], d["expr"], d["params"], seed=5, **kw)
    assert np.array_equal(a.samples, b.samples) and not np.array_equal(a.samples, c.samples)


# ------------------------------------------------------------------------------------------ predictive and tables

def test_predictive_bands_are_ordered_and_contain_the_median(line):
    r, _, _ = line
    grid = np.linspace(0, 10, 25)
    centre, b = r.predictive(grid)
    assert np.all(b[0] <= b[1]) and np.all(b[1] <= centre + 1e-9) and np.all(centre <= b[2] + 1e-9) and np.all(b[2] <= b[3])
    narrow, nb = r.predictive(grid, include_noise=False)
    assert np.mean(nb[3] - nb[0]) < np.mean(b[3] - b[0])


def test_the_summary_table_lists_every_parameter_with_asymmetric_errors(demos):
    d, r = demos["Logistic growth"]
    rows = r.summary_table()
    assert [row["parameter"] for row in rows] == ["K", "r", "x0", "sigma"] and all(row["−1σ (16%)"] < 0 < row["+1σ (84%)"] for row in rows)


# ------------------------------------------------------------------------------------------ input validation

@pytest.mark.parametrize("kwargs, message", [
    (dict(expr="a*x + z", params=["a"]), "not listed"),
    (dict(expr="a*x", params=["a", "b"]), "do not appear"),
    (dict(expr="a*x", params=["x"]), "cannot be a parameter"),
    (dict(expr="s*x", params=["sigma"]), "reserved"),
    (dict(expr="a*x", params=["a", "a"]), "unique"),
    (dict(expr="a*x +", params=["a"]), "Could not parse"),
    (dict(expr="a*x", params=["a"], steps=10), "Steps"),
    (dict(expr="a*x", params=["a"], burn_fraction=0.99), "burn-in"),
    (dict(expr="a*x", params=["a"], sigma=-1.0), "sigma must be positive"),
    (dict(expr="a*x", params=["a"], walkers=7), "even"),
    (dict(expr="a*x", params=["a"], priors={"a": Prior("uniform", 2, 1)}), "lower < upper"),
    (dict(expr="a*x", params=["a"], priors={"a": Prior("normal", 0, -1)}), "standard deviation"),
    (dict(expr="a*x", params=["a"], p0=[1, 2]), "initial guess"),
])
def test_bad_inputs_are_refused_with_a_reason(kwargs, message):
    base = dict(xs=np.linspace(0, 5, 20), ys=np.linspace(0, 10, 20), steps=200)
    with pytest.raises(McmcError, match=message):
        mcmc.sample_posterior(**{**base, "expr": kwargs.pop("expr"), "param_names": kwargs.pop("params"), **kwargs})


def test_bad_data_is_refused():
    with pytest.raises(McmcError, match="at least"):
        mcmc.sample_posterior([1, 2], [1, 2], "a*x", ["a"])
    with pytest.raises(McmcError, match="non-finite"):
        mcmc.sample_posterior([1, 2, 3, 4, 5], [1, 2, float("nan"), 4, 5], "a*x", ["a"])
    with pytest.raises(McmcError, match="same length"):
        mcmc.sample_posterior([1, 2, 3, 4, 5], [1, 2, 3], "a*x", ["a"])
    with pytest.raises(McmcError, match="Unknown demo"):
        mcmc.demo_dataset("nope")


def test_a_prior_that_excludes_the_fit_is_explained():
    xs = np.linspace(1, 5, 20)
    with pytest.raises(McmcError, match="priors may exclude"):
        mcmc.sample_posterior(xs, 3.0 * xs, "a*x", ["a"], p0=[3.0], priors={"a": Prior("uniform", 10.0, 20.0)}, sigma=0.1, steps=200)


# ------------------------------------------------------------------------------------------ figures

def test_the_figures_build(demos):
    d, r = demos["Exponential decay with an offset"]
    corner = mp.corner_figure(r, truth=d["truth"])
    assert sum(1 for t in corner.data if t.type == "histogram") == len(r.names)
    assert sum(1 for t in corner.data if t.type == "histogram2dcontour") == len(r.names) * (len(r.names) - 1) // 2
    assert len(mp.trace_figure(r).data) >= len(r.names)
    pred = mp.predictive_figure(r)
    assert {t.name for t in pred.data} >= {"95% predictive", "68% predictive", "posterior median", "data"}
