"""Equation discovery: that it finds the right equations, says how sure it is, and refuses what it cannot do."""
import math

import numpy as np
import pytest

from modules import discover as dc


def coef_of(result, state, term):
    j = result.names.index(state)
    for i, t in enumerate(result.terms):
        if str(t) == term:
            return result.coef[i, j]
    raise KeyError(term)


def support(result):
    return {(result.names[j], str(result.terms[i])) for i, j in zip(*np.nonzero(result.coef))}


# ------------------------------------------------------------------------------------------ recovering known laws

def test_a_damped_oscillator_is_recovered_exactly_from_clean_data():
    t, X, names, _ = dc.demo_data("Damped oscillator")
    r = dc.fit(t, X, names, degree=2, derivative_method="gradient")
    assert support(r) == {("x", "v"), ("v", "x"), ("v", "v")}
    assert coef_of(r, "x", "v") == pytest.approx(1.0, abs=1e-3)
    assert coef_of(r, "v", "x") == pytest.approx(-4.0, abs=1e-2) and coef_of(r, "v", "v") == pytest.approx(-0.4, abs=1e-2)
    assert dc.recognise(r) == "harmonic oscillator (damped)"


def test_logistic_growth_is_recovered_and_named_with_its_carrying_capacity():
    t, X, names, _ = dc.demo_data("Logistic growth")
    r = dc.fit(t, X, names, degree=2, derivative_method="gradient")
    assert support(r) == {("x", "x"), ("x", "x**2")}
    assert coef_of(r, "x", "x") == pytest.approx(0.8, abs=1e-2) and coef_of(r, "x", "x**2") == pytest.approx(-0.08, abs=2e-3)
    assert "logistic growth" in dc.recognise(r) and "10" in dc.recognise(r)          # K = 0.8 / 0.08


def test_predator_prey_survives_one_percent_noise_with_the_smoother():
    t, X, names, _ = dc.demo_data("Predator-prey (Lotka-Volterra)", noise=0.01, seed=1)
    r = dc.fit(t, X, names, degree=2)
    assert support(r) == {("x", "x"), ("x", "x*y"), ("y", "y"), ("y", "x*y")}
    assert coef_of(r, "x", "x") == pytest.approx(1.0, rel=0.1) and coef_of(r, "y", "x*y") == pytest.approx(0.075, rel=0.1)
    assert dc.recognise(r) == "predator-prey (Lotka-Volterra)"


def test_the_lorenz_structure_is_found():
    t, X, names, _ = dc.demo_data("Lorenz attractor")
    r = dc.fit(t, X, names, degree=2, derivative_method="gradient")
    assert support(r) == {("x", "x"), ("x", "y"), ("y", "x"), ("y", "y"), ("y", "x*z"), ("z", "z"), ("z", "x*y")}
    assert coef_of(r, "x", "y") == pytest.approx(10.0, rel=0.02) and coef_of(r, "y", "x") == pytest.approx(28.0, rel=0.02)
    assert dc.recognise(r) == "Lorenz system"


def test_the_automatic_threshold_picks_the_true_sparsity_and_records_the_path():
    t, X, names, _ = dc.demo_data("Damped oscillator", noise=0.005, seed=3)
    r = dc.fit(t, X, names, degree=3)                        # a 10-term library per equation; 3 terms are real
    assert r.threshold_mode == "auto (BIC)" and r.n_terms == 3
    assert len(r.path) == 30 and min(p.n_terms for p in r.path) <= 3 <= max(p.n_terms for p in r.path)


# ------------------------------------------------------------------------------------------ evidence

def test_a_good_model_simulates_back_onto_the_data_and_is_verified():
    t, X, names, _ = dc.demo_data("Damped oscillator", noise=0.005, seed=3)
    r = dc.fit(t, X, names, degree=2, bootstrap=20)
    err = dc.simulation_error(r)
    assert err < 0.1 and min(r.r2) > 0.95
    assert dc.verdict(r, err)[0] == "verified"


def test_a_library_missing_the_true_term_is_caught_by_the_simulation_check():
    """The pendulum's sin(x) is not in a polynomial library: the derivative fit can look fine, running forward is the test."""
    t = np.linspace(0, 20, 2000)
    from scipy.integrate import solve_ivp
    sol = solve_ivp(lambda _t, s: [s[1], -9.0 * math.sin(s[0])], (0, 20), [2.5, 0.0], t_eval=t, rtol=1e-10, atol=1e-12)
    r = dc.fit(t, sol.y.T, ["x", "v"], degree=2, derivative_method="gradient")
    assert dc.verdict(r, dc.simulation_error(r))[0] in ("caution", "failed")
    r_trig = dc.fit(t, sol.y.T, ["x", "v"], degree=1, trig=True, constant=False, derivative_method="gradient")
    assert coef_of(r_trig, "v", "sin(x)") == pytest.approx(-9.0, rel=0.02)           # and with sin(x) available it is found


def test_bootstrap_inclusion_is_high_for_real_terms_and_reported_per_term():
    t, X, names, _ = dc.demo_data("Logistic growth", noise=0.01, seed=2)
    r = dc.fit(t, X, names, degree=3, bootstrap=30, seed=5)
    assert r.inclusion is not None and r.inclusion.shape == r.coef.shape
    assert coef_of(r, "x", "x") != 0 and r.inclusion[[str(s) for s in r.terms].index("x"), 0] > 0.9
    assert any("optimistic" in n for n in r.notes)
    rows = r.term_table()
    assert all(row["selected in resamples"] is not None for row in rows)


def test_a_run_is_reproducible_given_its_seed():
    t, X, names, _ = dc.demo_data("Logistic growth", noise=0.02, seed=4)
    a = dc.fit(t, X, names, degree=3, bootstrap=15, seed=9)
    b = dc.fit(t, X, names, degree=3, bootstrap=15, seed=9)
    assert np.array_equal(a.coef, b.coef) and np.array_equal(a.inclusion, b.inclusion)


def test_a_model_that_blows_up_is_a_finding_not_a_crash():
    t, X, names, _ = dc.demo_data("Logistic growth")
    r = dc.fit(t, X, names, degree=2, derivative_method="gradient")
    r.coef = r.coef * 40                                      # make the growth enormous: the simulation must overflow
    assert dc.simulation_error(r) == math.inf or dc.simulation_error(r) > 0.5
    assert dc.verdict(r, math.inf)[0] == "failed"


# ------------------------------------------------------------------------------------------ pieces

def test_derivatives_of_a_sine_are_accurate_with_both_methods():
    t = np.linspace(0, 6, 600)
    X = np.sin(t)[:, None]
    for method, tol in (("gradient", 1e-4), ("savgol", 1e-3)):
        d, _ = dc.estimate_derivatives(t, X, method)
        assert np.max(np.abs(d[20:-20, 0] - np.cos(t)[20:-20])) < tol


def test_the_smoother_beats_plain_differences_on_noisy_data():
    t = np.linspace(0, 6, 600)
    X = (np.sin(t) + np.random.default_rng(0).normal(0, 0.01, 600))[:, None]
    err = {m: np.max(np.abs(dc.estimate_derivatives(t, X, m)[0][40:-40, 0] - np.cos(t)[40:-40])) for m in ("gradient", "savgol")}
    assert err["savgol"] < err["gradient"] / 3


def test_uneven_time_is_resampled_and_the_note_says_so():
    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10, 800)); t[0], t[-1] = 0.0, 10.0
    X = np.exp(-0.5 * t)[:, None]
    r = dc.fit(t, X, ["x"], degree=1, constant=False)
    assert r.resampled and any("interpolated" in n for n in r.notes) and coef_of(r, "x", "x") == pytest.approx(-0.5, rel=0.05)
    assert dc.recognise(r) == "exponential decay"


def test_the_library_has_the_expected_terms():
    _, terms = dc.build_library(["x", "y"], degree=2, trig=True)
    assert [str(t) for t in terms] == ["1", "x", "y", "x**2", "x*y", "y**2", "sin(x)", "cos(x)", "sin(y)", "cos(y)"]


def test_bic_prefers_fewer_terms_at_equal_error_and_more_terms_only_for_a_real_drop_in_error():
    assert dc.bic(1.0, 1000, 3) < dc.bic(1.0, 1000, 6)
    assert dc.bic(0.5, 1000, 6) < dc.bic(1.0, 1000, 3)            # halving the error is worth three more terms


def test_exported_python_runs_and_matches_the_discovered_system():
    t, X, names, _ = dc.demo_data("Damped oscillator")
    r = dc.fit(t, X, names, degree=2, derivative_method="gradient")
    ns: dict = {}
    exec(dc.to_python(r), ns)                                 # noqa: S102 -- our own generated code, in a test
    state = [0.7, -0.3]
    expected = [float(r.rhs(j).subs({"x": state[0], "v": state[1]})) for j in range(2)]
    assert ns["rhs"](0.0, state) == pytest.approx(expected)
    assert all("\\dot" in s for s in r.latex())


# ------------------------------------------------------------------------------------------ input handling

def test_pasted_csv_with_and_without_a_header_parses():
    t, X, names = dc.parse_table("t,x,v\n0,1,0\n0.1,0.9,-0.1\n0.2,0.8,-0.2\n")
    assert names == ["x", "v"] and X.shape == (3, 2) and t[1] == 0.1
    t, X, names = dc.parse_table("0 1\n1 2\n2 3\n")
    assert names == ["x"] and X.shape == (3, 1)


@pytest.mark.parametrize("text, message", [
    ("t,x\n0,1\n", "at least a few rows"),
    ("t,x\n0,1\n1,a\n2,3\n", "not a number"),
    ("t,x\n0,1\n1,nan\n2,3\n", "non-finite"),
    ("t,x y\n0,1\n1,2\n2,3\n", "cannot be used"),
    ("t,x,x\n0,1,1\n1,2,2\n2,3,3\n", "unique"),
])
def test_bad_tables_are_refused_with_a_reason(text, message):
    with pytest.raises(dc.DiscoverError, match=message):
        dc.parse_table(text)


def test_inputs_the_method_cannot_support_are_refused():
    t = np.linspace(0, 1, 50)
    with pytest.raises(dc.DiscoverError, match="at least 10"):
        dc.fit(t[:5], np.ones((5, 1)), ["x"])
    with pytest.raises(dc.DiscoverError, match="strictly increasing"):
        dc.fit(t[::-1], np.ones((50, 1)), ["x"])
    with pytest.raises(dc.DiscoverError, match="too few|limit"):
        dc.fit(t, np.random.default_rng(0).normal(size=(50, 4)), ["a", "b", "c", "d"], degree=3)
    with pytest.raises(dc.DiscoverError, match="threshold"):
        dc.fit(np.linspace(0, 5, 300), np.sin(np.linspace(0, 5, 300))[:, None], ["x"], threshold=-1)
    with pytest.raises(dc.DiscoverError, match="Unknown demo"):
        dc.demo_data("nope")
