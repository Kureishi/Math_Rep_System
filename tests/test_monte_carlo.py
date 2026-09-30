import pytest

from modules.equation_engine import build_model
from modules.monte_carlo import run_monte_carlo, UncertainVariable, MAX_SAMPLES


def _kinematics_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def test_basic_propagation_produces_samples_and_stats():
    model = _kinematics_model()
    result = run_monte_carlo(
        model, "a", [UncertainVariable(symbol="v_f", mean=20.0, std=1.0)],
        n_samples=200, seed=42,
    )
    assert len(result.samples) > 0
    assert result.n_requested == 200
    assert result.mean == pytest.approx(2.0, abs=0.2)  # (20-8)/6 = 2.0 nominal
    assert result.std is not None and result.std > 0
    assert result.p5 < result.mean < result.p95


def test_mean_matches_nominal_deterministic_solve():
    """With a std small enough to be negligible, the Monte Carlo mean
    should land very close to the deterministic (20-8)/6 = 2.0 answer."""
    model = _kinematics_model()
    result = run_monte_carlo(
        model, "a", [UncertainVariable(symbol="v_f", mean=20.0, std=0.001)],
        n_samples=100, seed=1,
    )
    assert result.mean == pytest.approx(2.0, abs=0.01)


def test_wider_std_produces_wider_spread():
    model = _kinematics_model()
    narrow = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 0.5)], n_samples=200, seed=7)
    wide = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 5.0)], n_samples=200, seed=7)
    assert wide.std > narrow.std


def test_multiple_uncertain_variables_propagate_jointly():
    model = _kinematics_model()
    result = run_monte_carlo(
        model, "a",
        [UncertainVariable("v_f", 20.0, 1.0), UncertainVariable("v_i", 8.0, 1.0)],
        n_samples=200, seed=3,
    )
    assert len(result.samples) > 0
    assert result.mean == pytest.approx(2.0, abs=0.3)


def test_reproducible_with_same_seed():
    model = _kinematics_model()
    r1 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100, seed=99)
    r2 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100, seed=99)
    assert r1.samples == r2.samples


def test_rejects_target_not_in_solve_for():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        run_monte_carlo(model, "v_f", [UncertainVariable("v_i", 8.0, 1.0)], n_samples=100)


def test_rejects_empty_uncertain_vars():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        run_monte_carlo(model, "a", [], n_samples=100)


def test_rejects_non_positive_std():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 0.0)], n_samples=100)


def test_sample_count_clamped_to_max(monkeypatch):
    import modules.monte_carlo as mc_module
    monkeypatch.setattr(mc_module, "MAX_SAMPLES", 150)
    model = _kinematics_model()
    result = run_monte_carlo(
        model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100000, seed=5,
    )
    assert result.n_requested == 150


def test_sample_count_clamped_to_minimum():
    model = _kinematics_model()
    result = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=1, seed=5)
    assert result.n_requested == 10


def test_failed_samples_tracked_separately_from_successful_ones():
    """A variable whose sign restriction is regularly violated by its own
    sampled distribution should produce some failed draws without
    crashing the whole run."""
    model = build_model({
        "problem_domain": "physics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "m", "meaning": "mass", "known_value": None, "unit": "kg", "domain": "positive"},
            {"symbol": "v", "meaning": "velocity", "known_value": "2", "unit": "m/s"},
            {"symbol": "p", "meaning": "momentum", "known_value": None, "unit": "kg*m/s"},
        ],
        "equations": [{"name": "mom", "kind": "equation", "expression": "Eq(p, m * v)", "derivation": ""}],
        "solve_for": ["p"], "assumptions": [],
    })
    # mean well above zero but std wide enough that some samples land negative
    result = run_monte_carlo(model, "p", [UncertainVariable("m", 1.0, 5.0)], n_samples=150, seed=11)
    assert result.n_requested == 150
    assert len(result.samples) + result.n_failed == 150


# ---------------------------------------------------------------- seed reproducibility

def test_auto_generated_seed_is_a_concrete_int():
    model = _kinematics_model()
    result = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100)
    assert isinstance(result.seed, int)
    assert result.seed != 0  # extremely unlikely to land exactly on the dataclass default by chance


def test_two_auto_generated_seeds_differ():
    model = _kinematics_model()
    r1 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100)
    r2 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100)
    assert r1.seed != r2.seed


def test_reusing_the_returned_seed_reproduces_identical_samples():
    model = _kinematics_model()
    r1 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=200)
    r2 = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=200, seed=r1.seed)
    assert r1.samples == r2.samples
    assert r1.seed == r2.seed


def test_explicit_seed_is_echoed_back_unchanged():
    model = _kinematics_model()
    result = run_monte_carlo(model, "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=100, seed=42)
    assert result.seed == 42


def test_seed_is_set_even_on_the_deterministic_no_dependence_path():
    """When the target doesn't actually depend on any uncertain input,
    the early-return 'deterministic' branch must still carry a real
    seed -- every result should be reproducible-by-seed, not just the
    common case."""
    model = build_model({
        "problem_domain": "physics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": "3", "unit": None},
            {"symbol": "z", "meaning": "z", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(z, y * 2)", "derivation": ""}],
        "solve_for": ["z"], "assumptions": [],
    })
    result = run_monte_carlo(model, "z", [UncertainVariable("x", 1.0, 0.5)], n_samples=50, seed=7)
    assert result.seed == 7


# ---------------------------------------------------------------- error paths and fallbacks
import numpy as np
import sympy as sp

import modules.monte_carlo as mc
from modules.timeout_utils import ComputationTimeoutError


def test_model_without_algebraic_equations_is_rejected():
    # 'v' is mentioned by no relation, so target_kind() falls back to "equation"
    # and the target passes the first guard; the only relation is an inequality
    # on another variable, so there is nothing algebraic to solve.
    model = build_model({
        "problem_domain": "general", "problem_type": "inequality",
        "variables": [{"symbol": "v", "meaning": "v", "known_value": None, "unit": None},
                      {"symbol": "w", "meaning": "w", "known_value": None, "unit": None}],
        "equations": [{"name": "c", "kind": "inequality", "expression": "w <= 25", "derivation": ""}],
        "solve_for": ["v"], "assumptions": [],
    })
    with pytest.raises(ValueError, match="no algebraic equations"):
        run_monte_carlo(model, "v", [UncertainVariable("w", 1.0, 0.1)], n_samples=20, seed=1)


def test_solve_timeout_propagates_as_timeout(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "monte carlo solve")
    monkeypatch.setattr(mc, "run_with_timeout", boom)
    with pytest.raises(ComputationTimeoutError):
        run_monte_carlo(_kinematics_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=20, seed=1)


def test_solve_crash_becomes_value_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sympy exploded")
    monkeypatch.setattr(mc, "run_with_timeout", boom)
    with pytest.raises(ValueError, match="sympy exploded"):
        run_monte_carlo(_kinematics_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=20, seed=1)


def test_inconsistent_system_reports_unsolvable():
    model = build_model({
        "problem_domain": "general", "problem_type": "algebraic",
        "variables": [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
                      {"symbol": "k", "meaning": "k", "known_value": None, "unit": None}],
        "equations": [
            {"name": "e1", "kind": "equation", "expression": "Eq(a, 1)", "derivation": ""},
            {"name": "e2", "kind": "equation", "expression": "Eq(a, 2)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    with pytest.raises(ValueError, match="Couldn't symbolically solve"):
        run_monte_carlo(model, "a", [UncertainVariable("k", 1.0, 0.1)], n_samples=20, seed=1)


def test_target_missing_from_solution_is_reported(monkeypatch):
    monkeypatch.setattr(mc, "run_with_timeout", lambda *a, **k: [{sp.Symbol("zzz"): 1}])
    with pytest.raises(ValueError, match="didn't appear"):
        run_monte_carlo(_kinematics_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=20, seed=1)


def test_per_sample_fallback_when_vectorized_evaluation_fails(monkeypatch):
    # Simulates an expression lambdify can't vectorize (piecewise etc.): the
    # function raises for a whole-array call but works on one scalar at a time.
    def fake_lambdify(symbols, expr, modules=None):
        def f(*args):
            if any(np.ndim(a) > 0 for a in args):
                raise TypeError("can't vectorize")
            return args[0] * 2.0
        return f
    monkeypatch.setattr(mc.sp, "lambdify", fake_lambdify)
    r = run_monte_carlo(_kinematics_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=50, seed=3)
    assert r.n_failed == 0 and len(r.samples) == 50
    # the scalar path evaluated the SAME draws the vectorized path would have seen
    expected = np.random.default_rng(3).normal(20.0, 1.0, size=50) * 2.0
    assert r.samples == pytest.approx(expected.tolist())


def test_per_sample_fallback_counts_raising_samples_as_failed(monkeypatch):
    calls = {"n": 0}

    def fake_lambdify(symbols, expr, modules=None):
        def f(*args):
            if any(np.ndim(a) > 0 for a in args):
                raise TypeError("can't vectorize")
            calls["n"] += 1
            if calls["n"] % 2 == 0:
                raise ZeroDivisionError("bad draw")   # every 2nd sample blows up
            return args[0]
        return f
    monkeypatch.setattr(mc.sp, "lambdify", fake_lambdify)
    r = run_monte_carlo(_kinematics_model(), "a", [UncertainVariable("v_f", 20.0, 1.0)], n_samples=40, seed=3)
    assert r.n_failed == 20 and len(r.samples) == 20
    assert r.n_failed + len(r.samples) == r.n_requested


def test_all_samples_failing_returns_empty_result_without_stats():
    # sqrt of a variable drawn from Normal(-100, 1): every single draw is
    # negative -> NaN/complex -> every sample dropped. Must return a result
    # (samples=[], stats None), NOT raise or divide by zero computing a mean.
    model = build_model({
        "problem_domain": "general", "problem_type": "algebraic",
        "variables": [{"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
                      {"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(y, sqrt(x))", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    with np.errstate(invalid="ignore"):
        r = run_monte_carlo(model, "y", [UncertainVariable("x", -100.0, 1.0)], n_samples=60, seed=9)
    assert r.samples == [] and r.n_failed == 60 and r.n_requested == 60
    assert r.mean is None and r.std is None and r.p5 is None and r.p95 is None
    assert r.seed == 9
