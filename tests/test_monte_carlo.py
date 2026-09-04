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
