import pytest

from modules.equation_engine import build_model
from modules.parameter_sweep import sweep_parameters, sweep_result_to_grid, MAX_GRID_POINTS


def _force_model():
    """F = m * a"""
    return build_model({
        "problem_domain": "mechanics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "m", "meaning": "mass", "known_value": None, "unit": "kg"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
            {"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(F, m*a)", "derivation": ""}],
        "solve_for": ["F"], "assumptions": [],
    })


def _kinematics_model():
    """a = (v_f - v_i) / t"""
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


# ---------------------------------------------------------------- basic 2D sweep

def test_two_dimensional_sweep_matches_hand_calculation():
    model = _force_model()
    result = sweep_parameters(model, "F", {"m": [1.0, 2.0, 3.0], "a": [10.0, 20.0]})
    assert len(result.rows) == 6
    by_pair = {(r["m"], r["a"]): r["F"] for r in result.rows}
    assert by_pair[(1.0, 10.0)] == pytest.approx(10.0)
    assert by_pair[(2.0, 20.0)] == pytest.approx(40.0)
    assert by_pair[(3.0, 10.0)] == pytest.approx(30.0)


def test_swept_symbols_preserved_in_order():
    model = _force_model()
    result = sweep_parameters(model, "F", {"a": [10.0], "m": [1.0, 2.0]})
    assert result.swept_symbols == ["a", "m"]


# ---------------------------------------------------------------- 1D sweep (degenerate case)

def test_single_variable_sweep_degenerates_cleanly():
    model = _kinematics_model()
    result = sweep_parameters(model, "a", {"v_f": [16.0, 20.0, 24.0]})
    assert len(result.rows) == 3
    values = {r["v_f"]: r["a"] for r in result.rows}
    assert values[16.0] == pytest.approx((16 - 8) / 6)
    assert values[20.0] == pytest.approx((20 - 8) / 6)
    assert values[24.0] == pytest.approx((24 - 8) / 6)


# ---------------------------------------------------------------- three-dimensional sweep

def test_three_dimensional_sweep():
    model = _kinematics_model()
    result = sweep_parameters(model, "a", {"v_f": [16.0, 20.0], "v_i": [4.0, 8.0], "t": [3.0, 6.0]})
    assert len(result.rows) == 8
    row = next(r for r in result.rows if r["v_f"] == 20.0 and r["v_i"] == 8.0 and r["t"] == 6.0)
    assert row["a"] == pytest.approx(2.0)


# ---------------------------------------------------------------- irrelevant variable

def test_swept_variable_target_does_not_depend_on_still_produces_full_grid():
    """Sweeping a variable the target doesn't actually use (e.g. sweeping
    't' for a target that turns out not to depend on it once other
    values are fixed) should still produce one row per grid point, with
    a constant target value repeated across that variable's values."""
    model = build_model({
        "problem_domain": "physics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
            {"symbol": "unused", "meaning": "unused", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(y, x * 2)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    result = sweep_parameters(model, "y", {"x": [1.0, 2.0], "unused": [100.0, 200.0]})
    assert len(result.rows) == 4
    for r in result.rows:
        assert r["y"] == pytest.approx(r["x"] * 2)


# ---------------------------------------------------------------- error handling

def test_rejects_target_not_in_solve_for():
    model = _force_model()
    with pytest.raises(ValueError):
        sweep_parameters(model, "m", {"a": [1.0, 2.0]})


def test_rejects_empty_sweep_values():
    model = _force_model()
    with pytest.raises(ValueError):
        sweep_parameters(model, "F", {})


def test_rejects_variable_with_no_values():
    model = _force_model()
    with pytest.raises(ValueError):
        sweep_parameters(model, "F", {"m": []})


def test_rejects_grid_over_max_points():
    model = _force_model()
    with pytest.raises(ValueError):
        sweep_parameters(model, "F", {"m": list(range(200)), "a": list(range(200))})  # 40,000 points


# ---------------------------------------------------------------- sweep_result_to_grid

def test_sweep_result_to_grid_reconstructs_correctly():
    model = _force_model()
    result = sweep_parameters(model, "F", {"m": [1.0, 2.0, 3.0], "a": [10.0, 20.0]})
    x_values, y_values, z_matrix = sweep_result_to_grid(result)
    assert x_values == [1.0, 2.0, 3.0]
    assert y_values == [10.0, 20.0]
    assert z_matrix.shape == (2, 3)
    assert z_matrix[0, 0] == pytest.approx(10.0)   # a=10, m=1 -> F=10
    assert z_matrix[1, 2] == pytest.approx(60.0)   # a=20, m=3 -> F=60


def test_sweep_result_to_grid_rejects_wrong_variable_count():
    model = _kinematics_model()
    result = sweep_parameters(model, "a", {"v_f": [16.0, 20.0]})
    with pytest.raises(ValueError):
        sweep_result_to_grid(result)
