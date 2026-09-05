import pytest

from modules.equation_engine import build_model
from modules.interval_arithmetic import Interval, propagate_interval


# ---------------------------------------------------------------- Interval primitive

def test_addition():
    r = Interval(1, 2) + Interval(3, 5)
    assert (r.lo, r.hi) == (4, 7)


def test_subtraction():
    r = Interval(1, 2) - Interval(3, 5)
    assert (r.lo, r.hi) == (1 - 5, 2 - 3)  # (-4, -1)


def test_multiplication_both_positive():
    r = Interval(2, 3) * Interval(4, 5)
    assert (r.lo, r.hi) == (8, 15)


def test_multiplication_spanning_zero():
    r = Interval(-2, 3) * Interval(-1, 4)
    # corners: (-2)(-1)=2, (-2)(4)=-8, (3)(-1)=-3, (3)(4)=12
    assert (r.lo, r.hi) == (-8, 12)


def test_division_normal():
    r = Interval(4, 6) / Interval(2, 2)
    assert (r.lo, r.hi) == (2, 3)


def test_division_by_range_spanning_zero_raises():
    with pytest.raises(ZeroDivisionError):
        Interval(1, 2) / Interval(-1, 1)


def test_even_power_spanning_zero_has_minimum_zero():
    r = Interval(-2, 3) ** 2
    assert r.lo == 0
    assert r.hi == 9  # max(4, 9)


def test_even_power_all_positive():
    r = Interval(2, 3) ** 2
    assert (r.lo, r.hi) == (4, 9)


def test_odd_power_spanning_zero():
    r = Interval(-2, 3) ** 3
    assert (r.lo, r.hi) == (-8, 27)


def test_scalar_operations_coerce():
    r = Interval(1, 2) + 5
    assert (r.lo, r.hi) == (6, 7)
    r2 = 5 - Interval(1, 2)
    assert (r2.lo, r2.hi) == (3, 4)


def test_negation():
    r = -Interval(1, 3)
    assert (r.lo, r.hi) == (-3, -1)


def test_non_integer_power_of_negative_interval_raises():
    with pytest.raises(ValueError):
        Interval(-1, 4) ** 0.5


def test_inverted_construction_is_normalized():
    iv = Interval(5, 2)
    assert (iv.lo, iv.hi) == (2, 5)


def test_width_and_midpoint():
    iv = Interval(2, 8)
    assert iv.width() == 6
    assert iv.midpoint() == 5


# ---------------------------------------------------------------- propagate_interval end-to-end

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


def test_propagate_interval_matches_hand_calculation():
    model = _kinematics_model()
    # a = (v_f - 8) / 6, v_f in [19, 21] => a in [(19-8)/6, (21-8)/6]
    result = propagate_interval(model, "a", {"v_f": (19.0, 21.0)})
    assert result.lo == pytest.approx((19 - 8) / 6)
    assert result.hi == pytest.approx((21 - 8) / 6)


def test_propagate_interval_multiple_uncertain_inputs():
    model = _kinematics_model()
    result = propagate_interval(model, "a", {"v_f": (19.0, 21.0), "v_i": (7.0, 9.0)})
    # worst case: max a = (21-7)/6, min a = (19-9)/6
    assert result.hi == pytest.approx((21 - 7) / 6)
    assert result.lo == pytest.approx((19 - 9) / 6)


def test_propagate_interval_with_sqrt():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "d", "meaning": "distance", "known_value": None, "unit": "m"},
            {"symbol": "t", "meaning": "time", "known_value": None, "unit": "s"},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(t, sqrt(d))", "derivation": ""}],
        "solve_for": ["t"], "assumptions": [],
    })
    result = propagate_interval(model, "t", {"d": (4.0, 9.0)})
    assert result.lo == pytest.approx(2.0)
    assert result.hi == pytest.approx(3.0)


def test_degenerate_range_produces_zero_width():
    model = _kinematics_model()
    result = propagate_interval(model, "a", {"v_f": (20.0, 20.0)})
    assert result.width == pytest.approx(0.0)
    assert result.lo == pytest.approx(2.0)


def test_formula_latex_populated():
    model = _kinematics_model()
    result = propagate_interval(model, "a", {"v_f": (19.0, 21.0)})
    assert result.formula_latex is not None


# ---------------------------------------------------------------- error handling

def test_rejects_target_not_in_solve_for():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_interval(model, "v_f", {"v_i": (7.0, 9.0)})


def test_rejects_empty_ranges():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_interval(model, "a", {})


def test_rejects_inverted_range():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_interval(model, "a", {"v_f": (21.0, 19.0)})


def test_division_by_interval_spanning_zero_raises_clean_error():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(y, 1 / x)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    with pytest.raises(ValueError):
        propagate_interval(model, "y", {"x": (-1.0, 1.0)})
