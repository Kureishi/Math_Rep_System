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


# ---------------------------------------------------------------- remaining Interval branches
import math

import sympy as sp

import modules.interval_arithmetic as ia
from modules.interval_arithmetic import _interval_sqrt, _interval_exp, _interval_log
from modules.timeout_utils import ComputationTimeoutError


def test_repr_is_readable():
    assert repr(Interval(1, 2.5)) == "Interval(1, 2.5)"


def test_zeroth_power_is_exactly_one():
    r = Interval(-3, 5) ** 0
    assert (r.lo, r.hi) == (1.0, 1.0)


def test_even_power_of_all_negative_interval_swaps_endpoints():
    # [-3, -2]**2 = [4, 9]: for a wholly negative range the LARGER magnitude
    # is the LOWER endpoint, so the naive (lo**n, hi**n) = (9, 4) would be
    # inverted -- the swap is the whole point of this branch.
    r = Interval(-3, -2) ** 2
    assert (r.lo, r.hi) == (4.0, 9.0)


def test_even_power_spanning_zero_takes_larger_magnitude_endpoint():
    r = Interval(-5, 2) ** 2
    assert (r.lo, r.hi) == (0.0, 25.0)


def test_non_integer_power_of_nonnegative_interval_works():
    r = Interval(4, 9) ** 0.5
    assert r.lo == pytest.approx(2.0) and r.hi == pytest.approx(3.0)


def test_negative_integer_power_of_negative_interval_raises():
    with pytest.raises(ValueError):
        Interval(-2, -1) ** -1


def test_interval_exponent_is_rejected():
    with pytest.raises(TypeError):
        Interval(1, 2) ** Interval(1, 2)
    with pytest.raises(TypeError):
        Interval(1, 2) ** "2"


def test_reflected_operations_with_scalars():
    assert (10 - Interval(1, 3)).lo == 7 and (10 - Interval(1, 3)).hi == 9
    r = 6 / Interval(2, 3)
    assert (r.lo, r.hi) == (2.0, 3.0)


def test_interval_sqrt_exp_log_monotonic_endpoints():
    s = _interval_sqrt(Interval(4, 9))
    assert (s.lo, s.hi) == (2.0, 3.0)
    e = _interval_exp(Interval(0, 1))
    assert e.lo == 1.0 and e.hi == pytest.approx(math.e)
    l = _interval_log(Interval(1, math.e))
    assert l.lo == 0.0 and l.hi == pytest.approx(1.0)


def test_interval_sqrt_of_range_reaching_below_zero_raises():
    with pytest.raises(ValueError):
        _interval_sqrt(Interval(-1, 4))


def test_interval_log_of_range_reaching_zero_raises():
    with pytest.raises(ValueError):
        _interval_log(Interval(0, 4))  # log(0) is -inf; lo <= 0 is rejected, not silently -inf


def test_interval_functions_accept_plain_scalars():
    assert _interval_sqrt(9).lo == 3.0
    assert _interval_exp(0).hi == 1.0


# ---------------------------------------------------------------- propagate_interval error paths

def test_propagate_interval_exp_and_log_through_lambdify():
    model = build_model({
        "problem_domain": "general", "problem_type": "algebraic",
        "variables": [
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(y, exp(x) + log(x))", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    r = propagate_interval(model, "y", {"x": (1.0, 2.0)})
    assert r.lo == pytest.approx(math.exp(1) + 0.0)
    assert r.hi == pytest.approx(math.exp(2) + math.log(2))


def test_propagate_interval_target_independent_of_ranges_is_deterministic():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "vf", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "vi", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": "m/s^2"},
            {"symbol": "m", "meaning": "an unrelated mass", "known_value": "3", "unit": "kg"},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""}],
        "solve_for": ["a"], "assumptions": [],
    })
    # a range is supplied for m, but a doesn't depend on m -> zero-width answer at the exact value
    r = propagate_interval(model, "a", {"m": (1.0, 5.0)})
    assert r.lo == r.hi == pytest.approx(2.0)
    assert r.width == 0.0 and r.formula_latex


def test_propagate_interval_model_without_algebraic_equations():
    # target_kind() falls back to "equation" for a solve_for variable that
    # no relation mentions at all, so the target passes the first guard --
    # and the model's only relation is an inequality on a DIFFERENT
    # variable, leaving nothing algebraic to solve.
    model = build_model({
        "problem_domain": "general", "problem_type": "inequality",
        "variables": [{"symbol": "v", "meaning": "speed", "known_value": None, "unit": None},
                      {"symbol": "w", "meaning": "other", "known_value": None, "unit": None}],
        "equations": [{"name": "c", "kind": "inequality", "expression": "w <= 25", "derivation": ""}],
        "solve_for": ["v"], "assumptions": [],
    })
    with pytest.raises(ValueError, match="no algebraic equations"):
        propagate_interval(model, "v", {"w": (1.0, 2.0)})


def test_propagate_interval_rejects_inequality_target():
    model = build_model({
        "problem_domain": "general", "problem_type": "inequality",
        "variables": [{"symbol": "v", "meaning": "speed", "known_value": None, "unit": None}],
        "equations": [{"name": "c", "kind": "inequality", "expression": "v <= 25", "derivation": ""}],
        "solve_for": ["v"], "assumptions": [],
    })
    with pytest.raises(ValueError, match="isn't an algebraic solve_for target"):
        propagate_interval(model, "v", {"v": (1.0, 2.0)})


def test_propagate_interval_inconsistent_system_reports_unsolvable():
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
        propagate_interval(model, "a", {"k": (1.0, 2.0)})


def test_propagate_interval_solve_timeout_propagates(monkeypatch):
    def boom(*a, **k):
        raise ComputationTimeoutError(0.1, "interval solve")
    monkeypatch.setattr(ia, "run_with_timeout", boom)
    with pytest.raises(ComputationTimeoutError):
        propagate_interval(_kinematics_model(), "a", {"v_f": (19.0, 21.0)})


def test_propagate_interval_solve_crash_becomes_value_error(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("sympy exploded")
    monkeypatch.setattr(ia, "run_with_timeout", boom)
    with pytest.raises(ValueError, match="sympy exploded"):
        propagate_interval(_kinematics_model(), "a", {"v_f": (19.0, 21.0)})


def test_propagate_interval_target_missing_from_solution(monkeypatch):
    monkeypatch.setattr(ia, "run_with_timeout", lambda *a, **k: [{sp.Symbol("zzz"): 1}])
    with pytest.raises(ValueError, match="didn't appear"):
        propagate_interval(_kinematics_model(), "a", {"v_f": (19.0, 21.0)})


def test_propagate_interval_lambdify_failure_is_wrapped(monkeypatch):
    def bad_lambdify(*a, **k):
        raise TypeError("unsupported")
    monkeypatch.setattr(ia.sp, "lambdify", bad_lambdify)
    with pytest.raises(ValueError, match="interval-evaluable"):
        propagate_interval(_kinematics_model(), "a", {"v_f": (19.0, 21.0)})


def test_propagate_interval_non_interval_result_is_degenerate(monkeypatch):
    # if the lambdified function collapses to a plain number, it's reported as a
    # zero-width interval rather than crashing on `.lo`
    monkeypatch.setattr(ia.sp, "lambdify", lambda *a, **k: (lambda *args: 5.0))
    r = propagate_interval(_kinematics_model(), "a", {"v_f": (19.0, 21.0)})
    assert r.lo == r.hi == 5.0 and r.width == 0.0


def test_propagate_interval_evaluation_failure_is_wrapped():
    # a = 1/(v_f - 20) with v_f in [19, 21] divides by a range spanning zero
    model = build_model({
        "problem_domain": "general", "problem_type": "algebraic",
        "variables": [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
                      {"symbol": "v_f", "meaning": "vf", "known_value": None, "unit": None}],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(a, 1/(v_f - 20))", "derivation": ""}],
        "solve_for": ["a"], "assumptions": [],
    })
    with pytest.raises(ValueError, match="Couldn't evaluate"):
        propagate_interval(model, "a", {"v_f": (19.0, 21.0)})
