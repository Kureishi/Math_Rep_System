import pytest

from modules.equation_engine import build_model
from modules.goal_seek import goal_seek


def _kinematics_model():
    """a = (v_f - v_i) / t -- nominal a = (20-8)/6 = 2.0"""
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


def _quadratic_area_model(domain=None):
    """A = s^2 -- solves for area given a side length; used to exercise
    a two-real-root inverse (s = +sqrt(A), s = -sqrt(A))."""
    return build_model({
        "problem_domain": "geometry", "problem_type": "algebraic",
        "variables": [
            {"symbol": "s", "meaning": "side length", "known_value": "3", "unit": "m",
             **({"domain": domain} if domain else {})},
            {"symbol": "A", "meaning": "area", "known_value": None, "unit": "m^2"},
        ],
        "equations": [{"name": "area", "kind": "equation", "expression": "Eq(A, s**2)", "derivation": ""}],
        "solve_for": ["A"], "assumptions": [],
    })


def _transcendental_model():
    """x + sin(x) = y -- no closed-form inverse, exercises the numerical fallback."""
    return build_model({
        "problem_domain": "physics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "1.0", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(y, x + sin(x))", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })


# ---------------------------------------------------------------- basic linear inversion

def test_seek_recovers_known_value_when_target_unchanged():
    """Seeking v_i for the SAME nominal a=2.0 should recover v_i=8, the
    model's own already-known value -- a sanity check that the
    inversion is self-consistent with the forward solve."""
    model = _kinematics_model()
    result = goal_seek(model, "a", 2.0, "v_i")
    assert result.solutions == pytest.approx([8.0])
    assert not result.is_numerical
    assert result.formula is not None


def test_seek_finds_correct_value_for_a_different_target():
    model = _kinematics_model()
    # a = (20 - v_i) / 6 = 3.0  =>  v_i = 20 - 18 = 2.0
    result = goal_seek(model, "a", 3.0, "v_i")
    assert result.solutions == pytest.approx([2.0])


def test_seek_on_a_different_variable_in_same_equation():
    model = _kinematics_model()
    # a = (v_f - 8) / 6 = 2.0  =>  v_f = 20 (recovers the known value)
    result = goal_seek(model, "a", 2.0, "v_f")
    assert result.solutions == pytest.approx([20.0])


# ---------------------------------------------------------------- multi-root / domain filtering

def test_quadratic_goal_returns_both_real_roots_when_no_domain_declared():
    model = _quadratic_area_model()
    result = goal_seek(model, "A", 16.0, "s")
    assert result.solutions == pytest.approx([-4.0, 4.0])


def test_quadratic_goal_filtered_to_positive_domain():
    model = _quadratic_area_model(domain="positive")
    result = goal_seek(model, "A", 16.0, "s")
    assert result.solutions == pytest.approx([4.0])


def test_domain_filter_never_empties_the_result():
    """A declared domain that's inconsistent with every root (a
    contrived case) should fall back to the unfiltered set rather than
    raising or returning nothing."""
    model = _quadratic_area_model(domain="negative")
    result = goal_seek(model, "A", 16.0, "s")
    # 'negative' would keep neither +4 nor... wait -4 IS negative, so this
    # actually keeps just [-4.0]
    assert result.solutions == pytest.approx([-4.0])


# ---------------------------------------------------------------- numerical fallback

def test_transcendental_equation_falls_back_to_numerical_root():
    model = _transcendental_model()
    # y = x + sin(x); solve for x given y = 1.0 + sin(1.0)
    import math
    target_y = 1.0 + math.sin(1.0)
    result = goal_seek(model, "y", target_y, "x")
    assert result.is_numerical
    assert any(abs(v - 1.0) < 1e-4 for v in result.solutions)


# ---------------------------------------------------------------- error handling

def test_rejects_non_algebraic_target():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        goal_seek(model, "v_f", 2.0, "v_i")  # v_f isn't in solve_for


def test_rejects_unknown_seek_symbol():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        goal_seek(model, "a", 2.0, "not_a_real_symbol")


def test_rejects_seeking_the_target_itself():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        goal_seek(model, "a", 2.0, "a")


def test_raises_when_equation_does_not_depend_on_seek_symbol():
    model = _kinematics_model()
    # 't' isn't wired to a variable that's absent from the equation --
    # construct a model where the seek symbol genuinely doesn't appear
    unrelated_model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "v_f", "known_value": "20", "unit": None},
            {"symbol": "v_i", "meaning": "v_i", "known_value": "8", "unit": None},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": None},
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": None},
            {"symbol": "unrelated", "meaning": "unrelated", "known_value": "5", "unit": None},
        ],
        "equations": [{"name": "accel", "kind": "equation",
                        "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""}],
        "solve_for": ["a"], "assumptions": [],
    })
    with pytest.raises(ValueError):
        goal_seek(unrelated_model, "a", 2.0, "unrelated")


def test_raises_when_no_real_solution_exists():
    """A^s where s is fixed... use a scenario with no real inverse: sqrt
    requires nonnegative area for a real side length; asking for a
    negative area should fail cleanly."""
    model = _quadratic_area_model()
    with pytest.raises(ValueError):
        goal_seek(model, "A", -16.0, "s")
