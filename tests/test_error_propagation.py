import math

import pytest

from modules.equation_engine import build_model
from modules.error_propagation import propagate_error, UncertainVariable


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


def _product_model(m_known="2", v_known="3"):
    """p = m * v -- simple product, easy to hand-verify propagation.
    m/v default to concrete known values, matching how this is actually
    used: an uncertain variable is a subset of the model's otherwise-
    known inputs, not a variable left fully unresolved."""
    return build_model({
        "problem_domain": "mechanics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "m", "meaning": "mass", "known_value": m_known, "unit": "kg"},
            {"symbol": "v", "meaning": "velocity", "known_value": v_known, "unit": "m/s"},
            {"symbol": "p", "meaning": "momentum", "known_value": None, "unit": "kg*m/s"},
        ],
        "equations": [{"name": "mom", "kind": "equation", "expression": "Eq(p, m * v)", "derivation": ""}],
        "solve_for": ["p"], "assumptions": [],
    })


# ---------------------------------------------------------------- basic correctness

def test_linear_case_matches_hand_calculation():
    """a = (v_f - v_i) / t is linear in v_f, so d(a)/d(v_f) = 1/t = 1/6
    exactly -- sigma_a = sigma_vf / 6, computable by hand."""
    model = _kinematics_model()
    result = propagate_error(model, "a", [UncertainVariable("v_f", 20.0, 1.2)])
    assert result.value == pytest.approx(2.0)  # (20-8)/6
    assert result.std == pytest.approx(1.2 / 6)
    assert result.partials["v_f"] == pytest.approx(1 / 6)


def test_product_rule_matches_hand_calculation():
    """p = m*v; dp/dm = v, dp/dv = m. At m=2, v=3:
    sigma_p^2 = (v*sigma_m)^2 + (m*sigma_v)^2 = (3*0.1)^2 + (2*0.2)^2"""
    model = _product_model()
    result = propagate_error(
        model, "p", [UncertainVariable("m", 2.0, 0.1), UncertainVariable("v", 3.0, 0.2)],
    )
    expected_std = math.sqrt((3.0 * 0.1) ** 2 + (2.0 * 0.2) ** 2)
    assert result.value == pytest.approx(6.0)
    assert result.std == pytest.approx(expected_std)
    assert result.partials["m"] == pytest.approx(3.0)   # dp/dm = v = 3
    assert result.partials["v"] == pytest.approx(2.0)   # dp/dv = m = 2


def test_contributions_sum_to_one():
    model = _product_model()
    result = propagate_error(
        model, "p", [UncertainVariable("m", 2.0, 0.1), UncertainVariable("v", 3.0, 0.2)],
    )
    assert sum(result.contributions.values()) == pytest.approx(1.0)
    # v's term (2*0.2=0.4) dominates over m's term (3*0.1=0.3) here
    assert result.contributions["v"] > result.contributions["m"]


def test_formula_latex_is_populated():
    model = _kinematics_model()
    result = propagate_error(model, "a", [UncertainVariable("v_f", 20.0, 1.0)])
    assert result.formula_latex is not None
    assert "v_{f}" in result.formula_latex or "v_f" in result.formula_latex


# ---------------------------------------------------------------- edge cases

def test_variable_with_zero_sensitivity_reported_not_errored():
    """A target that doesn't actually depend on an uncertain input
    should report a zero partial/contribution, not raise."""
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
    result = propagate_error(model, "z", [UncertainVariable("x", 1.0, 0.5)])
    assert result.partials["x"] == 0.0
    assert result.contributions["x"] == 0.0
    assert result.std == 0.0


def test_single_uncertain_variable_all_contribution_is_itself():
    model = _kinematics_model()
    result = propagate_error(model, "a", [UncertainVariable("v_f", 20.0, 1.0)])
    assert result.contributions["v_f"] == pytest.approx(1.0)


# ---------------------------------------------------------------- error handling

def test_rejects_target_not_in_solve_for():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_error(model, "v_f", [UncertainVariable("v_i", 8.0, 1.0)])


def test_rejects_empty_uncertain_vars():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_error(model, "a", [])


def test_rejects_non_positive_std():
    model = _kinematics_model()
    with pytest.raises(ValueError):
        propagate_error(model, "a", [UncertainVariable("v_f", 20.0, 0.0)])


def test_multiple_uncertain_variables_wider_std_with_more_inputs():
    model = _product_model()
    one_var = propagate_error(model, "p", [UncertainVariable("m", 2.0, 0.1)])
    two_vars = propagate_error(
        model, "p", [UncertainVariable("m", 2.0, 0.1), UncertainVariable("v", 3.0, 0.2)],
    )
    assert two_vars.std > one_var.std
