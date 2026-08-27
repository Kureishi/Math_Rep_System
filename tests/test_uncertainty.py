import math
import pytest
import sympy as sp

from modules.equation_engine import build_model
from modules.uncertainty import (
    propagate_uncertainty, solve_symbolic_for_target, uncertainty_for_target,
)
from modules.verifier import _known_substitutions
from modules.solver import compute_steps


def _kinematics_model(with_uncertainty=True):
    unc = {"uncertainty": "0.5"} if with_uncertainty else {}
    unc_t = {"uncertainty": "0.1"} if with_uncertainty else {}
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s", **unc},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s", **unc},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s", **unc_t},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def test_solve_symbolic_for_target_returns_formula_in_knowns():
    model = _kinematics_model()
    expr = solve_symbolic_for_target(model, "a")
    v_f, v_i, t = sp.symbols("v_f v_i t")
    assert expr is not None
    assert sp.simplify(expr - (v_f - v_i) / t) == 0


def test_propagate_uncertainty_matches_hand_calculation():
    # a = (v_f - v_i) / t; da/dv_f = 1/t, da/dv_i = -1/t, da/dt = -(v_f-v_i)/t^2
    v_f, v_i, t = sp.symbols("v_f v_i t")
    expr = (v_f - v_i) / t
    values = {v_f: 20.0, v_i: 8.0, t: 6.0}
    uncertainties = {v_f: 0.5, v_i: 0.5, t: 0.1}
    result = propagate_uncertainty(expr, values, uncertainties)
    assert result is not None
    assert result.nominal == pytest.approx(2.0, abs=1e-9)

    # hand-computed terms
    d_vf = 1 / 6.0
    d_vi = -1 / 6.0
    d_t = -(20.0 - 8.0) / 6.0 ** 2
    expected_variance = (d_vf * 0.5) ** 2 + (d_vi * 0.5) ** 2 + (d_t * 0.1) ** 2
    assert result.uncertainty == pytest.approx(math.sqrt(expected_variance), rel=1e-6)
    assert result.relative_uncertainty == pytest.approx(result.uncertainty / 2.0, rel=1e-6)


def test_propagate_uncertainty_reports_dominant_source():
    x, y = sp.symbols("x y")
    expr = x + y
    values = {x: 1.0, y: 1.0}
    uncertainties = {x: 5.0, y: 0.01}  # x should dominate overwhelmingly
    result = propagate_uncertainty(expr, values, uncertainties)
    assert result.dominant_source == "x"


def test_propagate_uncertainty_returns_none_if_no_relevant_symbols():
    x = sp.Symbol("x")
    result = propagate_uncertainty(x, {x: 1.0}, {})
    assert result is None


def test_propagate_uncertainty_ignores_zero_uncertainty():
    x, y = sp.symbols("x y")
    expr = x + y
    result = propagate_uncertainty(expr, {x: 1.0, y: 1.0}, {x: 0.0, y: 0.2})
    assert result is not None
    assert set(result.contributions.keys()) == {"y"}


def test_uncertainty_for_target_end_to_end():
    model = _kinematics_model(with_uncertainty=True)
    knowns = {k: float(v) for k, v in _known_substitutions(model).items()}
    result = uncertainty_for_target(model, "a", knowns)
    assert result is not None
    assert result.nominal == pytest.approx(2.0, abs=1e-6)
    assert result.uncertainty > 0


def test_uncertainty_for_target_none_when_no_variable_has_uncertainty():
    model = _kinematics_model(with_uncertainty=False)
    knowns = {k: float(v) for k, v in _known_substitutions(model).items()}
    result = uncertainty_for_target(model, "a", knowns)
    assert result is None


def test_uncertainty_for_target_none_for_non_algebraic_target():
    model = build_model({
        "problem_domain": "growth", "problem_type": "ode",
        "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "rate", "known_value": "0.03", "unit": "1/yr", "uncertainty": "0.001"},
            {"symbol": "P", "meaning": "population", "known_value": None, "unit": "people", "is_function": True},
        ],
        "equations": [
            {"name": "growth", "kind": "ode", "expression": "Eq(Derivative(P(t), t), k*P(t))", "derivation": ""},
        ],
        "initial_conditions": [{"expression": "P(0)", "value": "500", "note": ""}],
        "solve_for": ["P"], "assumptions": [],
    })
    knowns = {k: float(v) for k, v in _known_substitutions(model).items()}
    assert uncertainty_for_target(model, "P", knowns) is None


def test_solver_shows_uncertainty_propagation_step():
    model = _kinematics_model(with_uncertainty=True)
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["a"]]
    assert "Propagate measurement uncertainty" in descriptions


def test_solver_omits_uncertainty_step_when_not_declared():
    model = _kinematics_model(with_uncertainty=False)
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["a"]]
    assert "Propagate measurement uncertainty" not in descriptions


def test_variable_uncertainty_field_parses_from_json():
    model = _kinematics_model(with_uncertainty=True)
    v_f = next(v for v in model.variables if v.symbol == "v_f")
    a = next(v for v in model.variables if v.symbol == "a")
    assert v_f.uncertainty == pytest.approx(0.5)
    assert a.uncertainty is None
