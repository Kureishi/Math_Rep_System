import pytest
import sympy as sp

from modules.equation_engine import build_model
from modules.matrix_utils import build_linear_system, linear_system_view
from modules.verifier import _known_substitutions
from modules.solver import alternate_method_steps


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


def _coupled_model():
    return build_model({
        "problem_domain": "circuit", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        "solve_for": ["x", "y"], "assumptions": [],
    })


# ---------------------------------------------------------------- matrix_utils force=


def test_sequentially_solvable_system_hidden_by_default():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    assert linear_system_view(model, knowns) is None


def test_force_reveals_matrix_view_for_sequentially_solvable_system():
    """The kinematics example has only ONE equation (a alone), so it was
    never a 'system' to begin with -- force=True can't invent a system
    out of a single equation. Use a genuinely 2-equation, sequentially-
    solvable system instead to test that force bypasses the
    sequential-solvability gate specifically."""
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
            {"symbol": "d", "meaning": "displacement", "known_value": None, "unit": "m"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
            {"name": "disp", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2 + t*v_i)", "derivation": ""},
        ],
        "solve_for": ["a", "d"], "assumptions": [],
    })
    knowns = _known_substitutions(model)
    assert linear_system_view(model, knowns) is None  # default: hidden, sequentially solvable
    forced = linear_system_view(model, knowns, force=True)
    assert forced is not None  # force=True: matrix view available anyway
    assert set(forced.symbols) == {"a", "d"}


def test_genuinely_coupled_system_unaffected_by_force_flag():
    model = _coupled_model()
    knowns = _known_substitutions(model)
    default_view = linear_system_view(model, knowns)
    forced_view = linear_system_view(model, knowns, force=True)
    assert default_view is not None
    assert forced_view is not None
    assert default_view.classification == forced_view.classification


def test_build_linear_system_force_flag_passthrough():
    model = _coupled_model()
    result = build_linear_system(model.equations, ["x", "y"], {}, force=True)
    assert result is not None


# ---------------------------------------------------------------- alternate_method_steps


def test_back_substitution_check_for_single_equation_target():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    steps = alternate_method_steps(model, "a", knowns)
    assert steps is not None
    descriptions = [s.description for s in steps]
    assert any("back-substitution" in d for d in descriptions)
    assert any("matches" in d.lower() for d in descriptions)
    assert not any("MISMATCH" in d for d in descriptions)
    # single-equation target -- no Cramer's rule available
    assert not any("Cramer" in d for d in descriptions)


def test_cramers_rule_included_for_coupled_system_target():
    model = _coupled_model()
    knowns = _known_substitutions(model)
    steps = alternate_method_steps(model, "x", knowns)
    assert steps is not None
    descriptions = [s.description for s in steps]
    assert any("Cramer" in d for d in descriptions)
    # find the Cramer's rule step and confirm the numeric answer matches
    cramer_step = next(s for s in steps if "Cramer" in s.description)
    assert "11" in cramer_step.expression and "5" in cramer_step.expression  # x = 11/5


def test_alternate_method_none_for_non_algebraic_target():
    model = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    assert alternate_method_steps(model, "x", {}) is None


def test_alternate_method_none_when_target_not_in_any_equation():
    model = _kinematics_model()
    assert alternate_method_steps(model, "nonexistent", {}) is None
