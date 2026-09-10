"""Tests for modules/ode_utils.py's numerical_cross_check -- a second,
independent (numerical integration) verification path for first-order
initial-value ODE problems, alongside the existing symbolic
checkodesol-style check in verifier.py's _ode_checks."""
import sympy as sp

from modules.equation_engine import build_model
from modules.ode_utils import solve_ode, group_coupled_odes, numerical_cross_check


def _single_decay_model(k=0.5, n0=100.0):
    raw = {
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "decay rate", "known_value": k, "unit": None, "is_function": False},
            {"symbol": "N", "meaning": "quantity", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))",
                        "derivation": "d"}],
        "solve_for": ["N"], "assumptions": [],
        "initial_conditions": [{"expression": "N(0)", "value": n0}],
    }
    return build_model(raw)


def _coupled_decay_chain_model():
    raw = {
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k1", "meaning": "rate1", "known_value": 0.3, "unit": None, "is_function": False},
            {"symbol": "k2", "meaning": "rate2", "known_value": 0.1, "unit": None, "is_function": False},
            {"symbol": "A", "meaning": "A", "known_value": None, "unit": None, "is_function": True},
            {"symbol": "B", "meaning": "B", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [
            {"name": "A decay", "kind": "ode", "expression": "Eq(Derivative(A(t), t), -k1*A(t))",
             "derivation": "d"},
            {"name": "B production", "kind": "ode",
             "expression": "Eq(Derivative(B(t), t), k1*A(t) - k2*B(t))", "derivation": "d"},
        ],
        "solve_for": ["A", "B"], "assumptions": [],
        "initial_conditions": [{"expression": "A(0)", "value": 50.0}, {"expression": "B(0)", "value": 10.0}],
    }
    return build_model(raw)


def _second_order_shm_model():
    raw = {
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "w", "meaning": "omega", "known_value": 2.0, "unit": None, "is_function": False},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [{"name": "shm", "kind": "ode",
                        "expression": "Eq(Derivative(y(t), t, 2) + w**2*y(t), 0)", "derivation": "d"}],
        "solve_for": ["y"], "assumptions": [],
        "initial_conditions": [{"expression": "y(0)", "value": 1.0},
                                 {"expression": "Derivative(y(t), t).subs(t, 0)", "value": 0.0}],
    }
    return build_model(raw)


def test_correct_single_ode_solution_confirmed_by_independent_integration():
    model = _single_decay_model()
    solutions = solve_ode(model)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    result = numerical_cross_check(model, groups[0], solutions)
    assert result.applicable
    assert result.ok
    assert result.max_relative_error < 1e-4


def test_correct_coupled_system_confirmed_by_independent_integration():
    model = _coupled_decay_chain_model()
    solutions = solve_ode(model)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    assert len(groups) == 1
    result = numerical_cross_check(model, groups[0], solutions)
    assert result.applicable
    assert result.ok


def test_deliberately_wrong_solution_is_caught():
    """This is the central claim of this rigor upgrade: a solution that
    satisfies the ODE's algebraic FORM but doesn't match the actual
    initial condition (simulated here directly by substituting a
    sign-flipped closed form) must be flagged, not silently passed."""
    model = _single_decay_model(k=0.5, n0=100.0)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    t = sp.Symbol("t")
    N = sp.Function("N")
    wrong_solutions = {"N": sp.Eq(N(t), 100.0 * sp.exp(0.5 * t))}  # sign-flipped
    result = numerical_cross_check(model, groups[0], wrong_solutions)
    assert result.applicable
    assert not result.ok
    assert result.max_relative_error > 0.1


def test_second_order_ode_out_of_scope_reported_honestly():
    model = _second_order_shm_model()
    solutions = solve_ode(model)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    result = numerical_cross_check(model, groups[0], solutions)
    assert not result.applicable
    assert "first-order" in result.reason.lower()


def test_missing_initial_condition_out_of_scope_reported_honestly():
    raw = {
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "decay rate", "known_value": 0.5, "unit": None, "is_function": False},
            {"symbol": "N", "meaning": "quantity", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))",
                        "derivation": "d"}],
        "solve_for": ["N"], "assumptions": [], "initial_conditions": [],
    }
    model = build_model(raw)
    solutions = solve_ode(model)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    result = numerical_cross_check(model, groups[0], solutions)
    assert not result.applicable
    assert "initial condition" in result.reason.lower()


def test_unknown_parameter_out_of_scope_reported_honestly():
    """The rate constant k has no known_value here -- there's nothing
    numeric to integrate, and this must be reported as such rather than
    crashing trying to lambdify a free symbol."""
    raw = {
        "problem_domain": "physics", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "decay rate", "known_value": None, "unit": None, "is_function": False},
            {"symbol": "N", "meaning": "quantity", "known_value": None, "unit": None, "is_function": True},
        ],
        "equations": [{"name": "decay", "kind": "ode", "expression": "Eq(Derivative(N(t), t), -k*N(t))",
                        "derivation": "d"}],
        "solve_for": ["N"], "assumptions": [],
        "initial_conditions": [{"expression": "N(0)", "value": 100.0}],
    }
    model = build_model(raw)
    solutions = solve_ode(model)
    groups = group_coupled_odes([e for e in model.equations if e.kind == "ode"])
    result = numerical_cross_check(model, groups[0], solutions)
    assert not result.applicable
    assert "known numeric value" in result.reason.lower()
