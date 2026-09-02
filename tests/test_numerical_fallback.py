import pytest
import sympy as sp

from modules.numerical_fallback import find_numerical_roots, numerical_fallback_for_equation
from modules.equation_engine import build_model
from modules.solver import compute_steps


# ---------------------------------------------------------------- find_numerical_roots


def test_finds_root_of_transcendental_equation():
    x = sp.Symbol("x")
    expr = x + sp.sin(x) - 5
    roots = find_numerical_roots(expr, x)
    assert len(roots) >= 1
    assert roots[0].value == pytest.approx(5.617555, rel=1e-4)
    assert roots[0].residual < 1e-6
    assert roots[0].is_numerical is True


def test_finds_root_of_x_exp_x():
    x = sp.Symbol("x")
    expr = x * sp.exp(x) - 10
    roots = find_numerical_roots(expr, x)
    assert len(roots) >= 1
    assert roots[0].value == pytest.approx(1.745528, rel=1e-4)


def test_no_real_roots_returns_empty_list():
    x = sp.Symbol("x")
    expr = x ** 2 + 1  # no real roots
    roots = find_numerical_roots(expr, x)
    assert roots == []


def test_finds_multiple_distinct_roots():
    x = sp.Symbol("x")
    expr = sp.cos(x) - sp.Rational(1, 2)  # infinitely many roots -- should find distinct ones
    roots = find_numerical_roots(expr, x, max_roots=3)
    assert len(roots) >= 2
    values = [r.value for r in roots]
    for i in range(len(values)):
        for j in range(i + 1, len(values)):
            assert abs(values[i] - values[j]) > 1e-3


def test_respects_max_roots():
    x = sp.Symbol("x")
    expr = sp.cos(x) - sp.Rational(1, 2)
    roots = find_numerical_roots(expr, x, max_roots=1)
    assert len(roots) <= 1


def test_never_raises_on_pathological_expression():
    x = sp.Symbol("x")
    expr = sp.Abs(x) ** x - 5
    roots = find_numerical_roots(expr, x)
    assert isinstance(roots, list)


def test_simple_linear_equation_also_findable_numerically():
    x = sp.Symbol("x")
    expr = 2 * x - 10
    roots = find_numerical_roots(expr, x)
    assert roots[0].value == pytest.approx(5.0, abs=1e-6)


# ---------------------------------------------------------------- numerical_fallback_for_equation


def test_wrapper_finds_root_for_single_unknown_equation():
    x = sp.Symbol("x")
    eq = sp.Eq(x + sp.sin(x), 5)
    roots = numerical_fallback_for_equation(eq, x)
    assert len(roots) >= 1
    assert roots[0].value == pytest.approx(5.617555, rel=1e-4)


def test_wrapper_returns_empty_for_multi_symbol_equation():
    x, y = sp.symbols("x y")
    eq = sp.Eq(x + y, 5)
    assert numerical_fallback_for_equation(eq, x) == []


def test_wrapper_returns_empty_when_target_absent():
    x, y = sp.symbols("x y")
    eq = sp.Eq(y, 5)
    assert numerical_fallback_for_equation(eq, x) == []


# ---------------------------------------------------------------- wired into solver.py


def test_solver_shows_numerical_fallback_for_transcendental_equation():
    model = build_model({
        "problem_domain": "transcendental equation", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(x + sin(x), 5)", "derivation": ""}],
        "solve_for": ["x"], "assumptions": [],
    })
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["x"]]
    assert "No exact symbolic solution -- numerical approximation" in descriptions
    numeric_step = next(s for s in steps["x"] if s.description.startswith("No exact"))
    assert "5.617" in numeric_step.expression or "5.62" in numeric_step.expression


def test_solver_does_not_use_numerical_fallback_for_exactly_solvable_equation():
    model = build_model({
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
    steps = compute_steps(model)
    descriptions = [s.description for s in steps["a"]]
    assert "No exact symbolic solution -- numerical approximation" not in descriptions
    assert "Numeric result" in descriptions


def test_solver_no_crash_for_coupled_multi_target_system():
    model = build_model({
        "problem_domain": "coupled", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(x + sin(y), 5)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(y, 2)", "derivation": ""},
        ],
        "solve_for": ["x", "y"], "assumptions": [],
    })
    steps = compute_steps(model)
    assert "x" in steps and "y" in steps
