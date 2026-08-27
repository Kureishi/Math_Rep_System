import pytest
import sympy as sp

from modules.equation_engine import build_model
from modules.code_export import formula_for_target, generate_python_function, generate_python_module


def _kinematics_model():
    return build_model({
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


# ---------------------------------------------------------------- formula_for_target


def test_formula_for_algebraic_target():
    model = _kinematics_model()
    f = formula_for_target(model, "a")
    assert f is not None
    assert f.kind == "algebraic"
    assert set(f.arg_names) == {"t", "v_f", "v_i"}


def test_formula_for_target_with_coupled_unknown_substitutes_it_away():
    """d depends on 'a', which is ALSO unknown -- the exported formula
    for d should be entirely in terms of knowns (v_f, v_i, t), not
    reference 'a' at all, since sp.solve() resolves the whole system
    simultaneously."""
    model = _kinematics_model()
    f = formula_for_target(model, "d")
    assert f is not None
    assert "a" not in f.arg_names
    assert set(f.arg_names) == {"t", "v_f", "v_i"}


def test_formula_for_ode_target():
    model = build_model({
        "problem_domain": "cooling", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "cooling rate", "known_value": "0.05", "unit": "1/min"},
            {"symbol": "T", "meaning": "temperature", "known_value": None, "unit": "C", "is_function": True},
        ],
        "equations": [
            {"name": "cooling", "kind": "ode", "expression": "Eq(Derivative(T(t), t), -k*(T(t) - 20))",
             "derivation": ""},
        ],
        "initial_conditions": [{"expression": "T(0)", "value": "90", "note": ""}],
        "solve_for": ["T"], "assumptions": [],
    })
    f = formula_for_target(model, "T")
    assert f is not None
    assert f.kind == "ode"
    assert f.independent_var == "t"
    assert "t" in f.arg_names and "k" in f.arg_names


def test_formula_for_recurrence_target():
    model = build_model({
        "problem_domain": "savings", "problem_type": "recurrence", "independent_variable": "n",
        "variables": [
            {"symbol": "a", "meaning": "balance", "known_value": None, "unit": "USD", "is_function": True},
        ],
        "equations": [
            {"name": "growth", "kind": "recurrence", "expression": "Eq(a(n+1), 1.05*a(n) + 200)",
             "derivation": ""},
        ],
        "initial_conditions": [{"expression": "a(0)", "value": "1000", "note": ""}],
        "solve_for": ["a"], "assumptions": [],
    })
    f = formula_for_target(model, "a")
    assert f is not None
    assert f.kind == "recurrence"
    assert f.independent_var == "n"


def test_formula_for_optimization_target_is_none():
    model = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    assert formula_for_target(model, "x") is None


# ---------------------------------------------------------------- generate_python_function


def test_generated_function_is_syntactically_valid_and_executes_correctly():
    model = _kinematics_model()
    f = formula_for_target(model, "a")
    src = generate_python_function(f, {v.symbol: v.meaning for v in model.variables}, "m/s^2")
    ns = {}
    exec(compile(src, "gen.py", "exec"), ns)
    result = ns["a"](t=6.0, v_f=20.0, v_i=8.0)
    assert result == pytest.approx(2.0)


def test_generated_function_includes_math_import_when_needed():
    x = sp.Symbol("x")
    from modules.code_export import ExportableFormula
    f = ExportableFormula("f", sp.sqrt(x), ["x"], "algebraic")
    src = generate_python_function(f)
    assert "import math" in src
    ns = {}
    exec(compile(src, "gen.py", "exec"), ns)
    assert ns["f"](x=16.0) == pytest.approx(4.0)


def test_generated_function_omits_math_import_when_not_needed():
    x, y = sp.symbols("x y")
    from modules.code_export import ExportableFormula
    f = ExportableFormula("f", x + y, ["x", "y"], "algebraic")
    src = generate_python_function(f)
    assert "import math" not in src


# ---------------------------------------------------------------- generate_python_module


def test_generated_module_bundles_all_targets_and_executes():
    model = _kinematics_model()
    src = generate_python_module(model)
    ns = {}
    exec(compile(src, "formulas.py", "exec"), ns)
    assert ns["a"](t=6.0, v_f=20.0, v_i=8.0) == pytest.approx(2.0)
    assert ns["d"](t=6.0, v_f=20.0, v_i=8.0) == pytest.approx(84.0)


def test_generated_module_demo_block_uses_known_values():
    model = _kinematics_model()
    src = generate_python_module(model)
    assert "v_f=20.0" in src
    assert "v_i=8.0" in src
    assert "t=6.0" in src


def test_generated_module_reports_no_formula_gracefully():
    model = build_model({
        "problem_domain": "optimization", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [],
        "objective": {"expression": "-2*x**2+120*x-800", "direction": "maximize", "optimize_over": ["x"]},
        "solve_for": ["x"], "assumptions": [],
    })
    src = generate_python_module(model)
    assert "No exportable closed-form formula" in src
    # should still be valid (if inert) Python
    exec(compile(src, "gen.py", "exec"), {})
