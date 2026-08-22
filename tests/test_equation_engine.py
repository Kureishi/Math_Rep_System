import json
import sympy as sp

from modules.equation_engine import build_model, symbols_and_functions_used, target_kind


def test_algebraic_equation_parses(kinematics_json):
    model = build_model(json.loads(kinematics_json))
    eq = model.equations[0]
    assert eq.kind == "equation"
    assert eq.parse_error is None
    assert eq.sympy_eq == sp.Eq(sp.Symbol("v"), sp.Symbol("u") + sp.Symbol("a") * sp.Symbol("t"))


def test_known_values_parsed_as_floats(kinematics_json):
    model = build_model(json.loads(kinematics_json))
    v = next(v for v in model.variables if v.symbol == "v")
    a = next(v for v in model.variables if v.symbol == "a")
    assert v.known_value == 20.0
    assert a.known_value is None


def test_solve_for_normalizes_list():
    model = build_model({
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [], "solve_for": ["a", "d"], "assumptions": [],
    })
    assert model.solve_for == ["a", "d"]


def test_solve_for_normalizes_comma_joined_string():
    """This is a real bug that showed up in practice: a model returning
    'solve_for': 'a, d' as one string instead of a JSON array."""
    model = build_model({
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [], "solve_for": "a, d", "assumptions": [],
    })
    assert model.solve_for == ["a", "d"]


def test_solve_for_normalizes_null():
    model = build_model({
        "variables": [], "equations": [], "solve_for": None, "assumptions": [],
    })
    assert model.solve_for == []


def test_inequality_parses_as_relational(inequality_json):
    model = build_model(json.loads(inequality_json))
    eq = model.equations[0]
    assert eq.kind == "inequality"
    assert eq.parse_error is None
    assert isinstance(eq.sympy_eq, sp.core.relational.Relational)
    assert not isinstance(eq.sympy_eq, sp.Eq)


def test_inequality_rejects_equality_syntax():
    """An equation wrongly tagged as an inequality should fail to parse
    cleanly rather than silently accepting it."""
    model = build_model({
        "variables": [{"symbol": "v", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "bad", "kind": "inequality", "expression": "Eq(v, 5)", "derivation": "x"}],
        "solve_for": [], "assumptions": [],
    })
    eq = model.equations[0]
    assert eq.sympy_eq is None
    assert eq.parse_error is not None


def test_ode_parses_with_function_binding(ode_json):
    model = build_model(json.loads(ode_json))
    eq = model.equations[0]
    assert eq.kind == "ode"
    assert eq.parse_error is None
    assert model.problem_type == "ode"
    assert model.independent_variable == "t"
    # the function should be bound to sp.Function, not sp.Symbol
    from sympy.core.function import AppliedUndef
    assert eq.sympy_eq.atoms(AppliedUndef)


def test_ode_initial_condition_parses(ode_json):
    model = build_model(json.loads(ode_json))
    assert len(model.initial_conditions) == 1
    ic = model.initial_conditions[0]
    assert ic.parse_error is None
    assert ic.value == 500.0


def test_symbols_and_functions_used_includes_function_names(ode_json):
    model = build_model(json.loads(ode_json))
    used = symbols_and_functions_used(model.equations[0])
    assert "N" in used  # the function name, not just its argument
    assert "t" in used
    assert "k" in used


def test_target_kind_dispatches_correctly(kinematics_json, inequality_json, ode_json):
    algebraic_model = build_model(json.loads(kinematics_json))
    assert target_kind(algebraic_model, "a") == "equation"

    ineq_model = build_model(json.loads(inequality_json))
    assert target_kind(ineq_model, "v") == "inequality"

    ode_model = build_model(json.loads(ode_json))
    assert target_kind(ode_model, "N") == "ode"


def test_unparseable_equation_records_error():
    model = build_model({
        "variables": [{"symbol": "x", "meaning": "x", "known_value": None, "unit": None}],
        "equations": [{"name": "bad", "kind": "equation", "expression": "Eq(x, ***)", "derivation": "x"}],
        "solve_for": [], "assumptions": [],
    })
    eq = model.equations[0]
    assert eq.sympy_eq is None
    assert eq.parse_error is not None


def test_two_target_coupled_system(kinematics_two_target_json):
    model = build_model(json.loads(kinematics_two_target_json))
    assert model.solve_for == ["a", "d"]
    assert all(e.parse_error is None for e in model.equations)
