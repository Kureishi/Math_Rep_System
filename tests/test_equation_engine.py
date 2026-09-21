import json
import sympy as sp

from modules.equation_engine import build_model, symbols_and_functions_used, target_kind, extract_model


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


def test_extract_model_passes_model_override_to_client():
    """Regression test: extract_model()'s optional `model` parameter
    (added for paranoid.py's secondary-model cross-verification) must
    reach client.chat() as its own `model` kwarg, not get silently
    dropped."""
    captured = {}

    class CapturingClient:
        def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
            captured["model"] = model
            return "{}"

    extract_model(CapturingClient(), "a problem", model="secondary-model-name")
    assert captured["model"] == "secondary-model-name"


def test_extract_model_defaults_to_no_override():
    captured = {}

    class CapturingClient:
        def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
            captured["model"] = model
            return "{}"

    extract_model(CapturingClient(), "a problem")
    assert captured["model"] is None


# ---------------------------------------------------------------- geometry extraction
def test_geometry_field_parsed_when_valid():
    raw = {
        "problem_domain": "geometry", "variables": [], "equations": [], "solve_for": [], "assumptions": [],
        "geometry": {"shape": "triangle", "knowns": {"a": "12", "b": "18", "C": "50"}},
    }
    model = build_model(raw)
    assert model.geometry is not None
    assert model.geometry.shape == "triangle"
    assert model.geometry.knowns == {"a": 12.0, "b": 18.0, "C": 50.0}


def test_geometry_field_absent_when_not_given():
    raw = {"problem_domain": "mechanics", "variables": [], "equations": [], "solve_for": [], "assumptions": []}
    model = build_model(raw)
    assert model.geometry is None


def test_geometry_field_degrades_gracefully_with_wrong_number_of_knowns():
    """Only 2 knowns given (a valid triangle needs exactly 3) -- must
    degrade to None rather than crash the whole extraction over this
    OPTIONAL field."""
    raw = {
        "problem_domain": "geometry", "variables": [], "equations": [], "solve_for": [], "assumptions": [],
        "geometry": {"shape": "triangle", "knowns": {"a": "12", "b": "18"}},
    }
    model = build_model(raw)
    assert model.geometry is None


def test_geometry_field_ignores_unrecognized_shape():
    raw = {
        "problem_domain": "geometry", "variables": [], "equations": [], "solve_for": [], "assumptions": [],
        "geometry": {"shape": "circle", "knowns": {"a": "12", "b": "18", "C": "50"}},
    }
    model = build_model(raw)
    assert model.geometry is None


def test_geometry_field_drops_unrecognized_keys_but_keeps_valid_ones():
    raw = {
        "problem_domain": "geometry", "variables": [], "equations": [], "solve_for": [], "assumptions": [],
        "geometry": {"shape": "triangle",
                      "knowns": {"a": "12", "b": "18", "C": "50", "not_a_real_key": "99"}},
    }
    model = build_model(raw)
    # the bad key drops out, leaving only 3 valid ones -- still solvable
    assert model.geometry is not None
    assert model.geometry.knowns == {"a": 12.0, "b": 18.0, "C": 50.0}


def test_geometry_field_does_not_affect_normal_equation_extraction():
    """The central compatibility guarantee: adding a geometry field
    alongside normal equations must not change how those equations are
    parsed at all."""
    raw = {
        "problem_domain": "mechanics",
        "variables": [{"symbol": "F", "meaning": "force"}, {"symbol": "m", "meaning": "mass", "known_value": 2.0},
                       {"symbol": "a", "meaning": "acceleration", "known_value": 3.0}],
        "equations": [{"name": "eq1", "expression": "Eq(F, m*a)"}],
        "solve_for": ["F"], "assumptions": [],
    }
    model = build_model(raw)
    assert model.geometry is None
    assert len(model.equations) == 1
    assert model.equations[0].sympy_eq is not None


def test_geometry_field_via_validated_llm_schema_payload():
    """End-to-end through the SAME pydantic validation gate the LLM
    boundary actually uses (see llm_schema.py), not just build_model()
    called directly with a hand-built dict."""
    from modules.llm_schema import validate_extraction_payload
    raw = {
        "problem_domain": "geometry", "equations": [],
        "geometry": {"shape": "triangle", "knowns": {"a": 12, "b": 18, "C": 50}},
    }
    validated = validate_extraction_payload(raw)
    model = build_model(validated)
    assert model.geometry is not None
    assert model.geometry.knowns == {"a": 12.0, "b": 18.0, "C": 50.0}
