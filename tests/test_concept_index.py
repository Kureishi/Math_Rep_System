"""Tests for modules/concept_index.py -- concept tagging via
named_formulas.py's recognizer, with a domain-label fallback."""
from modules.equation_engine import build_model
from modules.concept_index import concept_tags_for_model


def _model(equations, variables, domain="mechanics"):
    raw = {"problem_domain": domain, "variables": variables, "equations": equations,
           "solve_for": [], "assumptions": []}
    return build_model(raw)


def test_recognizes_named_formula_from_context():
    model = _model(
        [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "Newton 2nd law"}],
        [{"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"}],
    )
    assert concept_tags_for_model(model) == ["Newton's second law"]


def test_falls_back_to_domain_when_nothing_recognized():
    model = _model(
        [{"name": "eq1", "expression": "Eq(q, 7*z**5 - 3)", "derivation": "made up"}],
        [{"symbol": "q", "meaning": "", "known_value": None, "unit": None},
         {"symbol": "z", "meaning": "", "known_value": 1.0, "unit": None}],
        domain="algebra",
    )
    assert concept_tags_for_model(model) == ["domain: algebra"]


def test_multiple_equations_can_yield_multiple_tags():
    model = _model(
        [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "d1"},
         {"name": "eq2", "expression": "Eq(p, m*v)", "derivation": "d2"}],
        [{"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"},
         {"symbol": "p", "meaning": "momentum", "known_value": None, "unit": "kg m/s"},
         {"symbol": "v", "meaning": "velocity", "known_value": 4.0, "unit": "m/s"}],
    )
    tags = concept_tags_for_model(model)
    assert "Newton's second law" in tags
    assert "Momentum" in tags


def test_no_domain_and_nothing_recognized_yields_no_tags():
    model = _model(
        [{"name": "eq1", "expression": "Eq(q, 7*z**5 - 3)", "derivation": "made up"}],
        [{"symbol": "q", "meaning": "", "known_value": None, "unit": None},
         {"symbol": "z", "meaning": "", "known_value": 1.0, "unit": None}],
        domain="",
    )
    assert concept_tags_for_model(model) == []
