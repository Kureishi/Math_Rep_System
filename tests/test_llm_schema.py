"""Tests for modules/llm_schema.py -- pydantic validation at the LLM
JSON extraction boundary."""
import pytest
from pydantic import ValidationError

from modules.llm_schema import validate_extraction_payload
from modules.equation_engine import build_model


def test_valid_payload_passes_through_unchanged_in_substance():
    payload = {
        "problem_domain": "mechanics",
        "variables": [{"symbol": "F", "meaning": "force"}, {"symbol": "m", "known_value": 2.0}],
        "equations": [{"name": "eq1", "expression": "Eq(F, m*a)"}],
        "solve_for": ["F"],
    }
    result = validate_extraction_payload(payload)
    assert result["problem_domain"] == "mechanics"
    assert result["equations"][0]["expression"] == "Eq(F, m*a)"
    assert result["variables"][1]["known_value"] == 2.0


def test_missing_equation_expression_raises_clear_validation_error():
    payload = {"variables": [{"symbol": "F"}], "equations": [{"name": "eq1"}]}
    with pytest.raises(ValidationError) as exc_info:
        validate_extraction_payload(payload)
    errors = exc_info.value.errors()
    assert any("expression" in str(e["loc"]) for e in errors)


def test_missing_variable_symbol_raises_clear_validation_error():
    payload = {"variables": [{"meaning": "force"}], "equations": [{"expression": "Eq(F,1)"}]}
    with pytest.raises(ValidationError) as exc_info:
        validate_extraction_payload(payload)
    errors = exc_info.value.errors()
    assert any("symbol" in str(e["loc"]) for e in errors)


def test_top_level_list_instead_of_object_raises_validation_error():
    with pytest.raises(ValidationError):
        validate_extraction_payload([1, 2, 3])


def test_tolerates_numeric_string_known_value():
    """Matches build_model()'s own existing tolerance -- some models
    emit "5" instead of 5 for a known value, and that's not treated as
    malformed."""
    payload = {"variables": [{"symbol": "m", "known_value": "5"}],
               "equations": [{"expression": "Eq(x,1)"}]}
    result = validate_extraction_payload(payload)
    assert result["variables"][0]["known_value"] == "5"


def test_tolerates_extra_unrecognized_fields():
    """The LLM adding a field build_model() doesn't use shouldn't fail
    validation for that alone -- only missing REQUIRED fields should."""
    payload = {
        "variables": [{"symbol": "F", "some_field_we_dont_use": "whatever"}],
        "equations": [{"expression": "Eq(F,1)"}],
        "a_whole_extra_top_level_field": {"nested": "data"},
    }
    result = validate_extraction_payload(payload)
    assert result["variables"][0]["symbol"] == "F"


def test_missing_optional_sections_use_the_same_defaults_as_build_model():
    """A minimal payload with none of the optional sections (no
    initial_conditions, no objective, no assumptions) should validate
    the same way build_model()'s own .get(key, default) calls always
    handled it."""
    payload = {"equations": [{"expression": "Eq(x, 1)"}]}
    result = validate_extraction_payload(payload)
    assert result["problem_domain"] == "unspecified"
    assert result["initial_conditions"] == []
    assert result["objective"] is None
    assert result["assumptions"] == []


def test_validated_payload_still_builds_a_working_model():
    """End-to-end: a payload that passes validation must still build
    into a fully functional ProblemModel through build_model() exactly
    as before -- this is a validation GATE, not a rewrite of
    build_model()'s own parsing."""
    payload = {
        "problem_domain": "mechanics",
        "variables": [
            {"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
            {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
            {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"},
        ],
        "equations": [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "Newton 2nd law"}],
        "solve_for": ["F"], "assumptions": [],
    }
    validated = validate_extraction_payload(payload)
    model = build_model(validated)
    assert model.problem_domain == "mechanics"
    assert len(model.equations) == 1
    assert model.equations[0].sympy_eq is not None
    assert [v.symbol for v in model.variables] == ["F", "m", "a"]
