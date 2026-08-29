import json
import pytest

from modules.equation_engine import build_model
from modules.paranoid import run_paranoid_check


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


def _payload():
    return {
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
    }


class _FixedClient:
    def __init__(self, payload_json):
        self.payload_json = payload_json

    def chat(self, **kwargs):
        return self.payload_json


class _RaisingClient:
    def chat(self, **kwargs):
        raise RuntimeError("secondary model not loaded")


def test_no_secondary_model_configured_does_not_run(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory()
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0})
    assert result.ran is False
    assert result.secondary_model is None


def test_secondary_model_agreeing_reports_no_disagreements():
    model = _kinematics_model()
    client = _FixedClient(json.dumps(_payload()))
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0}, secondary_model_name="other-model")
    assert result.ran is True
    assert result.secondary_model == "other-model"
    assert result.equations_match == pytest.approx(1.0)
    assert result.disagreements == {}
    assert result.secondary_answers["a"] == pytest.approx(2.0)


def test_secondary_model_different_formula_flags_disagreement():
    model = _kinematics_model()
    wrong_payload = _payload()
    wrong_payload["equations"] = [
        {"name": "accel", "kind": "equation", "expression": "Eq(a, v_f/t)", "derivation": ""},
    ]
    client = _FixedClient(json.dumps(wrong_payload))
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0}, secondary_model_name="other-model")
    assert result.ran is True
    assert result.equations_match == pytest.approx(0.0)
    assert "a" in result.disagreements
    primary_val, secondary_val = result.disagreements["a"]
    assert primary_val == pytest.approx(2.0)
    assert secondary_val == pytest.approx(20 / 6)


def test_secondary_model_within_tolerance_not_flagged():
    """A tiny floating-point-level difference (within cross_check_tolerance)
    shouldn't be flagged as a disagreement."""
    model = _kinematics_model()
    slightly_off_payload = _payload()
    client = _FixedClient(json.dumps(slightly_off_payload))
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0000001},
                                  secondary_model_name="other-model")
    assert result.disagreements == {}


def test_secondary_model_api_error_captured_not_raised():
    model = _kinematics_model()
    result = run_paranoid_check(_RaisingClient(), "a problem", model, {"a": 2.0},
                                  secondary_model_name="missing-model")
    assert result.ran is True
    assert result.error is not None
    assert "secondary model not loaded" in result.error


def test_secondary_model_malformed_json_captured_not_raised():
    model = _kinematics_model()
    client = _FixedClient("not valid json {{{")
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0}, secondary_model_name="other-model")
    assert result.ran is True
    assert result.error is not None


def test_falls_back_to_settings_secondary_model_when_not_given(monkeypatch):
    from config import settings
    model = _kinematics_model()
    client = _FixedClient(json.dumps(_payload()))
    monkeypatch.setattr(settings, "secondary_reasoning_model", "configured-model")
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0})
    assert result.ran is True
    assert result.secondary_model == "configured-model"


def test_disagreement_only_reported_for_shared_targets():
    """If the secondary model doesn't solve for a target the primary
    did, that's not itself a 'disagreement' -- just nothing to compare."""
    model = _kinematics_model()
    payload_no_target = _payload()
    payload_no_target["solve_for"] = []  # secondary produces no numeric answers at all
    client = _FixedClient(json.dumps(payload_no_target))
    result = run_paranoid_check(client, "a problem", model, {"a": 2.0}, secondary_model_name="other-model")
    assert result.disagreements == {}
