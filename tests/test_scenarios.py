import json

from modules.equation_engine import build_model
from modules.scenarios import generate_alternative_scenarios


def _model(fake_client_factory):
    payload = {
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v", "meaning": "v", "known_value": "20", "unit": "m/s"},
            {"symbol": "u", "meaning": "u", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "t", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "a", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "eq", "kind": "equation", "expression": "Eq(v, u + a*t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    }
    return build_model(payload)


def test_generate_scenarios_happy_path(fake_client_factory):
    model = _model(fake_client_factory)
    client = fake_client_factory(scenarios='[{"scenario": "s1", "mapping": "m1"}]')
    result = generate_alternative_scenarios(client, model)
    assert result == [{"scenario": "s1", "mapping": "m1"}]


def test_generate_scenarios_tolerates_malformed_json(fake_client_factory):
    model = _model(fake_client_factory)
    client = fake_client_factory(scenarios="not valid json at all")
    result = generate_alternative_scenarios(client, model)
    assert len(result) == 1
    assert "error" in result[0]
    assert "raw" in result[0]


def test_generate_scenarios_survives_api_error(fake_client_factory):
    """Regression test: an exception raised by client.chat() itself (e.g.
    a BadRequestError from the LLM backend/engine) must not propagate --
    the equations/verification/solution have already succeeded by the
    time this runs, so a failure here should degrade gracefully instead
    of crashing the whole pipeline."""
    model = _model(fake_client_factory)

    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("Engine protocol predict stream returned an error")

    result = generate_alternative_scenarios(RaisingClient(), model)
    assert len(result) == 1
    assert "error" in result[0]
    assert "Engine protocol predict stream returned an error" in result[0]["error"]


def test_generate_scenarios_tolerates_missing_keys(fake_client_factory):
    model = _model(fake_client_factory)
    client = fake_client_factory(scenarios='[{"scenario": "only scenario, no mapping"}]')
    result = generate_alternative_scenarios(client, model)
    assert result[0]["scenario"] == "only scenario, no mapping"
    assert result[0]["mapping"] == ""
