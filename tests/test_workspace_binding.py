"""A workspace value the problem points at must reach the model even when the language model ignored it (modules/workspace.py: bind_workspace_values)."""
import json

import pytest

from modules.equation_engine import build_model
from modules.pipeline import extract_and_verify
from modules.workspace import Workspace, WorkspaceEntry, bind_workspace_values, references_workspace_value
from tests.conftest import FakeClient

TEXT = "Using the acceleration ('a' in workspace) from before, how far would the car travel in 10 seconds instead of 6?"
ENTRIES = {"a": WorkspaceEntry(name="a", value=2.0, source="A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds", unit="m/s^2")}

# what a model that IGNORED the workspace returns: a is an unknown input, d is the target
IGNORED = {
    "problem_domain": "kinematics", "problem_type": "algebraic",
    "variables": [{"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
                  {"symbol": "t", "meaning": "time interval", "known_value": "10", "unit": "s"},
                  {"symbol": "v0", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
                  {"symbol": "d", "meaning": "distance traveled", "known_value": None, "unit": "m"}],
    "equations": [{"name": "distance", "kind": "equation", "expression": "Eq(d, v0*t + a*t**2/2)", "derivation": ""}],
    "solve_for": ["d"], "assumptions": [],
}


def _model(payload=IGNORED):
    return build_model(json.loads(json.dumps(payload)))


def test_a_quoted_workspace_name_is_filled_in_when_the_model_left_it_unknown():
    model = _model()
    notes = bind_workspace_values(model, ENTRIES, TEXT)
    a = next(v for v in model.variables if v.symbol == "a")
    assert a.known_value == 2.0 and a.unit == "m/s^2" and len(notes) == 1
    assert any("taken from the workspace" in s and "a = 2" in s for s in model.assumptions)       # visible in the result


def test_a_value_the_model_did_give_is_never_overwritten():
    payload = json.loads(json.dumps(IGNORED))
    payload["variables"][0]["known_value"] = "3"
    model = _model(payload)
    assert bind_workspace_values(model, ENTRIES, TEXT) == []
    assert next(v for v in model.variables if v.symbol == "a").known_value == 3.0


def test_a_single_letter_is_only_matched_when_quoted_because_a_is_also_an_article():
    assert references_workspace_value("A car with a speed of 5 from before", "a") == (False, False)
    assert references_workspace_value("use 'a' from the workspace", "a")[0] is True
    assert references_workspace_value("use “a” please", "a")[0] is True


def test_a_longer_name_is_matched_when_the_text_points_back_at_the_workspace():
    assert references_workspace_value("Using accel from the workspace, find d", "accel") == (False, True)
    assert references_workspace_value("Using accel from before, find d", "accel") == (False, True)
    assert references_workspace_value("An accel of 3 m/s^2, find d", "accel") == (False, False)       # no back-reference: not a workspace value


def test_an_explicitly_named_value_stops_being_a_target():
    payload = json.loads(json.dumps(IGNORED))
    payload["solve_for"] = ["a", "d"]
    model = _model(payload)
    bind_workspace_values(model, ENTRIES, TEXT)
    assert model.solve_for == ["d"]


def test_an_input_the_problem_does_not_mention_is_left_alone():
    model = _model()
    assert bind_workspace_values(model, ENTRIES, "A car accelerates at 'b' from before, how far in 10 s?") == []
    assert next(v for v in model.variables if v.symbol == "a").known_value is None
    assert bind_workspace_values(model, None, TEXT) == [] and bind_workspace_values(model, {}, TEXT) == []


def test_the_whole_pipeline_solves_the_follow_up_even_when_the_model_ignores_the_workspace():
    client = FakeClient(payload_json=json.dumps(IGNORED), final_answers={"d": 180.0})
    ws = Workspace({})
    ws.store("a", 2.0, "A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds", "m/s^2")
    derived = extract_and_verify(client, TEXT, known_context=ws.as_context_string(), workspace_entries=dict(ws.entries), max_retries=0)
    a = next(v for v in derived.model.variables if v.symbol == "a")
    assert a.known_value == 2.0
    from modules.solver import compute_steps
    steps = compute_steps(derived.model)["d"]
    assert any(s.description == "Numeric result" and s.expression.startswith("d = 180") for s in steps)


def test_the_extraction_prompt_tells_the_model_to_use_a_new_value_over_an_old_one():
    client = FakeClient(payload_json=json.dumps(IGNORED))
    from modules.equation_engine import extract_model
    extract_model(client, TEXT, known_context="- a = 2 m/s^2")
    user = client.calls[0][1]
    assert "NEW value" in user and "instead of 6" in user and "- a = 2 m/s^2" in user
