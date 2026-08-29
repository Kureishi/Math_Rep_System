import json
import pytest

from modules.equation_engine import build_model
from modules.followup import answer_followup
from modules.verifier import verify


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


class _RoutingClient:
    """Routes based on system prompt content, matching the pattern used
    for intent classification vs conceptual answering vs extraction."""
    def __init__(self, intent_json=None, conceptual_text="A grounded conceptual answer.",
                 extraction_json="{}"):
        self.intent_json = intent_json
        self.conceptual_text = conceptual_text
        self.extraction_json = extraction_json

    def chat(self, system, user="", temperature=0.0, json_mode=False, model=None):
        if "classify" in system.lower():
            return self.intent_json if self.intent_json is not None else '{"intent": "conceptual"}'
        if "FINAL_NUMERIC_ANSWER" in (system + user):
            return "FINAL_NUMERIC_ANSWER[a]: 2.0"
        if "answering a follow-up" in system.lower():
            return self.conceptual_text
        return self.extraction_json


def _report_for(model):
    return verify(model, _RoutingClient(), "a problem")


# ---------------------------------------------------------------- what-if recompute


def test_what_if_multiply_recomputes_correctly():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json=json.dumps(
        {"intent": "what_if", "symbol": "t", "operation": "multiply", "operand": 2}))
    answer = answer_followup(client, model, report, "what if t doubles?")
    assert answer.kind == "what_if"
    assert answer.computed_value == pytest.approx(1.0)  # (20-8)/12 = 1.0


def test_what_if_add_recomputes_correctly():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json=json.dumps(
        {"intent": "what_if", "symbol": "v_i", "operation": "add", "operand": 3}))
    answer = answer_followup(client, model, report, "what if v_i increases by 3?")
    assert answer.kind == "what_if"
    assert answer.computed_value == pytest.approx(1.5)  # (20-11)/6 = 1.5


def test_what_if_set_recomputes_correctly():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json=json.dumps(
        {"intent": "what_if", "symbol": "t", "operation": "set", "operand": 3}))
    answer = answer_followup(client, model, report, "what if t were 3 seconds?")
    assert answer.kind == "what_if"
    assert answer.computed_value == pytest.approx(4.0)  # (20-8)/3 = 4.0


def test_what_if_unknown_symbol_falls_back_to_conceptual():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json=json.dumps(
        {"intent": "what_if", "symbol": "nonexistent", "operation": "multiply", "operand": 2}),
        conceptual_text="Fallback conceptual answer.")
    answer = answer_followup(client, model, report, "what if xyz doubles?")
    assert answer.kind == "conceptual"
    assert answer.text == "Fallback conceptual answer."


def test_what_if_invalid_operation_falls_back_to_conceptual():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json=json.dumps(
        {"intent": "what_if", "symbol": "t", "operation": "unsupported_op", "operand": 2}),
        conceptual_text="Fallback conceptual answer.")
    answer = answer_followup(client, model, report, "some ambiguous question")
    assert answer.kind == "conceptual"


# ---------------------------------------------------------------- conceptual answers


def test_conceptual_question_gets_grounded_answer():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json='{"intent": "conceptual"}',
                              conceptual_text="Acceleration is the rate of change of velocity.")
    answer = answer_followup(client, model, report, "why this formula?")
    assert answer.kind == "conceptual"
    assert "rate of change" in answer.text


def test_grounded_prompt_includes_equations_and_values():
    model = _kinematics_model()
    report = _report_for(model)
    captured = {}

    class CapturingClient(_RoutingClient):
        def chat(self, system, user="", temperature=0.0, json_mode=False, model=None):
            if "answering a follow-up" in system.lower():
                captured["user"] = user
            return super().chat(system, user, temperature, json_mode, model)

    client = CapturingClient(intent_json='{"intent": "conceptual"}')
    answer_followup(client, model, report, "why this formula?")
    assert "accel" in captured["user"]
    assert "v_f" in captured["user"]


# ---------------------------------------------------------------- error handling


def test_empty_question_returns_error():
    model = _kinematics_model()
    report = _report_for(model)
    answer = answer_followup(_RoutingClient(), model, report, "   ")
    assert answer.kind == "error"


def test_malformed_intent_json_falls_back_to_conceptual():
    model = _kinematics_model()
    report = _report_for(model)
    client = _RoutingClient(intent_json="not valid json at all",
                              conceptual_text="Fallback answer.")
    answer = answer_followup(client, model, report, "what if t doubles?")
    assert answer.kind == "conceptual"
    assert answer.text == "Fallback answer."


def test_api_error_on_conceptual_call_captured_not_raised():
    class RaisingClient:
        def chat(self, system, user="", temperature=0.0, json_mode=False, model=None):
            if "classify" in system.lower():
                return '{"intent": "conceptual"}'
            raise RuntimeError("engine error")

    model = _kinematics_model()
    report = _report_for(model)
    answer = answer_followup(RaisingClient(), model, report, "why this formula?")
    assert answer.kind == "error"
    assert "engine error" in answer.text


def test_intent_classification_api_error_falls_back_to_conceptual():
    """If the intent-classification call itself fails, the whole
    question should still get answered conceptually rather than
    erroring out entirely."""
    class PartiallyRaisingClient:
        def chat(self, system, user="", temperature=0.0, json_mode=False, model=None):
            if "classify" in system.lower():
                raise RuntimeError("classification call failed")
            return "Still answered conceptually."

    model = _kinematics_model()
    report = _report_for(model)
    answer = answer_followup(PartiallyRaisingClient(), model, report, "what if t doubles?")
    assert answer.kind == "conceptual"
    assert answer.text == "Still answered conceptually."
