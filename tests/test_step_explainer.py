from modules.equation_engine import build_model
from modules.solver import SolutionStep
from modules.step_explainer import explain_step


def _model():
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


def _steps():
    return [
        SolutionStep(description="Start with the formula", expression="a = (v_f - v_i) / t"),
        SolutionStep(description="Substitute known values", expression="a = (20 - 8) / 6",
                      explanation="Plug in the given numbers."),
        SolutionStep(description="Simplify", expression="a = 2.0"),
    ]


class _CapturingClient:
    """Returns a fixed reply and records every call for assertions."""

    def __init__(self, reply="a canned explanation"):
        self.reply = reply
        self.calls: list[tuple[str, str]] = []

    def chat(self, system="", user="", temperature=0.0, json_mode=False, model=None):
        self.calls.append((system, user))
        return self.reply


class _RaisingClient:
    def chat(self, **kwargs):
        raise RuntimeError("network unreachable")


# ---------------------------------------------------------------- happy path

def test_explains_a_valid_step():
    model = _model()
    client = _CapturingClient("This step plugs in the given numbers.")
    result = explain_step(client, model, _steps(), step_number=2, target_name="a")
    assert result.error is None
    assert result.text == "This step plugs in the given numbers."
    assert result.step_number == 2


def test_prompt_grounds_only_in_the_target_step_not_the_whole_derivation():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=2, target_name="a")
    _, user_prompt = client.calls[-1]
    # the targeted step's own content must appear
    assert "Substitute known values" in user_prompt
    assert "a = (20 - 8) / 6" in user_prompt
    # a DIFFERENT step's description should NOT be included -- this is
    # meant to be narrow, not a whole-derivation dump
    assert "Start with the formula" not in user_prompt
    assert "a = 2.0" not in user_prompt


def test_existing_explanation_included_when_present():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=2, target_name="a")
    _, user_prompt = client.calls[-1]
    assert "Plug in the given numbers." in user_prompt


def test_step_without_existing_explanation_handled_cleanly():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=1, target_name="a")
    _, user_prompt = client.calls[-1]
    assert "Existing explanation" not in user_prompt


# ---------------------------------------------------------------- modes

def test_default_mode_instruction():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=1, mode="default")
    _, user_prompt = client.calls[-1]
    assert "Explain what this step is doing and why." in user_prompt


def test_simpler_mode_instruction_differs_from_default():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=1, mode="simpler")
    _, user_prompt = client.calls[-1]
    assert "more simply" in user_prompt


def test_example_mode_instruction():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=1, mode="example")
    _, user_prompt = client.calls[-1]
    assert "worked mini-example" in user_prompt


def test_unknown_mode_falls_back_to_default_instruction():
    model = _model()
    client = _CapturingClient()
    explain_step(client, model, _steps(), step_number=1, mode="not_a_real_mode")
    _, user_prompt = client.calls[-1]
    assert "Explain what this step is doing and why." in user_prompt


# ---------------------------------------------------------------- error handling

def test_out_of_range_step_number_reports_error_not_crash():
    model = _model()
    client = _CapturingClient()
    result = explain_step(client, model, _steps(), step_number=99)
    assert result.error is not None
    assert client.calls == []  # never should have called the LLM


def test_zero_or_negative_step_number_reports_error():
    model = _model()
    client = _CapturingClient()
    result = explain_step(client, model, _steps(), step_number=0)
    assert result.error is not None


def test_empty_steps_list_reports_error_not_crash():
    model = _model()
    client = _CapturingClient()
    result = explain_step(client, model, [], step_number=1)
    assert result.error is not None


def test_client_exception_reported_as_error_not_raised():
    model = _model()
    result = explain_step(_RaisingClient(), model, _steps(), step_number=1)
    assert result.error is not None
    assert result.text == ""
