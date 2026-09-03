from modules.equation_engine import build_model
from modules.worksheet import generate_worksheet_problems, generate_targeted_worksheet_problems


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


def test_generates_requested_count(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["problem one", "problem two", "problem three"]')
    problems = generate_worksheet_problems(client, model, count=3)
    assert problems == ["problem one", "problem two", "problem three"]


def test_truncates_to_requested_count(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["p1", "p2", "p3", "p4", "p5"]')
    problems = generate_worksheet_problems(client, model, count=2)
    assert len(problems) == 2


def test_count_clamped_to_valid_range(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["p1"]')
    # should not raise even with out-of-range counts
    generate_worksheet_problems(client, model, count=0)
    generate_worksheet_problems(client, model, count=100)


def test_filters_out_non_string_entries(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["good problem", 42, null, "another good one"]')
    problems = generate_worksheet_problems(client, model, count=5)
    assert problems == ["good problem", "another good one"]


def test_malformed_json_returns_empty_list_not_exception(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet="not valid json at all")
    assert generate_worksheet_problems(client, model) == []


def test_non_array_json_returns_empty_list(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='{"not": "an array"}')
    assert generate_worksheet_problems(client, model) == []


def test_api_error_returns_empty_list_not_exception():
    model = _kinematics_model()

    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("Engine protocol predict stream returned an error")

    assert generate_worksheet_problems(RaisingClient(), model) == []


def test_difficulty_note_included_in_prompt(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["p1"]')
    generate_worksheet_problems(client, model, count=1, difficulty="harder")
    _, user_prompt = client.calls[-1]
    assert "less round" in user_prompt or "extra" in user_prompt


def test_blank_and_whitespace_entries_filtered_out(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["real problem", "", "   ", "another real one"]')
    problems = generate_worksheet_problems(client, model, count=5)
    assert problems == ["real problem", "another real one"]


# ---------------------------------------------------------------- targeted generation

def test_targeted_without_patterns_falls_back_to_plain_generation(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["p1"]')
    problems = generate_targeted_worksheet_problems(client, model, patterns=[], count=1)
    assert problems == ["p1"]
    _, user_prompt = client.calls[-1]
    assert "struggled with" not in user_prompt


def test_targeted_with_patterns_includes_focus_note_in_prompt(fake_client_factory):
    model = _kinematics_model()
    client = fake_client_factory(worksheet='["p1"]')
    patterns = ["You've made a sign-error 3 times this week."]
    problems = generate_targeted_worksheet_problems(client, model, patterns=patterns, count=1)
    assert problems == ["p1"]
    _, user_prompt = client.calls[-1]
    assert "struggled with" in user_prompt
    assert "sign-error" in user_prompt


def test_targeted_generation_still_returns_empty_list_on_api_error():
    model = _kinematics_model()

    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("boom")

    result = generate_targeted_worksheet_problems(
        RaisingClient(), model, patterns=["some pattern"], count=1)
    assert result == []
