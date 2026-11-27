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


def test_a_value_the_model_gave_is_kept_unless_the_workspace_name_was_quoted():
    payload = json.loads(json.dumps(IGNORED))
    payload["variables"][0]["known_value"] = "3"
    model = _model(payload)
    assert bind_workspace_values(model, ENTRIES, "Using accel from before ...") == []              # no reference at all
    assert next(v for v in model.variables if v.symbol == "a").known_value == 3.0
    entries = {"accel": WorkspaceEntry(name="accel", value=2.0, source="p1", unit="m/s^2")}
    payload["variables"][0]["symbol"] = "accel"
    model = _model(payload)
    assert bind_workspace_values(model, entries, "Using accel from before, how far?") == []        # only LIKELY: the model's value stands
    assert bind_workspace_values(model, entries, "Using 'accel' from the workspace, how far?") != []  # explicit: the workspace wins
    assert next(v for v in model.variables if v.symbol == "accel").known_value == 2.0
    assert any("the model had 3" in s for s in model.assumptions)


def test_a_model_that_wrote_zero_for_the_workspace_variable_is_corrected():
    payload = json.loads(json.dumps(IGNORED))
    payload["variables"][0]["known_value"] = "0"
    model = _model(payload)
    assert bind_workspace_values(model, ENTRIES, TEXT) != []
    assert next(v for v in model.variables if v.symbol == "a").known_value == 2.0


def test_the_context_does_not_repeat_the_earlier_problems_text():
    """Regression: '(previously solved from: A car accelerates ... from 8 m/s ... over 6 seconds)' led the model to use v0 = 8 and t = 6 in the NEW problem."""
    ws = Workspace({})
    ws.store("a", 2.0, "A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds", "m/s^2")
    ctx = ws.as_context_string()
    assert ctx == "- a = 2 m/s^2 (a value from an earlier calculation)"
    assert "8 m/s" not in ctx and "6 seconds" not in ctx


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


def test_the_variables_panel_says_which_value_came_from_the_workspace():
    from streamlit.testing.v1 import AppTest

    def page():
        from modules.workspace import WorkspaceEntry, bind_workspace_values
        from tests.test_workspace_binding import ENTRIES, TEXT, _model
        from ui.results.summary import render_variables
        model = _model()
        bind_workspace_values(model, ENTRIES, TEXT)
        render_variables(model)

    at = AppTest.from_function(page, default_timeout=60)
    at.run()
    assert not at.exception
    assert [c.value for c in at.caption].count("↳ from the workspace") == 1
    assert next(n for n in at.number_input if n.key == "var_a").value == 2.0


# ------------------------------------------------------------------------------------------ follow-ups: "the car from before, 10 seconds instead of 6"

from modules.workspace import bind_earlier_values, earlier_context, earlier_entries, is_follow_up   # noqa: E402

FOLLOW_UP = "Using the acceleration from before, how far would the car travel in 10 seconds instead of 6?"

# the first problem: v0 = 8, v = 20, t = 6; a and d are the targets
FIRST = {
    "problem_domain": "kinematics", "problem_type": "algebraic",
    "variables": [{"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
                  {"symbol": "t", "meaning": "time interval", "known_value": "6", "unit": "s"},
                  {"symbol": "v0", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
                  {"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
                  {"symbol": "d", "meaning": "distance traveled", "known_value": None, "unit": "m"}],
    "equations": [{"name": "v", "kind": "equation", "expression": "Eq(v, v0 + a*t)", "derivation": ""},
                  {"name": "d", "kind": "equation", "expression": "Eq(d, v0*t + a*t**2/2)", "derivation": ""}],
    "solve_for": ["a", "d"], "assumptions": [],
}
# what the model returns for the follow-up: it reads the "6" of "instead of 6", and knows neither a nor v0
FOLLOW_MODEL = {
    "problem_domain": "kinematics", "problem_type": "algebraic",
    "variables": [{"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
                  {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
                  {"symbol": "d", "meaning": "distance traveled", "known_value": None, "unit": "m"},
                  {"symbol": "v0", "meaning": "initial velocity", "known_value": None, "unit": "m/s"}],
    "equations": [{"name": "distance", "kind": "equation", "expression": "Eq(d, v0*t + a*t**2/2)", "derivation": ""}],
    "solve_for": ["d"], "assumptions": [],
}


def test_follow_up_phrases_are_recognised_and_ordinary_problems_are_not():
    for text in (FOLLOW_UP, TEXT, "Same car, but 10 s instead of 6", "Using x from the workspace", "like before, but heavier"):
        assert is_follow_up(text), text
    assert not is_follow_up("A car accelerates from 8 m/s to 20 m/s in 6 s. How far does it travel before stopping?")


def test_the_earlier_problems_givens_and_the_workspace_become_follow_up_context():
    previous = _model(FIRST)
    entries = earlier_entries(previous, ENTRIES)
    assert {k: v.value for k, v in entries.items()} == {"t": 6.0, "v0": 8.0, "v": 20.0, "a": 2.0}      # a from the workspace
    block = earlier_context("A car accelerates from 8 m/s to 20 m/s over 6 seconds. Find a and d.", entries)
    assert block.startswith("EARLIER PROBLEM") and "only where the new problem refers to it" in block and "instead of 6" in block and "v0 = 8 m/s" in block
    assert earlier_context(None, {}) is None


def test_the_missing_input_comes_from_the_earlier_problem_and_instead_of_changes_the_time():
    model = _model(FOLLOW_MODEL)
    notes = bind_earlier_values(model, earlier_entries(_model(FIRST), ENTRIES), FOLLOW_UP)
    values = {v.symbol: v.known_value for v in model.variables}
    assert values == {"a": 2.0, "t": 10.0, "d": None, "v0": 8.0}
    assert any("v0 = 8 m/s was taken from the earlier problem" in n for n in notes) and any("t is 10, not 6" in n for n in notes)
    assert all(n in model.assumptions for n in notes)


def test_nothing_is_filled_in_for_an_ordinary_problem_and_a_target_is_never_filled():
    model = _model(FOLLOW_MODEL)
    assert bind_earlier_values(model, earlier_entries(_model(FIRST), ENTRIES), "A car travels for 6 s. How far?") == []
    assert next(v for v in model.variables if v.symbol == "v0").known_value is None
    model = _model(FOLLOW_MODEL)
    bind_earlier_values(model, {"d": WorkspaceEntry(name="d", value=84.0, source="p", unit="m")}, FOLLOW_UP)
    assert next(v for v in model.variables if v.symbol == "d").known_value is None                     # d is what is asked for


def test_instead_of_changes_nothing_when_two_inputs_share_the_old_value():
    payload = json.loads(json.dumps(FOLLOW_MODEL))
    payload["variables"][3]["known_value"] = "6"            # v0 is also 6: which one does "instead of 6" mean? Do not guess.
    model = _model(payload)
    bind_earlier_values(model, {}, FOLLOW_UP)
    assert {v.symbol: v.known_value for v in model.variables if v.known_value is not None} == {"t": 6.0, "v0": 6.0}


def test_a_workspace_value_is_found_by_its_meaning_when_the_name_is_not_in_the_text():
    model = _model(FOLLOW_MODEL)
    assert bind_workspace_values(model, ENTRIES, FOLLOW_UP) != []                                       # "the acceleration from before" -> a
    assert next(v for v in model.variables if v.symbol == "a").known_value == 2.0
    model = _model(FOLLOW_MODEL)
    assert bind_workspace_values(model, ENTRIES, "A car's velocity is 5. How far in 10 s?") == []        # not a follow-up


def test_the_whole_app_solves_the_follow_up_after_the_first_problem():
    """Two solves in one session through the real page: the follow-up's inputs come from the first problem and the workspace, and its time is 10."""
    import tempfile
    from pathlib import Path
    from streamlit.testing.v1 import AppTest
    from modules import history, practice
    import modules.llm_client as lc
    tmp = Path(tempfile.mkdtemp())
    current = {"payload": json.dumps(FIRST)}

    class C(FakeClient):
        def __init__(self, *a, **k):
            super().__init__(payload_json=current["payload"], final_answers={"d": 180.0, "a": 2.0})
            self.payload_json = current["payload"]

    history.DB_PATH, practice.DB_PATH = tmp / "h.db", tmp / "p.db"
    real = lc.LMStudioClient
    lc.LMStudioClient = C                                         # type: ignore[misc,assignment]
    try:
        at = AppTest.from_file(str(Path(__file__).resolve().parent.parent / "app.py"), default_timeout=180)
        at.run()
        at.text_area[0].set_value("A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. Find a and d.").run()
        next(b for b in at.button if "Represent & Solve" in b.label).click().run()
        assert not at.exception
        at.session_state["workspace_entries"] = dict(ENTRIES)
        current["payload"] = json.dumps(FOLLOW_MODEL)
        at.text_area[0].set_value(FOLLOW_UP).run()
        next(b for b in at.button if "Represent & Solve" in b.label).click().run()
        assert not at.exception
        values = {v.symbol: v.known_value for v in at.session_state["model"].variables}
        assert values == {"a": 2.0, "t": 10.0, "d": None, "v0": 8.0}
        assert {n.key: n.value for n in at.number_input if (n.key or "").startswith("var_")} == {"var_a": 2.0, "var_t": 10.0, "var_d": 0.0, "var_v0": 8.0}
        captions = [c.value for c in at.caption]
        assert captions.count("↳ from the workspace") == 1 and captions.count("↳ from the earlier problem") == 1 and captions.count("↳ changed as the problem says") == 1
    finally:
        lc.LMStudioClient = real                                  # type: ignore[misc]
