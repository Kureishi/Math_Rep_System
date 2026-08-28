import json
import pytest

from modules.batch_solver import split_batch_text, solve_one, solve_batch, batch_summary, BatchItemResult


# ---------------------------------------------------------------- split_batch_text


def test_split_on_dash_delimiter():
    text = "Problem one.\n---\nProblem two.\n---\nProblem three."
    result = split_batch_text(text)
    assert result == ["Problem one.", "Problem two.", "Problem three."]


def test_split_on_blank_lines_when_no_dashes():
    text = "Problem one.\n\nProblem two,\nspanning two lines.\n\nProblem three."
    result = split_batch_text(text)
    assert result == ["Problem one.", "Problem two,\nspanning two lines.", "Problem three."]


def test_split_empty_text_returns_empty_list():
    assert split_batch_text("") == []
    assert split_batch_text("   \n  \n  ") == []


def test_split_single_problem_no_delimiter():
    assert split_batch_text("Just one problem here.") == ["Just one problem here."]


def test_split_ignores_blank_entries_between_dashes():
    text = "Problem one.\n---\n\n---\nProblem two."
    result = split_batch_text(text)
    assert result == ["Problem one.", "Problem two."]


# ---------------------------------------------------------------- solve_one / solve_batch


def _kinematics_payload():
    return json.dumps({
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


def test_solve_one_success(fake_client_factory):
    client = fake_client_factory(payload_json=_kinematics_payload(), final_answers={"a": 2.0})
    result = solve_one(client, 0, "A car problem")
    assert result.error is None
    assert result.model is not None
    assert result.report is not None
    assert result.report.passed is True
    assert result.report.sympy_numeric_answers["a"] == pytest.approx(2.0)


def test_solve_one_api_error_captured_not_raised():
    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("Engine protocol predict stream returned an error")

    result = solve_one(RaisingClient(), 0, "A problem")
    assert result.error is not None
    assert "Engine protocol" in result.error
    assert result.model is None


def test_solve_one_malformed_json_captured_not_raised(fake_client_factory):
    client = fake_client_factory(payload_json="not valid json {{{")
    result = solve_one(client, 0, "A problem")
    assert result.error is not None
    assert result.model is None


def test_solve_batch_preserves_order_and_index(fake_client_factory):
    client = fake_client_factory(payload_json=_kinematics_payload(), final_answers={"a": 2.0})
    results = solve_batch(client, ["problem A", "problem B", "problem C"])
    assert [r.index for r in results] == [0, 1, 2]
    assert [r.problem_text for r in results] == ["problem A", "problem B", "problem C"]


def test_solve_batch_progress_callback_invoked_correctly():
    client_calls = []

    class SimpleClient:
        def chat(self, **kwargs):
            return _kinematics_payload()

    progress_log = []
    solve_batch(SimpleClient(), ["p1", "p2", "p3"],
                progress_callback=lambda done, total: progress_log.append((done, total)))
    assert progress_log == [(1, 3), (2, 3), (3, 3)]


def test_solve_batch_one_bad_problem_does_not_sink_others():
    class FlakyClient:
        def chat(self, **kwargs):
            if "bad" in kwargs.get("user", ""):
                raise RuntimeError("simulated failure on the bad problem")
            return _kinematics_payload()

    results = solve_batch(FlakyClient(), ["good 1", "bad", "good 2"])
    assert results[0].error is None
    assert results[1].error is not None
    assert results[2].error is None


# ---------------------------------------------------------------- batch_summary


def test_batch_summary_all_solved(fake_client_factory):
    client = fake_client_factory(payload_json=_kinematics_payload(), final_answers={"a": 2.0})
    results = solve_batch(client, ["p1", "p2"])
    summary = batch_summary(results)
    assert summary == {"total": 2, "solved": 2, "needed_retry": 0, "failed": 0}


def test_batch_summary_counts_errors_as_failed():
    class RaisingClient:
        def chat(self, **kwargs):
            raise RuntimeError("boom")

    results = solve_batch(RaisingClient(), ["p1", "p2"])
    summary = batch_summary(results)
    assert summary["failed"] == 2
    assert summary["solved"] == 0


def test_batch_summary_handles_result_with_neither_error_nor_report():
    """Regression test: a BatchItemResult with error=None and report=None
    (not reachable via solve_one(), but defensively possible) must still
    be counted as failed, not silently vanish from every bucket."""
    results = [BatchItemResult(index=0, problem_text="orphan result")]
    summary = batch_summary(results)
    assert summary["total"] == 1
    assert summary["failed"] == 1
    assert summary["solved"] == 0
