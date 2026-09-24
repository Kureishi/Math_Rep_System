"""
modules/pipeline.py -- the extract -> verify -> retry -> steps -> (narrate) ->
(scenarios) sequence app.py's word-problem page and batch_solver.py both call.

Before this module existed the retry loop lived inline in app.py's Streamlit
script code and was exercised only by the interactive app itself (never by
pytest) -- these tests are new coverage, not a port of something pre-existing.
"""

from modules.pipeline import extract_and_verify, run_pipeline
from tests.conftest import FakeClient, KINEMATICS_BUGGY_JSON, KINEMATICS_JSON


class SelfCorrectingClient(FakeClient):
    """Returns KINEMATICS_BUGGY_JSON (fails verification: missing *t term) on
    the first extraction call, then the correct KINEMATICS_JSON once the
    retry-suffix text (added to the system prompt when extract_model() is
    called with retry_reason set) shows up -- i.e. it "corrects itself" only
    once told what was wrong, the same way retrying a real model is meant to
    help."""

    def __init__(self):
        super().__init__(payload_json=KINEMATICS_BUGGY_JSON, final_answers={"a": 2.0})
        self.extraction_calls = 0

    def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
        if "FINAL_NUMERIC_ANSWER" not in (system + user) and "explain math steps" not in system.lower() \
                and "worksheet" not in system.lower() and "creative but mathematically" not in system.lower():
            self.extraction_calls += 1
            if "failed a consistency check" in system:
                return KINEMATICS_JSON
        return super().chat(system, user, temperature, json_mode, model)


class AlwaysWrongClient(FakeClient):
    """Never self-corrects -- every extraction, retried or not, returns the
    same buggy payload. Used to test that the retry loop gives up after
    max_retries rather than looping forever."""

    def __init__(self):
        super().__init__(payload_json=KINEMATICS_BUGGY_JSON, final_answers={"a": 2.0})


# ---------------------------------------------------------------- extract_and_verify


def test_first_attempt_passes_needs_no_retry():
    client = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    result = extract_and_verify(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.")
    assert result.report.passed
    assert result.retries == 0


def test_failed_verification_retries_and_recovers():
    client = SelfCorrectingClient()
    result = extract_and_verify(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.")
    assert result.report.passed
    assert result.retries == 1
    assert client.extraction_calls == 2  # the bad attempt, then the corrected one


def test_persistent_failure_returns_failed_report_without_raising():
    """A model that never corrects itself: the loop stops at max_retries and
    hands back the LAST attempt's (still-failing) report, rather than
    raising -- the caller decides how to present a derivation that never
    verified."""
    client = AlwaysWrongClient()
    result = extract_and_verify(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.", max_retries=2)
    assert not result.report.passed
    assert result.retries == 2


def test_max_retries_zero_never_retries_even_on_failure():
    client = SelfCorrectingClient()
    result = extract_and_verify(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.", max_retries=0)
    assert not result.report.passed
    assert result.retries == 0
    assert client.extraction_calls == 1


def test_known_context_is_forwarded_to_extraction():
    """known_context (workspace values) must reach the extraction call, not
    get silently dropped by the new wrapper -- checked via CapturingClient-
    style inspection of what extract_model actually saw."""
    captured = {}

    class CapturingClient(FakeClient):
        def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
            if "FINAL_NUMERIC_ANSWER" not in (system + user):
                captured["user"] = user
            return super().chat(system, user, temperature, json_mode, model)

    client = CapturingClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    extract_and_verify(client, "Some problem.", known_context="- d = 84 m (previously solved)")
    assert "d = 84 m" in captured["user"]


# ---------------------------------------------------------------- run_pipeline


def test_run_pipeline_computes_steps_and_narrates_by_default():
    client = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    result = run_pipeline(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.")
    assert result.report.passed
    assert "a" in result.steps
    assert len(result.steps["a"]) > 0
    # narration ran (default narrate=True): at least one extra call beyond
    # extraction + verification + final-answer for the narration prompt
    narration_calls = [c for c in client.calls if "explain math steps" in c[0].lower()]
    assert narration_calls


def test_run_pipeline_narrate_false_skips_the_narration_call():
    client = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    run_pipeline(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.",
                 narrate=False, with_scenarios=False)
    narration_calls = [c for c in client.calls if "explain math steps" in c[0].lower()]
    scenario_calls = [c for c in client.calls if "creative but mathematically" in c[0].lower()]
    assert not narration_calls
    assert not scenario_calls


def test_run_pipeline_with_scenarios_true_generates_scenarios():
    client = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    result = run_pipeline(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.",
                           narrate=False, with_scenarios=True)
    assert result.scenarios


def test_run_pipeline_stage_hook_is_called_for_each_stage():
    """app.py passes st.spinner as `stage`; here a plain recorder proves the
    hook fires around each stage without needing Streamlit at all."""
    from contextlib import contextmanager
    labels = []

    @contextmanager
    def recording_stage(label):
        labels.append(label)
        yield

    client = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    run_pipeline(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.", stage=recording_stage)
    assert any("Deriving" in label for label in labels)
    assert any("Verifying" in label for label in labels)
    assert any("Computing" in label for label in labels)


def test_run_pipeline_propagates_retries_from_extraction():
    client = SelfCorrectingClient()
    result = run_pipeline(client, "A car accelerates from 8 m/s to 20 m/s in 6 s.",
                           narrate=False, with_scenarios=False)
    assert result.retries == 1
    assert result.report.passed
