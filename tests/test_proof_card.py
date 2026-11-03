"""Proof-carrying answers: the equation audit, the step badges, the trust level and its stated limits, and replay."""
import json

import pytest

from modules import proof_card as pc
from modules.equation_engine import build_model
from modules.solve_compare import deterministic_report
from modules.verifier import CheckResult, VerificationReport
from tests.conftest import KINEMATICS_BUGGY_JSON, KINEMATICS_TWO_TARGET_JSON


@pytest.fixture(scope="module")
def good():
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    return model, deterministic_report(model)


def _with_cross_check(report, agree=True):
    """The same report as if the independent re-solve had run (it asks a language model, so it is added by hand)."""
    out = VerificationReport(checks=list(report.checks), sympy_numeric_answers=dict(report.sympy_numeric_answers))
    for target in out.sympy_numeric_answers:
        out.add(f"Independent cross-check: {target}", agree, "agrees" if agree else "DISAGREE", margin_ratio=0.01 if agree else None)
    return out


# ------------------------------------------------------------------------------------------ equation audit

def test_a_correct_answer_satisfies_every_equation(good):
    model, report = good
    audits = pc.audit_equations(model, report.sympy_numeric_answers)
    assert [a.status for a in audits] == ["verified", "verified"] and all(a.residual < 1e-9 for a in audits)


def test_a_wrong_answer_is_caught_by_the_audit_whatever_the_other_checks_said(good):
    model, report = good
    wrong = {**report.sympy_numeric_answers, "a": report.sympy_numeric_answers["a"] * 1.1}
    audits = pc.audit_equations(model, wrong)
    assert any(a.status == "failed" and a.residual > 1e-3 for a in audits)


def test_an_equation_with_an_unsolved_unknown_is_unchecked_not_passed(good):
    model, _ = good
    audits = pc.audit_equations(model, {})
    assert {a.status for a in audits} == {"unchecked"} and "no value" in audits[0].detail


# ------------------------------------------------------------------------------------------ trust level

def test_verified_needs_the_independent_resolve_to_agree(good):
    model, report = good
    assert pc.trust_summary(model, _with_cross_check(report)).level == "verified"


def test_without_an_independent_resolve_the_answer_is_caution_with_the_reason_stated(good):
    model, report = good
    summary = pc.trust_summary(model, report)
    assert summary.level == "caution" and any("independent re-solve" in s for s in summary.limits)


def test_a_disagreeing_resolve_fails_the_answer(good):
    model, report = good
    summary = pc.trust_summary(model, _with_cross_check(report, agree=False))
    assert summary.level == "failed" and summary.score < 0.5


def test_a_wrong_answer_is_failed_even_when_every_recorded_check_passed(good):
    model, report = good
    tampered = VerificationReport(checks=list(report.checks),
                                  sympy_numeric_answers={**report.sympy_numeric_answers, "d": 1.0})
    assert tampered.passed and pc.trust_summary(model, tampered).level == "failed"


def test_the_limits_always_say_what_the_checks_cannot_establish(good):
    model, report = good
    assert any("reading of the problem" in s for s in pc.trust_summary(model, _with_cross_check(report)).limits)


def test_a_failing_model_reports_the_failing_check():
    model = build_model(json.loads(KINEMATICS_BUGGY_JSON))
    summary = pc.trust_summary(model, deterministic_report(model))
    assert summary.level == "failed" and summary.limits


# ------------------------------------------------------------------------------------------ step badges

def test_step_badges_follow_what_the_step_is(good):
    model, report = good
    full = _with_cross_check(report)
    audits = pc.audit_equations(model, full.sympy_numeric_answers)
    start = pc.step_badge("Start from: final velocity equation", "a", full, audits)
    assert start and start.status == "verified" and start.icon == "✅"
    result = pc.step_badge("Numeric result", "a", full, audits)
    assert result and result.status == "verified" and "independent re-solve agrees" in result.label
    assert pc.step_badge("Isolate and simplify to solve for a", "a", full, audits) is None      # no invented badge
    assert pc.step_badge("Start from: an equation that is not in the model", "a", full, audits) is None


def test_the_numeric_result_badge_is_caution_without_an_independent_resolve_and_failed_when_it_disagrees(good):
    model, report = good
    audits = pc.audit_equations(model, report.sympy_numeric_answers)
    assert pc.step_badge("Numeric result", "a", report, audits).status == "caution"
    bad = _with_cross_check(report, agree=False)
    assert pc.step_badge("Numeric result", "a", bad, audits).status == "failed"


# ------------------------------------------------------------------------------------------ replay

def _record(report, raw, text="A car accelerates."):
    return {"problem_text": text, "raw_json": raw,
            "verification": {"passed": report.passed, "checks": [vars(c) for c in report.checks],
                             "sympy_numeric_answers": report.sympy_numeric_answers, "llm_independent_answers": {}},
            "steps_by_target": {}, "scenarios": []}


def _bundle(*records):
    return json.dumps({"format_version": 1, "exported_at": "x", "history": list(records), "chains": [], "workspace": []})


def test_an_honest_bundle_replays_identically_and_says_the_resolve_was_not_replayed(good):
    model, report = good
    stored = _with_cross_check(report)
    result = pc.replay_bundle(_bundle(_record(stored, model.raw_json)))
    assert result.ok and result.items[0].status == "reproduced"
    assert any("not replayed" in d for d in result.items[0].details)


def test_a_tampered_answer_is_reported_as_differing(good):
    model, report = good
    forged = VerificationReport(checks=list(report.checks), sympy_numeric_answers={"a": 99.0, "d": 1.0})
    result = pc.replay_bundle(_bundle(_record(forged, model.raw_json)))
    assert not result.ok and result.items[0].status == "differs"
    assert any(d.startswith("a: was 99") for d in result.items[0].details)


def test_a_check_whose_outcome_changed_is_reported(good):
    model, report = good
    lying = VerificationReport(checks=[CheckResult("Equation parsing", False, "x")] + list(report.checks[1:]),
                               sympy_numeric_answers=report.sympy_numeric_answers)
    result = pc.replay_record(_record(lying, model.raw_json))
    assert result.status == "differs"


def test_one_unreadable_record_does_not_stop_the_others(good):
    model, report = good
    result = pc.replay_bundle(_bundle({"problem_text": "broken", "raw_json": {"nonsense": 1}, "verification": {}},
                                      _record(report, model.raw_json)))
    assert result.counts() == {"reproduced": 1, "differs": 0, "not replayable": 1} and not result.ok


@pytest.mark.parametrize("text", ["not json", json.dumps({"format_version": 99}), json.dumps({"format_version": 1, "history": []})])
def test_bad_bundles_are_errors_not_crashes(text):
    result = pc.replay_bundle(text)
    assert result.errors and not result.ok


# ------------------------------------------------------------------------------------------ the CLI

def test_cli_replay_exits_zero_for_an_honest_bundle_and_one_for_a_forged_one(good, tmp_path, capsys):
    import cli
    model, report = good
    ok, bad = tmp_path / "ok.json", tmp_path / "bad.json"
    ok.write_text(_bundle(_record(report, model.raw_json)))
    forged = VerificationReport(checks=list(report.checks), sympy_numeric_answers={"a": 99.0, "d": 1.0})
    bad.write_text(_bundle(_record(forged, model.raw_json)))
    assert cli.main(["replay", str(ok)]) == 0
    assert "1 reproduced" in capsys.readouterr().out
    assert cli.main(["replay", str(bad)]) == 1
    assert "[DIFFERS]" in capsys.readouterr().out
    assert cli.main(["replay", str(tmp_path / "missing.json")]) == 2
