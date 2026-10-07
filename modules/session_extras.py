"""
The parts of a report that live in the browser session rather than in the solved model: follow-up Q&A,
tutor-mode guesses, and Monte Carlo runs -- gathered, serialised for the history record, and turned into the
`ReportExtras` the report builder takes. Streamlit-free: it reads any mapping standing in for
`st.session_state`, so it is testable without a running app.

Follow-ups and tutor guesses used to vanish when the session ended or another problem was opened (the
follow-up list was not even cleared when a NEW problem was solved, so one problem's questions appeared under
the next). They are now saved with the problem's history record (modules/history.py's extras) and restored
when it is reopened, so an export made next week still has them.

Everything read back from storage is untrusted (it may come from a project bundle), so it is validated and
bounded rather than assumed well-formed.
"""
from collections.abc import Mapping
from typing import Any

from modules.followup import FollowupAnswer
from modules.report_content import FollowupItem, ReportExtras, TutorItem, mc_run_dict

MAX_FOLLOWUPS = 50
MAX_TEXT = 4000
MAX_MC_SAMPLES_IN_REPORT = 4000
_KINDS = ("what_if", "conceptual", "error")


def _clip(value: Any, limit: int = MAX_TEXT) -> str:
    return str(value)[:limit] if value is not None else ""


# ------------------------------------------------------------------ follow-ups

def followups_to_json(history: list[tuple[str, FollowupAnswer]]) -> list[dict[str, Any]]:
    out = []
    for question, answer in history[-MAX_FOLLOWUPS:]:
        out.append({"q": _clip(question), "kind": answer.kind, "text": _clip(answer.text),
                    "value": answer.computed_value, "target": answer.target, "symbol": answer.symbol})
    return out


def followups_from_json(data: Any) -> list[tuple[str, FollowupAnswer]]:
    if not isinstance(data, list):
        return []
    out: list[tuple[str, FollowupAnswer]] = []
    for item in data[:MAX_FOLLOWUPS]:
        if not isinstance(item, Mapping) or not isinstance(item.get("q"), str):
            continue
        kind: str = item["kind"] if item.get("kind") in _KINDS else "conceptual"
        value = item.get("value")
        out.append((item["q"][:MAX_TEXT], FollowupAnswer(
            kind=kind, text=_clip(item.get("text")),
            computed_value=float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None,
            target=item.get("target") if isinstance(item.get("target"), str) else None,
            symbol=item.get("symbol") if isinstance(item.get("symbol"), str) else None)))
    return out


def followup_items(history: list[tuple[str, FollowupAnswer]]) -> list[FollowupItem]:
    return [FollowupItem(q, a.kind, a.text) for q, a in history]


# ------------------------------------------------------------------ tutor mode

def tutor_from_session(state: Any, steps_by_target: Mapping[str, list]) -> list[TutorItem]:
    """The tutor-mode activity currently in the session (`state` is any mapping-like, including Streamlit's
    session-state proxy, which is not a Mapping): for each target, the guess and how it was marked,
    and how many steps had been revealed. Only targets with a marked guess or a revealed step appear."""
    items: list[TutorItem] = []
    for target, steps in steps_by_target.items():
        feedback = state.get(f"tutor_feedback_{target}")
        revealed = state.get(f"tutor_reveal_{target}") or 0
        guess = str(state.get(f"tutor_guess_{target}") or "").strip()
        if feedback is None and not revealed:
            continue
        if feedback is not None and getattr(feedback, "error", None):
            correct, text = None, str(feedback.error)
        elif feedback is not None:
            correct, text = bool(feedback.correct), str(feedback.detail)
        else:
            correct, text = None, ""
        items.append(TutorItem(target, _clip(guess, 200) if feedback is not None else "", correct,
                               _clip(text, 500), int(min(revealed, len(steps))), len(steps)))
    return items


def tutor_to_json(items: list[TutorItem]) -> list[dict[str, Any]]:
    return [{"target": t.target, "guess": t.guess, "correct": t.correct, "feedback": t.feedback,
             "revealed": t.steps_revealed, "total": t.steps_total} for t in items]


def tutor_from_json(data: Any) -> list[TutorItem]:
    if not isinstance(data, list):
        return []
    out = []
    for item in data[:100]:
        if not isinstance(item, Mapping) or not isinstance(item.get("target"), str):
            continue
        correct = item.get("correct")
        out.append(TutorItem(
            item["target"][:100], _clip(item.get("guess"), 200),
            correct if isinstance(correct, bool) else None, _clip(item.get("feedback"), 500),
            int(item["revealed"]) if isinstance(item.get("revealed"), int) else 0,
            int(item["total"]) if isinstance(item.get("total"), int) else 0))
    return out


def merge_tutor(saved: list[TutorItem], live: list[TutorItem]) -> list[TutorItem]:
    """Saved transcripts, with any target that has live activity in this session replaced by the live one."""
    live_targets = {t.target for t in live}
    return [t for t in saved if t.target not in live_targets] + live


# ------------------------------------------------------------------ Monte Carlo and the whole bundle

def monte_carlo_runs(state: Any, model: Any, with_samples: bool = False) -> list[dict[str, Any]]:
    """The per-target Monte Carlo results currently in the session, as report dicts."""
    runs = []
    for target in model.solve_for:
        result = state.get(f"mc_result_{target}")
        if result is None or result.mean is None:
            continue
        unit = next((v.unit for v in model.variables if v.symbol == target), None)
        run = mc_run_dict(result, unit)
        if with_samples:
            run["samples"] = [float(x) for x in result.samples[:MAX_MC_SAMPLES_IN_REPORT]]
        runs.append(run)
    return runs


def build_extras(state: Any, model: Any, steps_by_target: Mapping[str, list],
                 live_figures: bool = False, include_provenance: bool = True) -> ReportExtras:
    """Everything a report needs from the session, for the problem currently open."""
    live_tutor = tutor_from_session(state, steps_by_target)
    saved_tutor = state.get("tutor_saved") or []
    return ReportExtras(
        followups=followup_items(state.get("followup_history") or []),
        tutor=merge_tutor(saved_tutor, live_tutor),
        monte_carlo=monte_carlo_runs(state, model, with_samples=live_figures),
        provenance=state.get("provenance"),
        include_provenance=include_provenance,
        live_figures=live_figures,
    )
