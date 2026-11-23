"""st.rerun() discards the toasts of the run it interrupts, so a function that shows a toast and then reruns shows nothing. Such code must park the message with ui.actions.flash()
(shown at the start of the next run). This scans ui/ for functions that call both st.toast and st.rerun."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Functions that contain both calls but never rerun AFTER the toast on the same path (checked by hand): the toast is in a branch of its own.
ALLOWED = {
    ("ui/common.py", "render_template_bar"),            # the toast is in the "Save" branch; the rerun is in the "Load" branch
    ("ui/word_problem.py", "render_word_problem_page"),  # the toast is shown AFTER the rerun that follows a solve (the status box is parked in session_state for it)
}


def _offenders() -> list[tuple[str, str]]:
    out = []
    for path in sorted([*(ROOT / "ui").rglob("*.py"), ROOT / "app.py"]):
        for fn in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            names = {n.func.attr for n in ast.walk(fn) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
                     and isinstance(n.func.value, ast.Name) and n.func.value.id == "st"}
            key = (path.relative_to(ROOT).as_posix(), fn.name)
            if {"toast", "rerun"} <= names and key not in ALLOWED:
                out.append(key)
    return out


def test_no_function_shows_a_toast_and_then_reruns():
    assert _offenders() == []


def test_a_flashed_toast_survives_until_the_next_run(monkeypatch):
    import streamlit as st
    from ui import actions
    shown = []
    monkeypatch.setattr(st, "session_state", {})
    monkeypatch.setattr(st, "toast", lambda text, icon=None: shown.append((text, icon)))
    actions.flash("saved", "📦")
    assert shown == []                                            # nothing is drawn in the run that parks it
    actions.apply_pending_updates()
    assert shown == [("saved", "📦")]
    actions.apply_pending_updates()
    assert shown == [("saved", "📦")]                             # and it is shown once
