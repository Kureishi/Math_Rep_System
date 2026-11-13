"""The "What next?" bar (modules/next_steps.py): the sensible next moves from a result, as buttons."""
import streamlit as st

from modules import next_steps as ns
from modules import practice
from modules.app_logging import logger


def _perform(step: ns.NextStep) -> None:
    if step.mode:
        # app.py applies these before the sidebar's mode radio exists, which is the only time that radio's value may be set
        st.session_state["_pending_mode"] = step.mode
        st.session_state["_pending_prefill"] = dict(step.prefill)
        st.rerun()
    elif step.action == ns.ADD_TO_PRACTICE:
        p = step.payload
        try:
            _, created = practice.add_item(p["title"], p["text"], p["raw"], p.get("concepts"), "open")
        except practice.PracticeError as exc:
            st.warning(f"This problem cannot be added to practice: {exc}")
            return
        st.success("Added. It will come up in your daily practice, with new numbers each time." if created else "It is already in your daily practice.")


def render_next_steps(steps: list[ns.NextStep], key_prefix: str) -> None:
    """A row of buttons for the suggested next moves; a navigation sets up the destination mode with its inputs filled in."""
    if not steps:
        return
    st.markdown("**What next?**")
    for col, step in zip(st.columns(len(steps)), steps):
        if col.button(step.label, key=f"ns_{key_prefix}_{step.key}", help=step.reason, width="stretch"):
            _perform(step)


def render_solution_next_steps(model, report, problem_text: str) -> None:
    """Under a solved problem. Suggestions are an extra: failing to compute them must never hide the results."""
    try:
        render_next_steps(ns.for_solution(model, report, problem_text), "solve")
    except Exception:  # noqa: BLE001
        logger.exception("could not build the next steps for a solution")


def render_fit_handoff(xs, ys, family: str, degree: int, expr_str, param_names, param_values: dict[str, float]) -> None:
    """Under a curve fit: offer the posterior sampler for it when the fit has an explicit model to hand over."""
    try:
        handoff = ns.fit_to_model(family, degree, expr_str, param_names, param_values)
        if handoff:
            render_next_steps(ns.for_fit(xs, ys, *handoff), "fit")
    except Exception:  # noqa: BLE001
        logger.exception("could not build the posterior handoff for a fit")
