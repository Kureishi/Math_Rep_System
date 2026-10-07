"""
Saving and restoring how a solved problem was being explored -- the Streamlit side of
modules/view_state.py (which decides WHICH settings count and is where the format lives).

  autosave()      called at the end of each results render and after each isolated panel; writes the
                  exploration state to the problem's history record, but only when it changed
  restore(id)     called when a past problem is loaded: clears the previous problem's settings, applies
                  the saved ones, and remembers which record is now open
  start_fresh(id) called when a new problem has just been solved and saved: clears the previous
                  problem's settings so they cannot leak into (or be saved under) this one

Why clear: the widget keys are built from symbol and target names ("mc_n_v", "sweep_symbols"), which
different problems share. Before this, a Monte Carlo sample count set for `v` in one problem silently
carried over to the next problem that also had a `v`. Now each problem starts from the defaults and
returns to its own saved settings.

Restoring happens BEFORE the page's widgets are created in the same run (the sidebar renders first), which
is the only time Streamlit allows a widget's value to be set from code.
"""
import json

import streamlit as st

from modules import history
from modules import session_extras as extras_module
from modules import view_state as vs
from modules.app_logging import logger

CURRENT_ID_KEY = "current_history_id"
_SAVED_KEY = "_view_saved"        # (history id, digest) of what is already in the database
_EXTRAS_SAVED_KEY = "_extras_saved"   # the same for the follow-ups and tutor transcript
_TUTOR_PREFIXES = ("tutor_feedback_", "tutor_reveal_", "tutor_guess_")


def _clear_exploration_keys() -> None:
    for key in vs.keys_to_clear(st.session_state):
        del st.session_state[key]


def _clear_problem_session_state() -> None:
    """The follow-up Q&A and tutor-mode activity belong to ONE problem. They were never cleared when another
    problem was solved or opened, so one problem's questions showed up under the next."""
    for key in [k for k in list(st.session_state.keys()) if isinstance(k, str) and k.startswith(_TUTOR_PREFIXES)]:
        del st.session_state[key]
    st.session_state["followup_history"] = []
    st.session_state["tutor_saved"] = []
    st.session_state["provenance"] = None


def start_fresh(history_id: int | None, provenance: dict | None = None) -> None:
    """A new problem is open: forget the previous one's settings, follow-ups and tutor activity, note which
    record it is, and keep (and save) how it was solved."""
    _clear_exploration_keys()
    _clear_problem_session_state()
    st.session_state[CURRENT_ID_KEY] = history_id
    st.session_state[_SAVED_KEY] = (history_id, vs.digest({}))
    st.session_state[_EXTRAS_SAVED_KEY] = (history_id, _extras_digest())
    if provenance is not None:
        st.session_state["provenance"] = provenance
        if history_id is not None:
            try:
                history.save_extra(history_id, "provenance", provenance)
            except Exception:  # noqa: BLE001
                logger.exception("Could not save provenance for history record %s", history_id)


def _extras_payload() -> dict:
    steps = st.session_state.get("steps") or {}
    live = extras_module.tutor_from_session(st.session_state, steps)
    tutor = extras_module.merge_tutor(st.session_state.get("tutor_saved") or [], live)
    return {"followups": extras_module.followups_to_json(st.session_state.get("followup_history") or []),
            "tutor": extras_module.tutor_to_json(tutor)}


def _extras_digest() -> str:
    return vs.digest(json.loads(json.dumps(_extras_payload(), default=str)))


def restore(history_id: int) -> int:
    """Open a saved problem's view. Returns how many settings were restored (0 for a problem saved
    before this feature, or never explored)."""
    _clear_exploration_keys()
    _clear_problem_session_state()
    try:
        values = vs.sanitize_saved(history.load_view_state(history_id))
    except Exception:  # noqa: BLE001 -- a damaged record must never stop the problem from opening
        logger.exception("Could not read the saved view for history record %s", history_id)
        values = {}
    for key, value in values.items():
        st.session_state[key] = value
    try:
        st.session_state["followup_history"] = extras_module.followups_from_json(history.load_extra(history_id, "followups"))
        st.session_state["tutor_saved"] = extras_module.tutor_from_json(history.load_extra(history_id, "tutor"))
        saved_provenance = history.load_extra(history_id, "provenance")
        st.session_state["provenance"] = saved_provenance if isinstance(saved_provenance, dict) else None
    except Exception:  # noqa: BLE001
        logger.exception("Could not read the saved follow-ups for history record %s", history_id)
    st.session_state[CURRENT_ID_KEY] = history_id
    st.session_state[_SAVED_KEY] = (history_id, vs.digest(values))
    st.session_state[_EXTRAS_SAVED_KEY] = (history_id, _extras_digest())
    return len(values)


def autosave() -> bool:
    """Write the open problem's exploration state, follow-ups and tutor transcript to its history record if
    they changed since the last write. Returns True if something was written. Never raises: losing a saved
    view is not worth interrupting the page for."""
    history_id = st.session_state.get(CURRENT_ID_KEY)
    if history_id is None:
        return False
    wrote = False
    try:
        state = vs.capture(st.session_state)
        marker = (history_id, vs.digest(state))
        if st.session_state.get(_SAVED_KEY) != marker:
            if not history.save_view_state(history_id, vs.wrap(state)):
                return False                       # the record has been pruned or deleted
            st.session_state[_SAVED_KEY] = marker
            wrote = True
        extras_marker = (history_id, _extras_digest())
        if st.session_state.get(_EXTRAS_SAVED_KEY) != extras_marker:
            payload = _extras_payload()
            history.save_extra(history_id, "followups", payload["followups"])
            history.save_extra(history_id, "tutor", payload["tutor"])
            st.session_state[_EXTRAS_SAVED_KEY] = extras_marker
            wrote = True
    except Exception:  # noqa: BLE001
        logger.exception("Could not save the view for history record %s", history_id)
        return False
    return wrote
