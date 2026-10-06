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
import streamlit as st

from modules import history
from modules import view_state as vs
from modules.app_logging import logger

CURRENT_ID_KEY = "current_history_id"
_SAVED_KEY = "_view_saved"        # (history id, digest) of what is already in the database


def _clear_exploration_keys() -> None:
    for key in vs.keys_to_clear(st.session_state):
        del st.session_state[key]


def start_fresh(history_id: int | None) -> None:
    """A new problem is open: forget the previous one's settings and note which record it is."""
    _clear_exploration_keys()
    st.session_state[CURRENT_ID_KEY] = history_id
    st.session_state[_SAVED_KEY] = (history_id, vs.digest({}))


def restore(history_id: int) -> int:
    """Open a saved problem's view. Returns how many settings were restored (0 for a problem saved
    before this feature, or never explored)."""
    _clear_exploration_keys()
    try:
        values = vs.sanitize_saved(history.load_view_state(history_id))
    except Exception:  # noqa: BLE001 -- a damaged record must never stop the problem from opening
        logger.exception("Could not read the saved view for history record %s", history_id)
        values = {}
    for key, value in values.items():
        st.session_state[key] = value
    st.session_state[CURRENT_ID_KEY] = history_id
    st.session_state[_SAVED_KEY] = (history_id, vs.digest(values))
    return len(values)


def autosave() -> bool:
    """Write the current exploration state to the open problem's record if it changed since the last
    write. Returns True if something was written. Never raises: losing a saved view is not worth
    interrupting the page for."""
    history_id = st.session_state.get(CURRENT_ID_KEY)
    if history_id is None:
        return False
    state = vs.capture(st.session_state)
    marker = (history_id, vs.digest(state))
    if st.session_state.get(_SAVED_KEY) == marker:
        return False
    try:
        if not history.save_view_state(history_id, vs.wrap(state)):
            return False                       # the record has been pruned or deleted
    except Exception:  # noqa: BLE001
        logger.exception("Could not save the view for history record %s", history_id)
        return False
    st.session_state[_SAVED_KEY] = marker
    return True
