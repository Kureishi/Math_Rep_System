"""
Buttons that set up OTHER widgets -- "add this input to the sweep", "load these values".

Streamlit lets code set a widget's value only before that widget is created in the current run, and a
button handler runs mid-script, usually after the target widget exists. So an action does not write the
widget directly: its `on_click` callback (which runs before the script starts) queues the values, and
`apply_pending_updates()` -- called once near the top of app.py, before the sidebar and the page create
any widget -- writes them. The same queue serves a click inside an isolated panel (ui/fragments.py),
where the rest of the page is not rerun by the click: `action_button` then asks for a full rerun so the
other panels pick the new values up.

This is the pattern app.py already used for the Quick Start gallery's mode switch (`_pending_mode`);
it is generalised here rather than duplicated, and that one is left as it was.
"""
from collections.abc import Mapping
from typing import Any

import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import ThreadState

PENDING_UPDATES_KEY = "_pending_updates"
PENDING_TOAST_KEY = "_pending_toast"


def queue_updates(updates: Mapping[str, Any], toast: str | None = None) -> None:
    """Remember `updates` ({widget key: value}) to be applied at the start of the next run. Safe to use
    as an `on_click` callback."""
    st.session_state.setdefault(PENDING_UPDATES_KEY, {}).update(updates)
    if toast:
        st.session_state[PENDING_TOAST_KEY] = toast


def apply_pending_updates() -> int:
    """Write any queued values into session_state and show the queued toast. Call before any widget is
    created in the run. Returns how many values were applied."""
    pending = st.session_state.pop(PENDING_UPDATES_KEY, None) or {}
    for key, value in pending.items():
        st.session_state[key] = value
    toast = st.session_state.pop(PENDING_TOAST_KEY, None)
    if toast:
        st.toast(toast, icon="✅")
    return len(pending)


def in_fragment() -> bool:
    return ThreadState.get().fragment_id is not None


def action_button(label: str, *, key: str, updates: Mapping[str, Any], toast: str | None = None,
                  help: str | None = None, type: str = "secondary", width: str = "content") -> bool:
    """A button that queues `updates` when clicked. Returns True on the run the click triggered."""
    clicked = st.button(label, key=key, help=help, type=type,  # type: ignore[arg-type]
                        on_click=queue_updates, kwargs={"updates": dict(updates), "toast": toast},
                        width=width)  # type: ignore[arg-type]
    if clicked and in_fragment():
        st.rerun()              # a click in an isolated panel reruns only that panel; the targets are elsewhere
    return clicked
