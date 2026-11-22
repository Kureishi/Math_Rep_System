"""
The sidebar's mode navigation: the modes grouped under Solve / Science / Simulation labs / Tools, each group a collapsible section.

Why not one `st.radio`: a flat radio list of every mode (nearly thirty) pushed History, the workspace and the project tools far down
the sidebar, and Streamlit radios cannot be grouped or collapsed. Each mode is now a button that sets `st.session_state["app_mode"]`
(the same key the radio used, so the command palette, "Ask anything", Quick start, the "What next?" bar, the URL parameter and every
test that writes the key work unchanged). The current mode is the highlighted button; the group that holds it opens by itself.

Because `app_mode` is no longer a widget's key, it can be written at any point of a run, not only before the radio is built -- the
pending-switch indirection in app.py is still applied (it is harmless) but is no longer required for this reason.

The groups come from each mode's own declaration (`ModeSpec.group`, modules/modes/*.py); a plugin that names a new group gets a section of its own.
"""
import streamlit as st

from modules.mode_registry import REGISTRY
from ui.common import restore_from_query_param, sync_query_param

MODE_KEY = "app_mode"
_NAV_PREFIX = "nav_"


def _select(label: str) -> None:
    """Button callback: callbacks run before the script reruns, so this is the one safe place to change the mode from a click."""
    st.session_state[MODE_KEY] = label


def current_mode() -> str:
    """The selected mode: the session's, else the one in the page's URL, else the first. A value that is not a mode (a stale or mistyped link) is replaced by the first."""
    labels = REGISTRY.labels
    restore_from_query_param(MODE_KEY)
    if st.session_state.get(MODE_KEY) not in labels:
        st.session_state[MODE_KEY] = labels[0]
    return st.session_state[MODE_KEY]


def render_mode_nav() -> str:
    """Draw the grouped navigation and return the selected mode."""
    mode = current_mode()
    active_group = REGISTRY.group_of(mode)
    st.markdown("**Mode**")
    st.caption(f"Now in: {mode}")
    for group, labels in REGISTRY.grouped():
        with st.expander(f"{group} · {len(labels)}", expanded=(group == active_group)):
            for label in labels:
                st.button(label, key=f"{_NAV_PREFIX}{label}", type="primary" if label == mode else "tertiary",
                          width="stretch", on_click=_select, args=(label,))
    sync_query_param(MODE_KEY)
    return mode
