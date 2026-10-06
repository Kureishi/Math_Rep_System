"""
Math Representation System -- Streamlit entrypoint.

Run with:
    streamlit run app.py

Requires LM Studio running locally with its server started
(Developer tab -> Start Server, default port 1234).

This file is deliberately thin: the pages live in ui/ (see ui/__init__.py for the map) and the
math/LLM logic lives in modules/, which has no Streamlit dependency.
"""
from typing import Any
import streamlit as st

# Everything imported ABOVE the set_page_config() call must be cheap: nothing can be drawn until the
# script reaches its first st.* call, and the heavy imports (the OpenAI SDK, pandas, scipy, the whole
# verification/solver stack -- several seconds cold) used to sit here, so a fresh page load showed a
# blank screen until all of them had finished. Now only the theme helpers are imported first (ui/
# __init__.py is lazy, so importing ui.theme does not pull the page modules in), the title and a
# loading notice are drawn, and THEN the heavy imports run behind them.
from ui.theme import inject_base_styles, render_hero

st.set_page_config(page_title="Math Representation System", page_icon="🧮", layout="wide")
inject_base_styles()

render_hero("🧮 Math Representation System",
            "Text or image → derived equations → self-verified solution → alternative applications.")

# Shown only on a session's first run (a page refresh starts a new session), and cleared just before the
# real page content is drawn below. Later reruns (every widget click) skip it -- flashing a notice on
# each interaction would be noise, and by then the imports are cached anyway.
_boot_notice = None
if not st.session_state.get("_booted"):
    _boot_notice = st.empty()
    _boot_notice.info("⏳ Loading the app -- the first load can take several seconds while the math "
                       "libraries start up. This page will fill in automatically.")

from modules.llm_client import LMStudioClient  # noqa: E402  (deliberately after the first paint, see above)
from modules.workspace import Workspace  # noqa: E402
from ui import PAGES  # noqa: E402
from ui.actions import apply_pending_updates  # noqa: E402
from ui.sidebar import render_sidebar  # noqa: E402
from ui.word_problem import render_word_problem_page  # noqa: E402

# ---------------------------------------------------------------- session
client = LMStudioClient()
ws = Workspace(st.session_state)
_SESSION_DEFAULTS: list[tuple[str, Any]] = [
    ("problem_text", ""), ("model", None), ("report", None),
    ("steps", None), ("scenarios", None), ("extracted_from_image", ""),
    ("pdf_bytes", None), ("plot_snapshots", {}), ("worksheet_problems", []),
    ("batch_results", None), ("last_saved_history_id", None), ("current_history_id", None),
    ("paranoid_result", None), ("followup_history", []),
    ("self_consistency_result", None), ("error_pattern_messages", []),
]
for key, default in _SESSION_DEFAULTS:
    st.session_state.setdefault(key, default)

# ---- apply a pending Quick Start example (mode switch + prefilled
# inputs), queued by render_quick_start_tab()'s "Try this" buttons.
# Must happen HERE, before the sidebar's app_mode radio widget (or any
# of the target mode's own widgets) are instantiated in this run --
# Streamlit forbids writing to a widget's session_state key once that
# widget already exists in the current script run, which is exactly
# what set st.session_state["app_mode"] = ... from inside the Quick
# Start tab's own button handler (itself running AFTER the radio
# widget, later in the same script) directly.
if "_pending_mode" in st.session_state:
    st.session_state["app_mode"] = st.session_state.pop("_pending_mode")
    for _k, _v in st.session_state.pop("_pending_prefill", {}).items():
        st.session_state[_k] = _v

# ---- apply values queued by action buttons ("add this input to the sweep", "load these values").
# Same constraint, same place as the block above: widgets may only be set before they are created in
# this run. See ui/actions.py.
apply_pending_updates()

sidebar = render_sidebar(client, ws)

if _boot_notice is not None:
    _boot_notice.empty()
st.session_state["_booted"] = True

# ---------------------------------------------------------------- mode dispatch
# `sidebar.mode` comes from the sidebar's navigation radio. Every mode except the default word-problem
# solver is a standalone page in ui.PAGES, gated with an early st.stop() so the word-problem page
# below never renders underneath it.
page = PAGES.get(sidebar.mode)
if page is not None:
    page(client)
    st.stop()

render_word_problem_page(client, ws, sidebar.ok)
