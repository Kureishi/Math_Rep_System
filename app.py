"""
Math Representation System -- Streamlit entrypoint.

Run with:
    streamlit run app.py

Requires LM Studio running locally with its server started
(Developer tab -> Start Server, default port 1234).

This file is deliberately thin: the pages live in ui/ (see ui/__init__.py for the map) and the
math/LLM logic lives in modules/, which has no Streamlit dependency.
"""
import streamlit as st

from modules.llm_client import LMStudioClient
from modules.workspace import Workspace
from ui import PAGES
from ui.sidebar import render_sidebar
from ui.theme import inject_base_styles, render_hero
from ui.word_problem import render_word_problem_page

st.set_page_config(page_title="Math Representation System", page_icon="🧮", layout="wide")
inject_base_styles()

# ---------------------------------------------------------------- session
client = LMStudioClient()
ws = Workspace(st.session_state)
for key, default in [("problem_text", ""), ("model", None), ("report", None),
                      ("steps", None), ("scenarios", None), ("extracted_from_image", ""),
                      ("pdf_bytes", None), ("plot_snapshots", {}), ("worksheet_problems", []),
                      ("batch_results", None), ("last_saved_history_id", None),
                      ("paranoid_result", None), ("followup_history", []),
                      ("self_consistency_result", None), ("error_pattern_messages", [])]:
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

sidebar = render_sidebar(client, ws)

render_hero("🧮 Math Representation System",
            "Text or image → derived equations → self-verified solution → alternative applications.")

# ---------------------------------------------------------------- mode dispatch
# `sidebar.mode` comes from the sidebar's navigation radio. Every mode except the default word-problem
# solver is a standalone page in ui.PAGES, gated with an early st.stop() so the word-problem page
# below never renders underneath it.
page = PAGES.get(sidebar.mode)
if page is not None:
    page(client)
    st.stop()

render_word_problem_page(client, ws, sidebar.ok)
