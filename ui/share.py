"""
The "Share / export" dialog: everything that takes a solved problem OUT of the app, in one place.

    Export   choose what goes in (a preset, or tick sections), then one click on a format:
             PDF, interactive HTML, Word, PowerPoint, LaTeX or Markdown
    Copy     LaTeX source and a plain-text summary, each in a code block with a copy button
    Chain    send a target's result to a problem chain

ONE CLICK: each format button is a deferred download (ui/exports.py) -- the file is built when the button is
clicked, not before, so there is no "generate, then download" step, opening the dialog costs nothing, and
the same file is not rebuilt if it is clicked again.

WHAT GOES IN: the sections of modules/report_content.py. Only sections that have content for THIS problem
are offered (no "Matrix representation" ticked for a problem with no matrix). The choice is remembered for the
session, and picking a preset replaces it; editing a tick afterwards switches the preset to "Custom".

A dialog is its own fragment: interacting with it reruns the dialog and leaves the page alone.
"""
import streamlit as st

from modules import chains
from modules.equation_engine import ProblemModel, target_kind
from modules.report_content import (
    DEFAULT_PRESET, PRESETS, SECTION_DESCRIPTIONS, SECTION_IDS, SECTION_TITLES,
)
from modules.report_export import FORMATS, ExportContext, available_sections
from modules.share_text import build_latex, build_plain_text
from modules.verifier import VerificationReport
from ui.cache import cached
from ui.exports import collect_context, download_button

_SELECTION_KEY = "share_selection"      # the chosen section ids, kept apart from the checkbox widgets
_CUSTOM = "Custom"


def render_share_button(model: ProblemModel, report: VerificationReport, steps_by_target) -> None:
    """The toolbar button that opens the dialog."""
    if st.button("📤 Share / export", key="open_share_dialog",
                 help="Download a report, copy LaTeX or a summary, or send a result to a chain"):
        share_dialog(model, report, steps_by_target)


@st.dialog("Share this solution", width="large")
def share_dialog(model: ProblemModel, report: VerificationReport, steps_by_target) -> None:
    ctx = collect_context()
    tab_export, tab_copy, tab_chain = st.tabs(["⬇️ Export", "📋 Copy", "🔗 Send to a chain"])
    with tab_export:
        _export_tab(ctx)
    with tab_copy:
        _copy_tab(model, report, steps_by_target, ctx.problem_text)
    with tab_chain:
        render_send_to_chain_form(model)


def _apply_preset() -> None:
    chosen = PRESETS.get(st.session_state["share_preset"])
    if chosen is None:
        return
    st.session_state[_SELECTION_KEY] = list(chosen)
    for sid in SECTION_IDS:
        st.session_state[f"share_sec_{sid}"] = sid in chosen


def _toggled(section_id: str) -> None:
    selection = set(st.session_state.get(_SELECTION_KEY, []))
    (selection.add if st.session_state[f"share_sec_{section_id}"] else selection.discard)(section_id)
    st.session_state[_SELECTION_KEY] = [s for s in SECTION_IDS if s in selection]
    st.session_state["share_preset"] = _CUSTOM


def _export_tab(ctx: ExportContext) -> None:
    available = cached("share_available", (ctx,), lambda: available_sections(ctx))
    selection = st.session_state.setdefault(_SELECTION_KEY, list(PRESETS[DEFAULT_PRESET]))
    st.session_state.setdefault("share_preset", DEFAULT_PRESET)

    st.selectbox("What to include", list(PRESETS) + [_CUSTOM], key="share_preset", on_change=_apply_preset,
                 help="A preset ticks a set of sections; change any tick and it becomes Custom.")
    cols = st.columns(3)
    for i, sid in enumerate(s for s in SECTION_IDS if s in available):
        key = f"share_sec_{sid}"
        with cols[i % 3]:
            if key in st.session_state:                      # state wins; passing a value too would be ignored
                st.checkbox(SECTION_TITLES[sid], key=key, on_change=_toggled, args=(sid,),
                            help=SECTION_DESCRIPTIONS[sid])
            else:
                st.checkbox(SECTION_TITLES[sid], value=sid in selection, key=key, on_change=_toggled,
                            args=(sid,), help=SECTION_DESCRIPTIONS[sid])
    skipped = [SECTION_TITLES[s] for s in SECTION_IDS if s not in available]
    if skipped:
        st.caption("Nothing to include for: " + ", ".join(skipped) + ".")
    if ctx.snapshots:
        st.caption(f"{len(ctx.snapshots)} plot(s) you marked with 📸 are in the Plots section.")
    else:
        st.caption("Tip: 📸 “Include this plot in the report” under a plot adds it to the Plots section.")

    chosen = [s for s in SECTION_IDS if s in available and st.session_state.get(f"share_sec_{s}", s in selection)]
    if not chosen:
        st.warning("Tick at least one section to export.")
        return
    st.markdown("**Download** -- built when you click")
    grid = st.columns(3)
    for i, spec in enumerate(FORMATS):
        with grid[i % 3]:
            download_button(spec.key, f"{spec.icon} {spec.label}", ctx, chosen, key=f"share_dl_{spec.key}")
            st.caption(spec.note)


def _copy_tab(model, report, steps_by_target, problem_text) -> None:
    include_steps = st.checkbox("Include the worked steps", key="share_latex_steps")
    st.caption("LaTeX (needs amsmath) -- use the copy icon at the top right of the block")
    st.code(build_latex(model, report, steps_by_target, include_steps=include_steps), language="latex")
    st.caption("Plain-text summary")
    st.code(build_plain_text(model, report, problem_text), language=None)


def render_send_to_chain_form(model: ProblemModel) -> None:
    """Feed a just-solved target into a problem chain, so it doesn't mean re-pasting the problem text into
    the separate Problem chains mode. Only offered when there's an algebraic result to expose downstream."""
    send_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    if not send_targets:
        st.caption("Only problems with an algebraic result can be sent to a chain.")
        return
    existing_chains = chains.list_chains()
    chain_choice = st.selectbox("Chain", ["+ New chain"] + [c["name"] for c in existing_chains],
                                key="send_to_chain_choice")
    send_target = st.selectbox("Expose which result downstream?", send_targets, key="send_to_chain_target")
    new_chain_name = ""
    if chain_choice == "+ New chain":
        new_chain_name = st.text_input("New chain name", key="send_to_chain_new_name",
                                       placeholder=f"{model.problem_domain} chain")
    if st.button("Send to chain", key="send_to_chain_button"):
        if chain_choice == "+ New chain":
            if not new_chain_name.strip():
                st.error("Give the new chain a name first.")
                target_chain_id = None
            else:
                target_chain_id = chains.create_chain(new_chain_name.strip())
        else:
            target_chain_id = next(c["id"] for c in existing_chains if c["name"] == chain_choice)
        if target_chain_id is not None:
            try:
                chains.add_step(target_chain_id, st.session_state.get("problem_text", ""), model, send_target)
            except ValueError as e:
                st.error(str(e))
            else:
                st.session_state["active_chain_id"] = target_chain_id
                st.success(f"Added as a step exposing `{send_target}` -- see the sidebar, or switch to "
                           "Problem chains to wire it up further.")
                st.toast(f"Added to chain, exposing `{send_target}`", icon="🔗")
