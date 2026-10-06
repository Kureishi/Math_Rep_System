"""
The "Share / export" dialog: one place for everything that takes a solved problem OUT of the app.

It replaces two things that used to sit in the results page itself -- the Export section at the very
bottom and the "Send this result to a chain" expander near the top -- with a single button. Opening it
gives three tabs:

    Download   Markdown, and a PDF (generate once, then download)
    Copy       LaTeX source and a plain-text summary, each in a code block with a copy button
    Chain      send a target's result to a problem chain (the old expander, unchanged)

Nothing is built until the dialog is open, which also takes the full Markdown report out of every rerun
(the old Export section rebuilt it each time any widget moved, only to hand it to a download button).

A dialog is its own fragment: interacting with it reruns the dialog and leaves the page alone, which is
why generating the PDF here must not call st.rerun() -- that would reload the whole page and close the
dialog. The PDF is stored in session state and its download button shown in the same run.
"""
import streamlit as st

from modules import chains
from modules.equation_engine import ProblemModel, target_kind
from modules.exporter import build_markdown, build_pdf_bytes
from modules.share_text import build_latex, build_plain_text
from modules.verifier import VerificationReport


def render_share_button(model: ProblemModel, report: VerificationReport, steps_by_target) -> None:
    """The toolbar button that opens the dialog."""
    if st.button("📤 Share / export", key="open_share_dialog",
                 help="Download a report, copy LaTeX or a summary, or send a result to a chain"):
        share_dialog(model, report, steps_by_target)


@st.dialog("Share this solution", width="large")
def share_dialog(model: ProblemModel, report: VerificationReport, steps_by_target) -> None:
    tab_download, tab_copy, tab_chain = st.tabs(["⬇️ Download", "📋 Copy", "🔗 Send to a chain"])
    problem_text = st.session_state["problem_text"]
    with tab_download:
        _download_tab(model, report, steps_by_target, problem_text)
    with tab_copy:
        _copy_tab(model, report, steps_by_target, problem_text)
    with tab_chain:
        render_send_to_chain_form(model)


def _download_tab(model, report, steps_by_target, problem_text) -> None:
    scenarios = st.session_state["scenarios"] or []
    snapshots = list(st.session_state["plot_snapshots"].values())
    if snapshots:
        st.caption(f"{len(snapshots)} plot(s) you marked with 📸 will be included in the report.")
    else:
        st.caption("Tip: use 📸 “Include this plot in the report” under a plot to put it in the report.")

    st.download_button("📄 Download as Markdown",
                       data=build_markdown(problem_text, model, report, steps_by_target, scenarios,
                                           plot_snapshots=snapshots),
                       file_name="solved_problem.md", mime="text/markdown", key="share_md",
                       on_click="ignore")

    if st.button("🖨️ Generate PDF", key="share_pdf_generate"):
        with st.spinner("Rendering PDF (typesetting equations)..."):
            st.session_state["pdf_bytes"] = build_pdf_bytes(problem_text, model, report, steps_by_target,
                                                            scenarios, plot_snapshots=snapshots)
    if st.session_state["pdf_bytes"] is not None:
        st.download_button("⬇️ Download PDF", data=st.session_state["pdf_bytes"],
                           file_name="solved_problem.pdf", mime="application/pdf", key="share_pdf",
                           on_click="ignore")


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
