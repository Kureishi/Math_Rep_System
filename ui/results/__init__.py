"""
The results view for a solved problem: `render_results()` calls one function per section, in
the order the old inline script rendered them.
"""
import streamlit as st

from modules.equation_engine import ProblemModel
from modules.llm_client import LMStudioClient
from modules.verifier import VerificationReport
from modules.workspace import Workspace
from ui.results.summary import (
    render_confidence_banner, render_geometry_schematic, render_motion_diagram,
    render_similar_past_problems, render_derived_equations, render_matrix_view, render_assumptions,
    render_variables, render_vector_summary, render_followup, render_scenarios,
)
from ui.safe import run_panel
from ui.view_state import autosave
from ui.results.verify_tab import render_verify_tab
from ui.results.explore_tab import render_dependency_and_sweeps, render_interactive_plot
from ui.results.practice_tab import render_practice_tab
from ui.results.solutions import render_ode_solution, render_recurrence_solution
from ui.results.time_views import render_ode_time_views, render_map_views
from ui.results.steps import render_step_by_step
from ui.results.proof_card import render_proof_card
from ui.next_steps import render_solution_next_steps


def render_results(client: LMStudioClient, ws: Workspace, model: ProblemModel, report: VerificationReport, problem_text: str) -> None:
    """Everything shown for a solved (or history-restored) problem, in the exact order the old
    inline script rendered it. The sections are separate functions now; the only values that flow
    between them are `opt_result`, `edited_values` and `steps_by_target` (computed below).

    Every panel runs under `run_panel` (ui/safe.py): if one fails -- a missing package, a bad value, a bug -- it shows what it could not do and the
    rest of the results still render. The derivation and the verification are never held hostage by the report exporter or a plot."""
    st.divider()
    st.session_state["_graph_focus"] = []     # set again by the Explore tab if a graph node is selected
    run_panel("Confidence summary", render_confidence_banner, report)
    run_panel("Proof card", render_proof_card, model, report)
    run_panel("Geometry schematic", render_geometry_schematic, model)
    run_panel("Motion diagram", render_motion_diagram, model, report)
    run_panel("Share / export", _share, model, report)

    st.caption("Secondary panels are grouped into tabs below -- verification checks, "
                "exploratory plots, and practice tools -- so the main solution flow "
                "below isn't buried under them.")
    tab_verify, tab_explore, tab_practice = st.tabs(["🔎 Verify", "📊 Explore", "🎯 Practice"])

    run_panel("Verification tab", render_verify_tab, client, model, report, problem_text, tab_verify)

    st.subheader(f"Domain: {model.problem_domain}")

    run_panel("Similar past problems", render_similar_past_problems, model)
    opt_result = run_panel("Derived equations", render_derived_equations, model)
    run_panel("Matrix view", render_matrix_view, model)
    run_panel("Dependency graph and sweeps", render_dependency_and_sweeps, model, tab_explore)
    run_panel("Assumptions", render_assumptions, model)
    edited_values = run_panel("Variables", render_variables, model, default={})
    run_panel("Vector summary", render_vector_summary, model, edited_values)

    steps_by_target = st.session_state["steps"] or {}
    if steps_by_target:
        run_panel("Step-by-step solution", render_step_by_step, client, ws, model, report, problem_text, opt_result, steps_by_target)
    elif st.session_state.get("model") is not None:
        st.info("No step-by-step solution is available for this problem (it was not computed, or could not be). The derived equations and their verification are above.")

    # "What next?" comes after the worked solution (that is when someone is ready to choose a next step) and before the follow-up box
    run_panel("Suggested next steps", render_solution_next_steps, model, report, problem_text)
    run_panel("Follow-up questions", render_followup, client, model, report)
    run_panel("Alternative scenarios", render_scenarios)
    run_panel("Practice tab", render_practice_tab, client, model, report, tab_practice)
    run_panel("ODE solution", render_ode_solution, ws, model)
    run_panel("ODE time views", render_ode_time_views, model)
    run_panel("Recurrence solution", render_recurrence_solution, ws, model)
    run_panel("Map views", render_map_views, model)
    run_panel("Interactive plot", render_interactive_plot, model, edited_values, tab_explore)
    run_panel("Autosave of view settings", autosave)   # remember how this problem is being explored (a no-op unless something changed)


def _share(model: ProblemModel, report: VerificationReport) -> None:
    """The share button, imported on use: the report exporters pull in several optional packages, and a missing one must only cost the export."""
    from ui.share import render_share_button
    render_share_button(model, report, st.session_state["steps"] or {})
