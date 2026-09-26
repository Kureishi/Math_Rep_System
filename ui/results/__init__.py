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
    render_confidence_banner, render_geometry_schematic, render_motion_diagram, render_send_to_chain,
    render_similar_past_problems, render_derived_equations, render_matrix_view, render_assumptions,
    render_variables, render_vector_summary, render_followup, render_scenarios, render_export,
)
from ui.results.verify_tab import render_verify_tab
from ui.results.explore_tab import render_dependency_and_sweeps, render_interactive_plot
from ui.results.practice_tab import render_practice_tab
from ui.results.solutions import render_ode_solution, render_recurrence_solution
from ui.results.steps import render_step_by_step


def render_results(client: LMStudioClient, ws: Workspace, model: ProblemModel, report: VerificationReport, problem_text: str) -> None:
    """Everything shown for a solved (or history-restored) problem, in the exact order the old
    inline script rendered it. The sections are separate functions now; the only values that flow
    between them are `opt_result`, `edited_values` and `steps_by_target` (computed below)."""
    st.divider()
    render_confidence_banner(report)
    render_geometry_schematic(model)
    render_motion_diagram(model, report)
    render_send_to_chain(model)

    st.caption("Secondary panels are grouped into tabs below -- verification checks, "
                "exploratory plots, and practice tools -- so the main solution flow "
                "below isn't buried under them.")
    tab_verify, tab_explore, tab_practice = st.tabs(["🔎 Verify", "📊 Explore", "🎯 Practice"])

    render_verify_tab(client, model, report, problem_text, tab_verify)

    st.subheader(f"Domain: {model.problem_domain}")

    render_similar_past_problems(model)
    opt_result = render_derived_equations(model)
    render_matrix_view(model)
    render_dependency_and_sweeps(model, tab_explore)
    render_assumptions(model)
    edited_values = render_variables(model)
    render_vector_summary(model, edited_values)

    steps_by_target = st.session_state["steps"] or {}
    if steps_by_target:
        render_step_by_step(client, ws, model, report, problem_text, opt_result, steps_by_target)

    render_followup(client, model, report)
    render_scenarios()
    render_practice_tab(client, model, report, tab_practice)
    render_ode_solution(ws, model)
    render_recurrence_solution(ws, model)
    render_interactive_plot(model, edited_values, tab_explore)
    render_export(model, report, steps_by_target)
