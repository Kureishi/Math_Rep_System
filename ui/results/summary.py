"""
Result sections that live in the main column of a solved problem: confidence banner, geometry schematic, send-to-chain, similar past problems, derived equations, matrix view, assumptions, variables, vector summary, follow-up Q&A, scenarios, export.
"""
import streamlit as st
import sympy as sp
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel, target_kind
from modules.verifier import VerificationReport, _known_substitutions
from modules.optimization_utils import solve_optimization
from modules.matrix_utils import linear_system_view
from modules.named_formulas import recognize_formula
from modules.geometry_solver import solve_triangle, render_triangle
from modules.followup import answer_followup
from modules.vector_utils import vector_summary
from modules.plotter import build_vector_plot
from modules.plot_snapshot import snapshot_vector_plot
from modules import history, chains
from modules.exporter import build_markdown, build_pdf_bytes
from ui.common import snapshot_button
from ui.theme import badge_row


def render_confidence_banner(report: VerificationReport):
    """The aggregated confidence banner: score, label, per-category pass counts and any critical failures."""
    # ---- confidence report: an aggregated, category-grouped view over
    # the raw check list, rather than making someone scan every check to
    # get a sense of "how much should I trust this". Rendered as a compact
    # badge row (see ui/theme.py) rather than a metric + a full-width
    # st.success/warning paragraph + per-category captions -- the same
    # information in roughly a quarter of the vertical space, which
    # matters because this banner sits above EVERY OTHER section of a
    # solved problem and was previously the single biggest chunk of
    # scrolling before reaching the actual equations.
    cr = report.confidence_report()
    conf_label, worst_ratio = report.confidence()

    if report.passed:
        overall_kind = "pass" if conf_label in ("essentially exact", "comfortable margin") else "warn"
        overall_text = f"✅ Verified -- {conf_label}"
    else:
        overall_kind = "fail"
        overall_text = "⚠️ Unresolved issues after retries"

    badges = [(f"Confidence {cr.score:.0%}", overall_kind), (overall_text, overall_kind)]
    for cat, summary in cr.categories.items():
        badges.append((f"{cat} {summary.passed}/{summary.total}",
                        "pass" if summary.all_passed else "fail"))
    badge_row(badges)

    if report.passed and overall_kind == "warn":
        st.caption("At least one check came close to its tolerance -- worth a second look before "
                    "trusting the result completely.")
    elif not report.passed:
        st.caption("Review the equations below carefully before trusting the result.")

    if cr.critical_failures:
        st.error("**Critical checks that failed:** " +
                  "; ".join(f"{c.label} -- {c.detail}" for c in cr.critical_failures))



def render_geometry_schematic(model: ProblemModel):
    """Independently solved + verified triangle schematic, when the extraction included a triangle."""
    # ---- geometry schematic: independently solved and verified by the
    # SAME geometry_solver.py machinery the standalone Geometry mode
    # uses directly (see equation_engine._parse_geometry's docstring for
    # why this is a parallel channel rather than folded into the normal
    # equation-based verification above). Only rendered when the LLM
    # extraction actually recognized this as a triangle-solving problem
    # -- most problems have model.geometry is None here, same as if
    # this whole block didn't exist.
    if model.geometry is not None and model.geometry.shape == "triangle":
        geom_result = solve_triangle(model.geometry.knowns)
        with st.expander("📐 Geometric schematic", expanded=True):
            if geom_result.error:
                st.warning(f"Recognized this as a triangle-solving problem, but couldn't solve it: "
                            f"{geom_result.error}")
            else:
                if len(geom_result.solutions) > 1:
                    st.info(f"This is the ambiguous SSA case -- {len(geom_result.solutions)} valid "
                            "triangles match these measurements.")
                for i, sol in enumerate(geom_result.solutions):
                    if len(geom_result.solutions) > 1:
                        st.markdown(f"**Solution {i + 1}**")
                    cols = st.columns(6)
                    for col, (label, val) in zip(cols, [("a", sol.a), ("b", sol.b), ("c", sol.c),
                                                           ("A", sol.A), ("B", sol.B), ("C", sol.C)]):
                        with col:
                            st.metric(label, f"{val:.4g}")
                    fig = render_triangle(sol, title=f"Solution {i + 1}" if len(geom_result.solutions) > 1
                                           else "Triangle")
                    st.plotly_chart(fig, width="stretch", key=f"word_problem_triangle_fig_{i}")



def render_send_to_chain(model: ProblemModel):
    """One-click shortcut to feed a just-solved target into a problem chain."""
    # ---- send to chain: a one-click shortcut so getting a just-solved
    # problem into a chain doesn't mean re-pasting its text into the
    # separate Problem chains mode. Only offered when there's an
    # algebraic result to actually expose downstream.
    send_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    if send_targets:
        with st.expander("🔗 Send this result to a chain"):
            existing_chains = chains.list_chains()
            chain_choice = st.selectbox(
                "Chain", ["+ New chain"] + [c["name"] for c in existing_chains],
                key="send_to_chain_choice",
            )
            send_target = st.selectbox("Expose which result downstream?", send_targets,
                                         key="send_to_chain_target")
            new_chain_name = ""
            if chain_choice == "+ New chain":
                new_chain_name = st.text_input(
                    "New chain name", key="send_to_chain_new_name",
                    placeholder=f"{model.problem_domain} chain",
                )
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
                        chains.add_step(target_chain_id, st.session_state.get("problem_text", ""),
                                          model, send_target)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        st.session_state["active_chain_id"] = target_chain_id
                        st.success(f"Added as a step exposing `{send_target}` -- see the sidebar, "
                                    "or switch to Problem chains to wire it up further.")
                        st.toast(f"Added to chain, exposing `{send_target}`", icon="🔗")



def render_similar_past_problems(model: ProblemModel):
    """Structurally similar previously solved problems."""
    # ---- find similar past problems: structural similarity (equation
    # shape, not problem-text wording) against everything already saved
    # to local history -- see similarity.py
    similar = history.find_similar(model, exclude_id=st.session_state.get("last_saved_history_id"))
    if similar:
        with st.expander(f"🔎 {len(similar)} similar past problem(s) found"):
            for s in similar:
                st.write(f"**{s['similarity']:.0%} match** -- {s['domain'] or 'unlabeled domain'} "
                          f"({s['timestamp'][:10]}): {s['problem_text'][:100]}"
                          f"{'...' if len(s['problem_text']) > 100 else ''}")



def render_derived_equations(model: ProblemModel):
    """Derived equations + derivations, and the optimization result when there is an objective. Returns opt_result (None if not an optimization problem)."""
    # ---- equations + derivations
    st.markdown("### Derived equations")
    KIND_BADGES = {"equation": "🟢 equation", "inequality": "🟡 inequality",
                    "ode": "🔵 differential equation", "recurrence": "🟣 recurrence relation"}
    var_by_symbol_for_names = {v.symbol: v for v in model.variables}
    for eq in model.equations:
        cols = st.columns([2, 3])
        with cols[0]:
            if eq.sympy_eq is not None:
                st.latex(sp.latex(eq.sympy_eq))
            else:
                st.error(f"Failed to parse: {eq.raw_expression}")
        with cols[1]:
            st.markdown(f"**{eq.name}**  `{KIND_BADGES.get(eq.kind, eq.kind)}`")
            st.write(eq.derivation)
            # ---- named-formula recognizer: purely a provenance/
            # pedagogy touch -- see named_formulas.py for why matching
            # needs the equation's own variable MEANINGS, not just the
            # problem's domain label, to disambiguate collisions like
            # F=m*a vs p=m*v (identical shape once canonicalized).
            if eq.sympy_eq is not None and eq.kind == "equation":
                eq_symbols = {s.name for s in eq.sympy_eq.free_symbols}
                meanings_text = " ".join(
                    var_by_symbol_for_names[s].meaning for s in eq_symbols
                    if s in var_by_symbol_for_names and var_by_symbol_for_names[s].meaning
                )
                context_text = f"{model.problem_domain} {meanings_text}"
                named = recognize_formula(eq, context_text)
                if len(named) == 1:
                    name, desc = named[0]
                    st.caption(f"📖 Recognized as **{name}** ({desc})")
                elif len(named) > 1:
                    names = ", ".join(n for n, _ in named)
                    st.caption(f"📖 Matches the shape of several named formulas: {names}")

    opt_result = solve_optimization(model) if model.objective is not None else None

    if model.objective is not None:
        st.markdown("### Objective")
        if model.objective.sympy_expr is not None:
            direction_word = "Minimize" if model.objective.direction == "minimize" else "Maximize"
            st.latex(f"\\text{{{direction_word}: }} {sp.latex(model.objective.sympy_expr)}")
            st.caption(f"Over: {', '.join(model.objective.optimize_over)}")

            if opt_result is not None:
                if opt_result.error:
                    st.error(opt_result.error)
                else:
                    method = "Lagrange multipliers (constrained)" if opt_result.used_lagrange else \
                        ("constraint substitution" if opt_result.eliminated_vars else "direct calculus")
                    st.caption(f"Method: {method}")
                    for pt, cls in zip(opt_result.critical_points, opt_result.classifications):
                        pretty_pt = ", ".join(f"{k}={float(v):.6g}" if v.is_number else f"{k}={v}"
                                                for k, v in pt.items())
                        st.write(f"**{pretty_pt}** -- {cls}")
                    for note in opt_result.feasibility_notes:
                        st.warning(note)
        else:
            st.error(f"Failed to parse objective: {model.objective.raw_expression} "
                      f"({model.objective.parse_error})")
    return opt_result



def render_matrix_view(model: ProblemModel):
    """A x = b view for genuine linear systems."""
    # ---- matrix representation, for genuine linear systems (>=2 equations,
    # >=2 shared unknowns) -- an additional structural VIEW onto the same
    # equations solver.py already solves via sp.solve(), not a separate answer
    matrix_result = linear_system_view(model, _known_substitutions(model))
    if matrix_result is not None:
        st.markdown("### Matrix representation")
        mcols = st.columns([1, 1, 1])
        with mcols[0]:
            st.caption("Coefficient matrix A")
            st.latex(sp.latex(matrix_result.A))
        with mcols[1]:
            st.caption("Unknowns x")
            st.latex(sp.latex(sp.Matrix([sp.Symbol(s) for s in matrix_result.symbols])))
        with mcols[2]:
            st.caption("Right-hand side b")
            st.latex(sp.latex(matrix_result.b))

        if matrix_result.is_square:
            st.write(f"**det(A) = {sp.latex(matrix_result.determinant)}**"
                      + (" -- singular" if matrix_result.determinant == 0 else ""))
            if matrix_result.eigenvalues:
                eig_text = ", ".join(
                    f"{sp.latex(val)}" + (f" (×{mult})" if mult > 1 else "")
                    for val, mult in matrix_result.eigenvalues.items())
                st.latex(r"\text{Eigenvalues: } " + eig_text)

        if matrix_result.consistent:
            st.success(matrix_result.classification)
        else:
            st.error(matrix_result.classification)



def render_assumptions(model: ProblemModel):
    """The assumptions the model stated."""
    if model.assumptions:
        st.markdown("**Assumptions made:**")
        for a in model.assumptions:
            st.write(f"- {a}")



def render_variables(model: ProblemModel):
    """Editable variable table. Returns edited_values ({symbol: value the person changed it to}), consumed by the plot and vector sections."""
    # ---- variables (editable, modification support)
    st.markdown("### Variables")
    edited_values = {}
    scalar_vars = [v for v in model.variables if not v.is_function]
    function_vars = [v for v in model.variables if v.is_function]
    if scalar_vars:
        var_cols = st.columns(min(4, max(1, len(scalar_vars))))
        for i, v in enumerate(scalar_vars):
            with var_cols[i % len(var_cols)]:
                default = v.known_value if v.known_value is not None else 0.0
                edited_values[v.symbol] = st.number_input(
                    f"{v.symbol} — {v.meaning} ({v.unit or 'unitless'})",
                    value=float(default), key=f"var_{v.symbol}",
                )
                if v.uncertainty:
                    st.caption(f"± {v.uncertainty:g} {v.unit or ''} (stated measurement uncertainty)")
    if function_vars:
        st.caption("Functions (solved as differential equations, not editable as plain numbers):")
        for v in function_vars:
            st.caption(f"`{v.symbol}` — {v.meaning} ({v.unit or 'unitless'})")
    return edited_values



def render_vector_summary(model: ProblemModel, edited_values):
    """Numeric components/magnitude/direction for each declared vector variable."""
    # ---- vector summary: for each declared vector variable, show its
    # numeric components (from the editable panel above), magnitude, and
    # unit vector once all its components are filled in
    vector_vars = [v for v in model.variables if v.is_vector and v.components]
    if vector_vars:
        st.markdown("### Vectors")
        edited_subs = {sp.Symbol(k): v for k, v in edited_values.items()}
        plottable_by_dim: dict[int, list[tuple[str, list[float]]]] = {}
        for v in vector_vars:
            summary = vector_summary(v.symbol, v.components, edited_subs)
            vcols = st.columns([2, 1, 1])
            with vcols[0]:
                comp_str = ", ".join(f"{c} = {val:g}" for c, val in
                                       (summary["components"].items() if summary else []))
                label = f"**{v.symbol}** — {v.meaning}"
                st.write(f"{label} ({comp_str})" if comp_str else
                          f"{label} (components: {', '.join(v.components)})")
            with vcols[1]:
                if summary and summary["magnitude"] is not None:
                    st.metric("Magnitude", f"{summary['magnitude']:.4g} {v.unit or ''}")
            with vcols[2]:
                if summary and summary["magnitude"] not in (None, 0):
                    unit_comps = [f"{c}/|{v.symbol}|" for c in v.components]
                    st.caption("Direction: " + ", ".join(unit_comps))
            if summary and len(v.components) in (2, 3):
                plottable_by_dim.setdefault(len(v.components), []).append(
                    (v.symbol, [summary["components"][c] for c in v.components]))

        for dim, vecs in plottable_by_dim.items():
            fig = build_vector_plot(vecs)
            st.plotly_chart(fig, width='stretch')
            snapshot_button(
                key=f"vectors_{dim}d",
                title=f"Vector diagram ({dim}D): " + ", ".join(name for name, _ in vecs),
                caption=", ".join(f"{name} = {comps}" for name, comps in vecs),
                render_fn=lambda vv=vecs: snapshot_vector_plot(vv),
            )



def render_followup(client: LMStudioClient, model: ProblemModel, report: VerificationReport):
    """Grounded follow-up Q&A."""
    # ---- grounded follow-up Q&A: "what if t doubles?" gets a REAL
    # recomputed answer (verified via SymPy, not LLM arithmetic); a
    # conceptual question gets an LLM answer grounded in the problem's
    # own equations/values -- see followup.py
    st.markdown("### Ask a follow-up question")
    followup_question = st.text_input(
        "e.g. \"what if t doubles?\" or \"why this formula?\"", key="followup_input",
    )
    if st.button("Ask", key="followup_button") and followup_question.strip():
        with st.spinner("Thinking..."):
            answer = answer_followup(client, model, report, followup_question)
        st.session_state["followup_history"].append((followup_question, answer))
    for q, a in reversed(st.session_state["followup_history"]):
        st.markdown(f"**Q: {q}**")
        if a.kind == "error":
            st.error(a.text)
        else:
            icon = "🧮" if a.kind == "what_if" else "💬"
            st.write(f"{icon} {a.text}")



def render_scenarios():
    """Alternative scenarios sharing the derived structure."""
    # ---- alternative scenarios
    if st.session_state["scenarios"]:
        st.markdown("### Where else this applies")
        for s in st.session_state["scenarios"]:
            if "error" in s:
                st.warning(s["error"])
                with st.expander("Raw model response"):
                    st.code(s.get("raw", ""))
            else:
                st.markdown(f"- **{s.get('scenario', '')}**  \n  _{s.get('mapping', '')}_")



def render_export(model: ProblemModel, report: VerificationReport, steps_by_target):
    """Markdown / PDF export controls."""
    # ---- export
    st.divider()
    st.markdown("### Export")
    scenarios_list = st.session_state["scenarios"] or []
    export_problem_text = st.session_state["problem_text"]
    plot_snapshots_list = list(st.session_state["plot_snapshots"].values())
    if plot_snapshots_list:
        st.caption(f"{len(plot_snapshots_list)} plot(s) will be included in the exported report.")

    c1, c2 = st.columns(2)
    with c1:
        md_content = build_markdown(export_problem_text, model, report, steps_by_target, scenarios_list,
                                      plot_snapshots=plot_snapshots_list)
        st.download_button("📄 Download as Markdown", data=md_content,
                            file_name="solved_problem.md", mime="text/markdown")
    with c2:
        if st.session_state["pdf_bytes"] is None:
            if st.button("🖨️ Generate PDF"):
                with st.spinner("Rendering PDF (typesetting equations)..."):
                    st.session_state["pdf_bytes"] = build_pdf_bytes(
                        export_problem_text, model, report, steps_by_target, scenarios_list,
                        plot_snapshots=plot_snapshots_list)
                st.rerun()
        else:
            st.download_button("⬇️ Download PDF", data=st.session_state["pdf_bytes"],
                                file_name="solved_problem.pdf", mime="application/pdf")
