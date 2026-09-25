"""
ODE and recurrence closed-form solution sections (plot + evaluate-at-a-point).
"""
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from sympy.core.function import AppliedUndef
from modules.equation_engine import ProblemModel, symbols_and_functions_used
from modules.verifier import _known_substitutions
from modules.ode_utils import solve_ode, group_coupled_odes
from modules.recurrence_utils import solve_recurrence, _independent_variable, extract_step_map
from modules.proof import build_recurrence_induction_proof
from modules.plot_snapshot import snapshot_ode_plot, snapshot_recurrence_plot
from modules.plotter import build_phase_portrait, build_cobweb_plot
from ui.common import snapshot_button
from modules.workspace import Workspace


def render_ode_solution(ws: Workspace, model: ProblemModel):
    """ODE solution: plot + evaluate-at-a-point."""
    # ---- ODE solution: plot + evaluate-at-a-point
    ode_solutions = solve_ode(model)
    if ode_solutions:
        st.markdown("### Differential equation solution")
        for func_name, sol in ode_solutions.items():
            st.latex(sp.latex(sol))
            applied = sol.lhs  # e.g. y(t)
            indep_sym = applied.args[0]
            rhs_sub = sol.rhs.subs(_known_substitutions(model))
            remaining = sorted(rhs_sub.free_symbols - {indep_sym}, key=str)

            param_vals = {}
            if remaining:
                st.caption("Remaining parameters:")
                pcols = st.columns(min(4, len(remaining)))
                for i, s in enumerate(remaining):
                    with pcols[i % len(pcols)]:
                        param_vals[s] = st.slider(str(s), 0.01, 20.0, 1.0, key=f"odeparam_{func_name}_{s}")
            rhs_final = rhs_sub.subs(param_vals)

            try:
                f = sp.lambdify(indep_sym, rhs_final, "numpy")
                t_range = st.slider(f"{indep_sym} range", 0.0, 100.0, (0.0, 10.0),
                                     key=f"oderange_{func_name}")
                xs = np.linspace(t_range[0], t_range[1], 300)
                ys = np.real(np.array(f(xs), dtype=complex))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name=f"{func_name}({indep_sym})"))
                fig.update_layout(xaxis_title=str(indep_sym), yaxis_title=func_name)
                st.plotly_chart(fig, width='stretch')

                ode_caption = (
                    f"ODE solution for {func_name}({indep_sym}) | range: {indep_sym} in "
                    f"[{t_range[0]:g}, {t_range[1]:g}]"
                    + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in param_vals.items())
                       if param_vals else "")
                )
                snapshot_button(
                    key=f"ode_{func_name}",
                    title=f"{func_name}({indep_sym}) solution curve",
                    caption=ode_caption,
                    render_fn=lambda rf=rhs_final, ind=indep_sym, tr=t_range, fn=func_name:
                        snapshot_ode_plot(fn, ind, rf, tr),
                )

                eval_point = st.number_input(f"Evaluate {func_name} at {indep_sym} =",
                                               value=float(t_range[1]), key=f"odeeval_{func_name}")
                eval_value = float(np.real(complex(f(eval_point))))
                st.write(f"**{func_name}({indep_sym}={eval_point:g}) = {eval_value:.6g}**")
                if st.button(f"➕ Extract this value to workspace", key=f"odeextract_{func_name}"):
                    unit = next((v.unit for v in model.variables if v.symbol == func_name), None)
                    ws.store(f"{func_name}_at_{eval_point:g}", eval_value,
                             source=f"{st.session_state['problem_text'][:60]}...", unit=unit)
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.caption(f"Couldn't plot this solution numerically: {e}")

    # ---- phase portrait: only for a genuinely COUPLED 2-variable system
    # (dx/dt = f(x,y), dy/dt = g(x,y)) -- a view neither function's own
    # plot above can give, since it's about how the two variables move
    # TOGETHER, not either one against time alone. Silently skipped (not
    # shown with a caveat) for anything that isn't exactly two mutually-
    # coupled functions with fully-known parameters, same convention as
    # the cobweb diagram in render_recurrence_solution below.
    ode_eqs = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    for group in group_coupled_odes(ode_eqs):
        if len(group) != 2:
            continue
        rhs_by_name = {}
        for e in group:
            lhs_func = next(iter(e.sympy_eq.lhs.atoms(AppliedUndef)))
            rhs_by_name[str(lhs_func.func)] = e.sympy_eq.rhs
        names = sorted(rhs_by_name)
        if len(names) != 2 or not all(n in ode_solutions for n in names):
            continue
        fname_x, fname_y = names
        indep_sym = ode_solutions[fname_x].lhs.args[0]
        x_sym, y_sym = sp.Symbol(fname_x), sp.Symbol(fname_y)
        applied_subs = {sp.Function(fname_x)(indep_sym): x_sym, sp.Function(fname_y)(indep_sym): y_sym}
        subs_known = _known_substitutions(model)
        dx_expr = rhs_by_name[fname_x].subs(subs_known).subs(applied_subs)
        dy_expr = rhs_by_name[fname_y].subs(subs_known).subs(applied_subs)
        if (dx_expr.free_symbols | dy_expr.free_symbols) - {x_sym, y_sym}:
            continue  # unresolved parameters -- skip rather than show a field that's wrong

        sol_x = ode_solutions[fname_x].rhs.subs(subs_known)
        sol_y = ode_solutions[fname_y].rhs.subs(subs_known)
        if (sol_x.free_symbols | sol_y.free_symbols) - {indep_sym}:
            continue

        try:
            dx_f = sp.lambdify((x_sym, y_sym), dx_expr, "numpy")
            dy_f = sp.lambdify((x_sym, y_sym), dy_expr, "numpy")
            fx = sp.lambdify(indep_sym, sol_x, "numpy")
            fy = sp.lambdify(indep_sym, sol_y, "numpy")
            with st.expander(f"🌀 Phase portrait: {fname_x} vs. {fname_y}"):
                st.caption("The direction field shows how the system would move from ANY "
                            "starting point; the highlighted path is the actual solved "
                            "trajectory from its initial condition.")
                t_range = st.slider(f"{indep_sym} range", 0.0, 100.0, (0.0, 10.0),
                                      key=f"phase_trange_{fname_x}_{fname_y}")
                ts = np.linspace(t_range[0], t_range[1], 200)
                txs = np.real(np.array([complex(fx(tv)) for tv in ts]))
                tys = np.real(np.array([complex(fy(tv)) for tv in ts]))
                pad_x = 0.2 * max(float(txs.max() - txs.min()), 1.0)
                pad_y = 0.2 * max(float(tys.max() - tys.min()), 1.0)
                phase_fig = build_phase_portrait(
                    dx_f, dy_f,
                    (float(txs.min() - pad_x), float(txs.max() + pad_x)),
                    (float(tys.min() - pad_y), float(tys.max() + pad_y)),
                    x_label=fname_x, y_label=fname_y, trajectory=(txs, tys))
                st.plotly_chart(phase_fig, width="stretch", key=f"phase_{fname_x}_{fname_y}")
        except Exception as exc:  # noqa: BLE001
            st.caption(f"Couldn't build a phase portrait: {exc}")


def render_recurrence_solution(ws: Workspace, model: ProblemModel):
    """Recurrence solution: discrete plot + evaluate-at-a-point."""
    # ---- recurrence solution: discrete plot + evaluate-at-a-point
    recurrence_solutions = solve_recurrence(model)
    if recurrence_solutions:
        st.markdown("### Recurrence (sequence) solution")
        for func_name, closed_form in recurrence_solutions.items():
            rec_eq = next((e for e in model.equations if e.kind == "recurrence"
                            and func_name in symbols_and_functions_used(e)), None)
            indep_sym = None
            if rec_eq is not None:
                funcs = rec_eq.sympy_eq.atoms(AppliedUndef) if hasattr(rec_eq.sympy_eq, "atoms") else set()
                indep_sym = _independent_variable(funcs)
            if indep_sym is None:
                indep_sym = sp.Symbol(model.independent_variable or "n")

            st.latex(f"{func_name}({indep_sym}) = {sp.latex(closed_form)}")

            # ---- induction proof: base case(s) + inductive step, reusing
            # the same substitution check verify_recurrence_solution already
            # does for the inductive step -- see proof.py
            if rec_eq is not None:
                base_cases = {}
                for ic in model.initial_conditions:
                    if ic.sympy_eq is None:
                        continue
                    lhs_funcs = ic.sympy_eq.lhs.atoms(AppliedUndef)
                    match = next((f for f in lhs_funcs if str(f.func) == func_name), None)
                    if match is not None and match.args[0].is_number:
                        base_cases[int(match.args[0])] = float(ic.sympy_eq.rhs)
                if base_cases:
                    with st.expander(f"📐 Show induction proof for {func_name}({indep_sym})"):
                        induction = build_recurrence_induction_proof(
                            rec_eq.sympy_eq, func_name, closed_form, indep_sym, base_cases)
                        if induction.error:
                            st.caption(induction.error)
                        else:
                            for step in induction.steps:
                                (st.success if step.verified else st.error)(f"**{step.label}**  \n{step.detail}")
                            (st.success if induction.valid else st.error)(induction.conclusion)

            closed_form_sub = closed_form.subs(_known_substitutions(model))
            remaining = sorted(closed_form_sub.free_symbols - {indep_sym}, key=str)

            param_vals = {}
            if remaining:
                st.caption("Remaining parameters:")
                pcols = st.columns(min(4, len(remaining)))
                for i, s in enumerate(remaining):
                    with pcols[i % len(pcols)]:
                        param_vals[s] = st.slider(str(s), 0.01, 20.0, 1.0, key=f"recparam_{func_name}_{s}")
            closed_form_final = closed_form_sub.subs(param_vals)

            try:
                f = sp.lambdify(indep_sym, closed_form_final, "numpy")
                n_range = st.slider(f"{indep_sym} range (terms shown)", 0, 100, (0, 10),
                                      key=f"recrange_{func_name}")
                ns = np.arange(n_range[0], n_range[1] + 1)
                ys = np.real(np.array([complex(f(n)) for n in ns]))
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ns, y=ys, mode="markers", name=f"{func_name}({indep_sym})",
                                           marker=dict(size=8)))
                fig.update_layout(xaxis_title=str(indep_sym), yaxis_title=func_name)
                st.plotly_chart(fig, width='stretch')

                rec_caption = (
                    f"Recurrence solution for {func_name}({indep_sym}) | range: {indep_sym} in "
                    f"[{n_range[0]}, {n_range[1]}]"
                    + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in param_vals.items())
                       if param_vals else "")
                )
                snapshot_button(
                    key=f"recurrence_{func_name}",
                    title=f"{func_name}({indep_sym}) sequence",
                    caption=rec_caption,
                    render_fn=lambda cf=closed_form_final, ind=indep_sym, nr=n_range, fn=func_name:
                        snapshot_recurrence_plot(fn, ind, cf, nr),
                )

                # ---- cobweb diagram: only defined for a genuinely
                # first-order recurrence (a(n+1) = g(a(n)), nothing else)
                # -- extract_step_map returns None for anything else
                # (second-order, n appearing directly in the map, etc.),
                # in which case this section is silently skipped rather
                # than shown broken or with a caveat nobody asked for.
                step_map = extract_step_map(model, func_name)
                if step_map is not None:
                    g_expr, g_var = step_map
                    with st.expander(f"🕸️ Cobweb diagram for {func_name}({indep_sym})"):
                        st.caption("Shows convergence, oscillation, or divergence at a glance: "
                                    "the curve is the one-step map, the diagonal is where a value "
                                    "would repeat itself, and the staircase traces the sequence "
                                    "bouncing between the two.")
                        try:
                            g_numeric = sp.lambdify(g_var, g_expr.subs(param_vals), "numpy")
                            x0 = float(ys[0])
                            cob_span = max(abs(x0), 1.0) * 2.5
                            cob_range = st.slider("Plot range", -10.0 * cob_span, 10.0 * cob_span,
                                                    (x0 - cob_span, x0 + cob_span),
                                                    key=f"cobweb_range_{func_name}")
                            cob_steps = st.slider("Steps to show", 2, 60, min(n_range[1] - n_range[0], 25) or 10,
                                                    key=f"cobweb_steps_{func_name}")
                            cob_fig = build_cobweb_plot(g_numeric, x0, cob_range, n_steps=cob_steps,
                                                          x_label=f"{func_name}({indep_sym})")
                            st.plotly_chart(cob_fig, width="stretch", key=f"cobweb_{func_name}")
                        except Exception as exc:  # noqa: BLE001
                            st.caption(f"Couldn't build a cobweb diagram: {exc}")

                eval_point = st.number_input(f"Evaluate {func_name} at {indep_sym} =",
                                               value=int(n_range[1]), step=1, key=f"receval_{func_name}")
                eval_value = float(np.real(complex(f(eval_point))))
                st.write(f"**{func_name}({indep_sym}={eval_point:g}) = {eval_value:.6g}**")
                if st.button(f"➕ Extract this value to workspace", key=f"recextract_{func_name}"):
                    unit = next((v.unit for v in model.variables if v.symbol == func_name), None)
                    ws.store(f"{func_name}_at_{eval_point:g}", eval_value,
                             source=f"{st.session_state['problem_text'][:60]}...", unit=unit)
                    st.rerun()
            except Exception as e:  # noqa: BLE001
                st.caption(f"Couldn't plot this solution numerically: {e}")
