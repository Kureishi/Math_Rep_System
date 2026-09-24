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
from modules.ode_utils import solve_ode
from modules.recurrence_utils import solve_recurrence, _independent_variable
from modules.proof import build_recurrence_induction_proof
from modules.plot_snapshot import snapshot_ode_plot, snapshot_recurrence_plot
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
