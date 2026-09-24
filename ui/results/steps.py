"""
The step-by-step solution section, split per feature: the step list, each target's answer extras, and the per-target analysis expanders.
"""
import streamlit as st
import numpy as np
import pandas as pd
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel, target_kind
from modules.verifier import VerificationReport, _known_substitutions
from modules.solver import alternate_method_steps
from modules.sensitivity import sweep_input, tornado_analysis
from modules.monte_carlo import run_monte_carlo, UncertainVariable, MAX_SAMPLES as MC_MAX_SAMPLES
from modules.error_propagation import propagate_error, UncertainVariable as ErrorPropUncertainVariable
from modules.interval_arithmetic import propagate_interval
from modules.goal_seek import goal_seek
from modules.sig_figs import check_sig_figs, raw_known_value_strings
from modules.step_explainer import explain_step
from modules.tutor_mode import check_final_answer_guess
from modules.notebook_export import build_notebook
from modules.unit_conversion import sweep_conversions, preferred_conversion
from modules.code_export import formula_for_target, generate_python_function, generate_python_module
from modules.plotter import build_tornado_chart, build_sweep_chart, build_histogram_plot
from modules.plot_snapshot import snapshot_tornado_chart, snapshot_sweep_chart, snapshot_histogram_plot
from ui.common import format_download_button, snapshot_button
from modules.workspace import Workspace


def render_step_list(client: LMStudioClient, model: ProblemModel, report: VerificationReport, steps, target_name):
    """One target's heading, tutor-mode toggle, and the (optionally progressively revealed) steps."""
    st.markdown(f"#### Solving for `{target_name}`")

    # ---- guided/tutor mode: predict-then-reveal instead of a
    # full wall of steps immediately, plus a graded final-
    # answer guess (checked against the already-VERIFIED numeric
    # answer, not a free-form symbolic comparison against
    # individual steps -- see tutor_mode.py's module docstring
    # for why intermediate-step grading isn't attempted).
    tutor_on = st.checkbox("🎓 Tutor mode (reveal one step at a time)",
                             key=f"tutor_on_{target_name}")
    reveal_key = f"tutor_reveal_{target_name}"
    if tutor_on:
        st.session_state.setdefault(reveal_key, 0)
        if target_name in report.sympy_numeric_answers:
            guess_col1, guess_col2 = st.columns([3, 1])
            with guess_col1:
                guess_str = st.text_input(f"What do you think {target_name} equals?",
                                            key=f"tutor_guess_{target_name}")
            with guess_col2:
                st.write("")  # vertical alignment spacer
                check_clicked = st.button("Check", key=f"tutor_check_{target_name}")
            if check_clicked and guess_str.strip():
                result = check_final_answer_guess(guess_str, target_name, report)
                st.session_state[f"tutor_feedback_{target_name}"] = result
            feedback = st.session_state.get(f"tutor_feedback_{target_name}")
            if feedback is not None:
                if feedback.error:
                    st.warning(feedback.error)
                else:
                    (st.success if feedback.correct else st.error)(feedback.detail)
        reveal_count = st.session_state[reveal_key]
    else:
        reveal_count = len(steps)  # tutor mode off -- show everything, as before

    for i, step in enumerate(steps, start=1):
        if i > reveal_count:
            break
        st.markdown(f"**Step {i}: {step.description}**")
        st.latex(step.expression)
        if step.explanation:
            st.caption(step.explanation)
        # ---- step-level "explain just this" drill-down: a
        # narrower, more surgical sibling of the whole-problem
        # follow-up Q&A further down -- grounds the LLM only in
        # THIS step's own content, not the full derivation. See
        # step_explainer.py.
        with st.expander(f"🔍 Explain just step {i}"):
            explain_mode = st.radio(
                "How?", ["default", "simpler", "example"], horizontal=True,
                key=f"explain_mode_{target_name}_{i}",
                format_func=lambda m: {"default": "Explain", "simpler": "Simpler",
                                         "example": "With an example"}[m],
            )
            if st.button("Explain this step", key=f"explain_btn_{target_name}_{i}"):
                with st.spinner("Explaining..."):
                    st.session_state[f"explain_result_{target_name}_{i}"] = explain_step(
                        client, model, steps, i, target_name, explain_mode)
            explain_result = st.session_state.get(f"explain_result_{target_name}_{i}")
            if explain_result is not None:
                if explain_result.error:
                    st.error(explain_result.error)
                else:
                    st.info(explain_result.text)

    if tutor_on and reveal_count < len(steps):
        if st.button(f"👉 Reveal step {reveal_count + 1} of {len(steps)}",
                      key=f"tutor_reveal_btn_{target_name}"):
            st.session_state[reveal_key] += 1
            st.rerun()



def render_target_answer(ws: Workspace, model: ProblemModel, report: VerificationReport, problem_text: str, opt_result, target_name):
    """One target's numeric answer extras: workspace extract, sig-fig note, unit conversions, Python export."""
    sympy_val = report.sympy_numeric_answers.get(target_name)
    if sympy_val is not None:
        if st.button(f"➕ Extract {target_name} to workspace", key=f"extract_{target_name}"):
            unit = next((v.unit for v in model.variables if v.symbol == target_name), None)
            ws.store(target_name, sympy_val,
                     source=f"{problem_text[:60]}...", unit=unit)
            st.rerun()

        # ---- sig-fig discipline check: flags a final answer
        # reported with implausibly MORE precision than the
        # problem's own given inputs actually support -- see
        # sig_figs.py. Advisory only, same spirit as
        # plausibility.py's magnitude check, just for PRECISION
        # instead of magnitude.
        sig_fig_note = check_sig_figs(raw_known_value_strings(model), sympy_val, target_name)
        if sig_fig_note is not None:
            st.warning(f"🔢 {sig_fig_note.message}")

        # ---- unit conversion sweep: offer the same numeric
        # answer in a handful of common alternate units, once its
        # declared unit is known -- purely a display convenience,
        # doesn't touch the verified value itself
        target_unit = next((v.unit for v in model.variables if v.symbol == target_name), None)

        preferred_system = st.session_state.get("preferred_unit_system", "None")
        if preferred_system != "None":
            preferred = preferred_conversion(sympy_val, target_unit, preferred_system)
            if preferred is not None:
                alt_unit, alt_val = preferred
                st.caption(f"({preferred_system} preference: {alt_val:.6g} {alt_unit})")

        conversions = sweep_conversions(sympy_val, target_unit)
        if conversions:
            with st.expander(f"Also equals... ({target_name} in other units)"):
                for alt_unit, alt_val in conversions:
                    st.write(f"{alt_val:.6g} {alt_unit}")

    elif (target_kind(model, target_name) == "optimization" and opt_result
            and not opt_result.error and opt_result.critical_points):
        opt_val = opt_result.critical_points[0].get(target_name)
        if opt_val is not None and opt_val.is_number:
            if st.button(f"➕ Extract {target_name} to workspace", key=f"extract_opt_{target_name}"):
                unit = next((v.unit for v in model.variables if v.symbol == target_name), None)
                ws.store(target_name, float(opt_val),
                         source=f"{problem_text[:60]}...", unit=unit)
                st.rerun()

    # ---- runnable code export: an actual Python function
    # computing this target from its inputs (algebraic/ODE/
    # recurrence closed forms only -- see code_export.py), as
    # SOURCE TEXT someone can drop into their own project, not
    # just a copy-pasteable formula
    formula = formula_for_target(model, target_name)
    if formula is not None:
        unit_for_target = next((v.unit for v in model.variables if v.symbol == target_name), None)
        py_src = generate_python_function(
            formula, {v.symbol: v.meaning for v in model.variables}, unit_for_target,
        )
        st.download_button(
            f"⬇️ Get {target_name}(...) as Python", data=py_src,
            file_name=f"{target_name}.py", mime="text/x-python",
            key=f"pyexport_{target_name}",
        )



def render_alternate_method(model: ProblemModel, target_name):
    """'Show me another way' toggle: back-substitution and Cramer's rule."""
    if st.toggle(f"Show me another way to solve for {target_name}", key=f"altmethod_{target_name}"):
        alt_steps = alternate_method_steps(model, target_name, _known_substitutions(model))
        if alt_steps:
            for i, step in enumerate(alt_steps, start=1):
                st.markdown(f"**{step.description}**")
                st.latex(step.expression)
        else:
            st.caption("No alternate method available for this target.")



def render_monte_carlo(model: ProblemModel, known_vars_here, target_name):
    """Monte Carlo uncertainty propagation expander."""
    if known_vars_here:
        with st.expander(f"🎲 Uncertainty propagation for {target_name}"):
            st.caption("Give one or more known inputs a measurement uncertainty and see "
                        "how that uncertainty propagates through to this target -- the "
                        "whole spread of plausible answers, not just a single point estimate.")
            mc_symbols = st.multiselect(
                "Which inputs have uncertainty?", [v.symbol for v in known_vars_here],
                key=f"mc_vars_{target_name}",
            )
            uncertain_vars = []
            if mc_symbols:
                # a compact table -- one row per selected input -- rather than one
                # st.columns() slot per variable, which stacks into a long scroll of
                # full-width blocks on a narrow (phone) screen; a data_editor renders
                # as a single scrollable widget regardless of row count
                mc_default_rows = [
                    {"Symbol": sym, "Std (±)": abs(next(
                        v for v in known_vars_here if v.symbol == sym).known_value) * 0.05 or 0.1}
                    for sym in mc_symbols
                ]
                mc_edited = st.data_editor(
                    pd.DataFrame(mc_default_rows), hide_index=True, width='stretch',
                    key=f"mc_editor_{target_name}_{','.join(sorted(mc_symbols))}",
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
                        "Std (±)": st.column_config.NumberColumn("Std (±)", min_value=0.0,
                                                                    format="%.4g"),
                    },
                )
                for _, row in mc_edited.iterrows():
                    std_val = row["Std (\u00b1)"]
                    if std_val is not None and std_val > 0:
                        var = next(v for v in known_vars_here if v.symbol == row["Symbol"])
                        uncertain_vars.append(
                            UncertainVariable(symbol=row["Symbol"], mean=var.known_value,
                                                std=float(std_val)))
            mc_n = st.slider("Number of samples", 100, min(MC_MAX_SAMPLES, 10000), 1000,
                               key=f"mc_n_{target_name}")

            # ---- reproducibility: a Monte Carlo run is only
            # a citable/reproducible result if the seed that
            # produced it is visible and reusable -- default
            # to a fresh random seed each time this panel is
            # first opened for this target, but keep it
            # STABLE across reruns (not re-randomized on
            # every rerun) so the number shown always
            # matches what "Run Monte Carlo" will actually
            # use until the person explicitly asks for a
            # new one.
            mc_seed_key = f"mc_seed_{target_name}"
            if mc_seed_key not in st.session_state:
                st.session_state[mc_seed_key] = int(
                    np.random.default_rng().integers(0, 2**31 - 1))
            mc_seed_cols = st.columns([3, 1])
            with mc_seed_cols[0]:
                st.number_input(
                    "Seed", min_value=0, max_value=2**31 - 1, key=mc_seed_key,
                    help="Same seed + same inputs always reproduces the exact same "
                         "samples -- note this down alongside a result you're citing.",
                )
            with mc_seed_cols[1]:
                st.button(
                    "🎲 New seed", key=f"mc_randomize_{target_name}",
                    on_click=lambda k=mc_seed_key: st.session_state.update(
                        {k: int(np.random.default_rng().integers(0, 2**31 - 1))}),
                )

            if st.button("Run Monte Carlo", key=f"mc_run_{target_name}") and uncertain_vars:
                with st.spinner(f"Sampling {mc_n} times..."):
                    try:
                        mc_result = run_monte_carlo(model, target_name, uncertain_vars,
                                                      n_samples=mc_n,
                                                      seed=st.session_state[mc_seed_key])
                    except ValueError as e:
                        st.error(str(e))
                        mc_result = None
                st.session_state[f"mc_result_{target_name}"] = mc_result
            mc_result = st.session_state.get(f"mc_result_{target_name}")
            if mc_result is not None and mc_result.samples:
                st.write(f"**{target_name} = {mc_result.mean:.6g} ± {mc_result.std:.4g}** "
                          f"(5th–95th percentile: {mc_result.p5:.6g} to {mc_result.p95:.6g})")
                st.caption(f"Seed used: {mc_result.seed} -- reuse it above to reproduce "
                            "this exact run.")
                if mc_result.n_failed:
                    st.caption(f"{mc_result.n_failed} of {mc_result.n_requested} samples "
                                "didn't produce a real result and were excluded.")
                hist_fig = build_histogram_plot(mc_result.samples, target_name,
                                                   mean=mc_result.mean, p5=mc_result.p5,
                                                   p95=mc_result.p95)
                st.plotly_chart(hist_fig, width='stretch', key=f"mc_hist_{target_name}")
                format_download_button(
                    key=f"mc_hist_{target_name}",
                    file_stem=f"monte_carlo_{target_name}_seed{mc_result.seed}",
                    render_fn=lambda fmt, r=mc_result: snapshot_histogram_plot(
                        r.samples, target_name, mean=r.mean, p5=r.p5, p95=r.p95, fmt=fmt),
                )
            elif mc_result is not None:
                st.warning("No samples produced a real result -- try smaller std values.")



def render_analytic_error(model: ProblemModel, known_vars_here, target_name):
    """Analytic (first-order) error propagation expander."""
    # ---- analytic error propagation: the textbook first-
    # order propagation-of-uncertainty formula, instant and
    # exact for a linear target -- complementary to the
    # Monte Carlo panel above (which is more trustworthy for
    # a highly nonlinear target or large uncertainties, but
    # needs sampling and gives no formula).
    if known_vars_here:
        with st.expander(f"📐 Analytic error propagation for {target_name}"):
            st.caption("The standard first-order formula (σ² = Σ (∂f/∂xᵢ)² σᵢ²) -- "
                        "instant, and exact when the target is linear in its uncertain "
                        "inputs. Shows the actual formula, not just a number.")
            ep_symbols = st.multiselect(
                "Which inputs have uncertainty?", [v.symbol for v in known_vars_here],
                key=f"ep_vars_{target_name}",
            )
            ep_uncertain_vars = []
            if ep_symbols:
                ep_default_rows = [
                    {"Symbol": sym, "Std (±)": abs(next(
                        v for v in known_vars_here if v.symbol == sym).known_value) * 0.05 or 0.1}
                    for sym in ep_symbols
                ]
                ep_edited = st.data_editor(
                    pd.DataFrame(ep_default_rows), hide_index=True, width='stretch',
                    key=f"ep_editor_{target_name}_{','.join(sorted(ep_symbols))}",
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
                        "Std (±)": st.column_config.NumberColumn("Std (±)", min_value=0.0,
                                                                    format="%.4g"),
                    },
                )
                for _, row in ep_edited.iterrows():
                    std_val = row["Std (\u00b1)"]
                    if std_val is not None and std_val > 0:
                        var = next(v for v in known_vars_here if v.symbol == row["Symbol"])
                        ep_uncertain_vars.append(
                            ErrorPropUncertainVariable(symbol=row["Symbol"],
                                                         mean=var.known_value,
                                                         std=float(std_val)))
            if st.button("Compute", key=f"ep_run_{target_name}") and ep_uncertain_vars:
                try:
                    ep_result = propagate_error(model, target_name, ep_uncertain_vars)
                except ValueError as e:
                    st.error(str(e))
                    ep_result = None
                st.session_state[f"ep_result_{target_name}"] = ep_result
            ep_result = st.session_state.get(f"ep_result_{target_name}")
            if ep_result is not None:
                st.write(f"**{target_name} = {ep_result.value:.6g} ± {ep_result.std:.4g}**")
                if ep_result.formula_latex:
                    st.latex(f"{target_name} = {ep_result.formula_latex}")
                st.caption("Contribution to total variance:")
                for sym, frac in sorted(ep_result.contributions.items(), key=lambda kv: -kv[1]):
                    st.write(f"- `{sym}`: {frac:.0%} "
                              f"(∂{target_name}/∂{sym} = {ep_result.partials[sym]:.4g})")



def render_interval_bounds(model: ProblemModel, known_vars_here, target_name):
    """Interval-arithmetic guaranteed-bounds expander."""
    # ---- interval arithmetic: a GUARANTEED bound instead of
    # a statistical one -- "cannot be outside this range"
    # rather than "95% likely to be in this range", given
    # each input as a hard [lo, hi] range instead of a
    # distribution.
    if known_vars_here:
        with st.expander(f"📏 Guaranteed bounds for {target_name}"):
            st.caption("Give each uncertain input a hard range (not a probability "
                        "distribution) and get back a range for this target that is "
                        "GUARANTEED to contain every possible result -- not a confidence "
                        "interval, a proof.")
            iv_symbols = st.multiselect(
                "Which inputs have a range?", [v.symbol for v in known_vars_here],
                key=f"iv_vars_{target_name}",
            )
            iv_ranges = {}
            if iv_symbols:
                iv_default_rows = []
                for sym in iv_symbols:
                    var = next(v for v in known_vars_here if v.symbol == sym)
                    center = float(var.known_value)
                    default_width = abs(center) * 0.1 or 0.5
                    iv_default_rows.append({"Symbol": sym, "Low": center - default_width,
                                              "High": center + default_width})
                iv_edited = st.data_editor(
                    pd.DataFrame(iv_default_rows), hide_index=True, width='stretch',
                    key=f"iv_editor_{target_name}_{','.join(sorted(iv_symbols))}",
                    column_config={
                        "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
                        "Low": st.column_config.NumberColumn("Low", format="%.4g"),
                        "High": st.column_config.NumberColumn("High", format="%.4g"),
                    },
                )
                for _, row in iv_edited.iterrows():
                    lo, hi = row["Low"], row["High"]
                    if lo is not None and hi is not None:
                        iv_ranges[row["Symbol"]] = (float(min(lo, hi)), float(max(lo, hi)))
            if st.button("Compute bounds", key=f"iv_run_{target_name}") and iv_ranges:
                try:
                    iv_result = propagate_interval(model, target_name, iv_ranges)
                except ValueError as e:
                    st.error(str(e))
                    iv_result = None
                st.session_state[f"iv_result_{target_name}"] = iv_result
            iv_result = st.session_state.get(f"iv_result_{target_name}")
            if iv_result is not None:
                st.success(f"**{target_name} is guaranteed to fall within "
                            f"[{iv_result.lo:.6g}, {iv_result.hi:.6g}]** "
                            f"(width {iv_result.width:.4g})")
                if iv_result.formula_latex:
                    st.latex(f"{target_name} = {iv_result.formula_latex}")



def render_goal_seek(model: ProblemModel, other_vars, target_name):
    """Goal-seek (inverse solve) expander."""
    if other_vars:
        with st.expander(f"🎯 Goal seek: find the input for a target {target_name}"):
            st.caption("What value of an input makes this target hit a specific number? "
                        "Solves for it directly -- exactly when possible, numerically "
                        "otherwise.")
            gs_symbol = st.selectbox("Solve for which input?",
                                       [v.symbol for v in other_vars],
                                       key=f"gs_symbol_{target_name}")
            gs_target_value = st.number_input(f"Desired value of {target_name}",
                                                 key=f"gs_target_{target_name}")
            if st.button("Seek", key=f"gs_run_{target_name}"):
                try:
                    gs_result = goal_seek(model, target_name, gs_target_value, gs_symbol)
                except ValueError as e:
                    st.error(str(e))
                    gs_result = None
                else:
                    st.session_state[f"gs_result_{target_name}"] = gs_result
            gs_result = st.session_state.get(f"gs_result_{target_name}")
            if gs_result is not None:
                values_str = ", ".join(f"{v:.6g}" for v in gs_result.solutions)
                st.write(f"**{gs_result.seek_symbol} = {values_str}**")
                if gs_result.is_numerical:
                    st.caption("Found numerically (no exact closed-form inverse) -- "
                                "treat as an approximation.")
                if gs_result.formula:
                    st.latex(f"{gs_result.seek_symbol} = {gs_result.formula}")



def render_sensitivity(model: ProblemModel, target_name):
    """Sensitivity / tornado-analysis expander."""
    # ---- sensitivity / what-if analysis: which input
    # matters most to this target, and how does the answer
    # move if one input changes on its own -- see sensitivity.py.
    # Distinct from the uncertainty-propagation step above:
    # this asks "if I deliberately changed this input" not
    # "given its stated measurement error"
    with st.expander(f"🎚️ Sensitivity analysis for {target_name}"):
        pct_range = st.slider("Sweep range (± %)", 5, 50, 20, key=f"sens_pct_{target_name}") / 100
        entries = tornado_analysis(model, target_name, _known_substitutions(model), pct_range)
        if not entries:
            st.caption("No swept inputs available for this target.")
        else:
            tornado_fig = build_tornado_chart(entries)
            st.plotly_chart(tornado_fig, width='stretch', key=f"tornado_{target_name}")
            snapshot_button(
                key=f"tornado_{target_name}",
                title=f"Sensitivity (tornado chart) for {target_name}",
                caption=f"±{pct_range:.0%} sweep of each input",
                render_fn=lambda e=entries: snapshot_tornado_chart(e),
            )
            sweep_symbol = st.selectbox(
                "Sweep a single input in detail", [e.symbol for e in entries],
                key=f"sweep_pick_{target_name}",
            )
            sweep_result = sweep_input(model, target_name, sweep_symbol,
                                         _known_substitutions(model), pct_range)
            if sweep_result:
                sweep_fig = build_sweep_chart(sweep_result)
                st.plotly_chart(sweep_fig, width='stretch', key=f"sweep_{target_name}")
                snapshot_button(
                    key=f"sweep_{target_name}_{sweep_symbol}",
                    title=f"{target_name} vs {sweep_symbol}",
                    caption=f"±{pct_range:.0%} sweep of {sweep_symbol}",
                    render_fn=lambda sr=sweep_result: snapshot_sweep_chart(sr),
                )



def render_step_by_step(client: LMStudioClient, ws: Workspace, model: ProblemModel, report: VerificationReport, problem_text: str, opt_result, steps_by_target):
    """The step-by-step section: one block per solved target, with that target's answer extras and
    (for algebraic targets) the uncertainty / bounds / goal-seek / sensitivity expanders."""
    st.markdown("### Step-by-step solution")

    module_src = generate_python_module(model)
    dl_cols = st.columns(2)
    with dl_cols[0]:
        if not module_src.startswith('"""No exportable'):
            st.download_button(
                "⬇️ Get all formulas as one Python file", data=module_src,
                file_name="formulas.py", mime="text/x-python",
            )
    with dl_cols[1]:
        notebook_json = build_notebook(problem_text, model, steps_by_target)
        st.download_button(
            "⬇️ Get as Jupyter notebook", data=notebook_json,
            file_name="solution.ipynb", mime="application/x-ipynb+json",
        )
    for target_name, steps in steps_by_target.items():
        render_step_list(client, model, report, steps, target_name)
        render_target_answer(ws, model, report, problem_text, opt_result, target_name)
        # Everything below only applies to algebraic (equation-kind) targets.
        if target_kind(model, target_name) == "equation":
            render_alternate_method(model, target_name)
            # ---- Monte Carlo uncertainty propagation: give one or
            # more KNOWN inputs a measurement uncertainty (mean ±
            # std) and see the resulting SPREAD in this target,
            # sampled jointly across all of them at once -- see
            # monte_carlo.py. Complementary to (not a replacement
            # for) the sensitivity analysis below, which varies one
            # input at a time deterministically rather than
            # propagating a joint distribution.
            known_vars_here = [v for v in model.variables if v.known_value is not None]
            render_monte_carlo(model, known_vars_here, target_name)
            render_analytic_error(model, known_vars_here, target_name)
            render_interval_bounds(model, known_vars_here, target_name)
            # ---- goal seek / inverse solve: the inverse of
            # chains.sweep_step_binding's "vary this input and see
            # what happens" -- "what value of this input gives a
            # SPECIFIC target number", solved directly rather than
            # by sweeping and reading a chart.
            other_vars = [v for v in model.variables if v.symbol != target_name]
            render_goal_seek(model, other_vars, target_name)
            render_sensitivity(model, target_name)
