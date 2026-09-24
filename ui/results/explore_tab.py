"""
The Explore tab of a solved problem: dependency graph, bulk uncertainty, parameter sweeps, and the live plots.
"""
import streamlit as st
import numpy as np
import pandas as pd
from modules.equation_engine import ProblemModel, target_kind
from modules.dependency_graph import build_dependency_graph
from modules.monte_carlo import run_monte_carlo, UncertainVariable, MAX_SAMPLES as MC_MAX_SAMPLES
from modules.parameter_sweep import sweep_parameters, sweep_result_to_grid
from modules.plotter import (
    plottable_free_symbols,
    build_plot,
    build_surface_plot,
    build_feasible_region_plot,
    build_dependency_graph_plot,
    build_contour_plot,
    build_sweep_heatmap,
)
from modules.plot_snapshot import (
    snapshot_line_plot,
    snapshot_surface_plot,
    snapshot_feasible_region,
    snapshot_dependency_graph,
    snapshot_contour_plot,
    snapshot_sweep_heatmap,
)
from ui.common import format_download_button, snapshot_button


def render_dependency_and_sweeps(model: ProblemModel, tab_explore):
    """Explore tab, first half: dependency graph, bulk uncertainty and parameter sweeps."""
    # ---- dependency graph: which known/unknown variables feed into
    # which equations -- most useful once there's more than one equation
    # to keep straight (see dependency_graph.py)
    dep_nodes, dep_edges = build_dependency_graph(model)
    equation_node_count = sum(1 for n in dep_nodes if n.kind == "equation")
    with tab_explore:
        if equation_node_count >= 2:
            with st.expander("🕸️ Dependency graph"):
                fig = build_dependency_graph_plot(dep_nodes, dep_edges)
                st.plotly_chart(fig, width='stretch')
                snapshot_button(
                    key="dependency_graph",
                    title="Dependency graph",
                    caption=f"{model.problem_domain} -- variable/equation dependencies",
                    render_fn=lambda: snapshot_dependency_graph(dep_nodes, dep_edges),
                )

        # ---- N-dimensional parameter sweep: grid-sweep two or more of
        # a problem's own inputs at once and get a results TABLE (plus
        # a heatmap for exactly 2 swept variables) -- distinct from
        # chains.sweep_step_binding (one variable, across a whole
        # CHAIN) and the interactive plot's own single-variable 1D
        # sweep. See parameter_sweep.py.
        sweep_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
        sweepable_vars = [v for v in model.variables if v.known_value is not None]
        if sweep_targets and len(sweepable_vars) >= 2:
            with st.expander("📊 N-dimensional parameter sweep"):
                st.caption("Grid-sweep two or more inputs at once and get a results table -- the "
                            "shape of a real sensitivity study, not just a single line read one "
                            "point at a time.")
                sweep_target = st.selectbox("Target", sweep_targets, key="sweep_target")
                sweep_symbols = st.multiselect(
                    "Which inputs to sweep? (2 or more for a genuine grid)",
                    [v.symbol for v in sweepable_vars], key="sweep_symbols",
                )
                sweep_ranges = {}
                if sweep_symbols:
                    sweep_default_rows = []
                    for sym in sweep_symbols:
                        var = next(v for v in sweepable_vars if v.symbol == sym)
                        center = float(var.known_value)
                        width = abs(center) * 0.2 or 1.0
                        sweep_default_rows.append({"Symbol": sym, "Low": center - width,
                                                     "High": center + width, "Points": 5})
                    sweep_edited = st.data_editor(
                        pd.DataFrame(sweep_default_rows), hide_index=True, width='stretch',
                        key=f"sweep_editor_{','.join(sorted(sweep_symbols))}",
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
                            "Low": st.column_config.NumberColumn("Low", format="%.4g"),
                            "High": st.column_config.NumberColumn("High", format="%.4g"),
                            "Points": st.column_config.NumberColumn("Points", min_value=2,
                                                                       max_value=50, step=1),
                        },
                    )
                    for _, row in sweep_edited.iterrows():
                        lo, hi, n_points = row["Low"], row["High"], row["Points"]
                        if lo is not None and hi is not None and n_points and int(n_points) >= 2:
                            sweep_ranges[row["Symbol"]] = list(
                                np.linspace(min(lo, hi), max(lo, hi), int(n_points)))

                if st.button("Run sweep", key="sweep_run_button") and sweep_ranges:
                    try:
                        sweep_result = sweep_parameters(model, sweep_target, sweep_ranges)
                    except ValueError as e:
                        st.error(str(e))
                        sweep_result = None
                    st.session_state["sweep_result"] = sweep_result

                sweep_result = st.session_state.get("sweep_result")
                if sweep_result is not None:
                    sweep_df = pd.DataFrame(sweep_result.rows)
                    st.dataframe(sweep_df, width='stretch', hide_index=True)

                    if len(sweep_result.swept_symbols) == 2:
                        x_sym, y_sym = sweep_result.swept_symbols
                        gx, gy, gz = sweep_result_to_grid(sweep_result)
                        heatmap_fig = build_sweep_heatmap(gx, gy, gz, x_sym, y_sym, sweep_result.target)
                        st.plotly_chart(heatmap_fig, width='stretch', key="sweep_heatmap")
                        format_download_button(
                            key="sweep_heatmap", file_stem=f"sweep_{sweep_result.target}",
                            render_fn=lambda fmt, gx=gx, gy=gy, gz=gz, xs=x_sym, ys=y_sym,
                                             t=sweep_result.target:
                                snapshot_sweep_heatmap(gx, gy, gz, xs, ys, t, fmt=fmt),
                        )

                    st.download_button("⬇️ Download sweep as CSV", data=sweep_df.to_csv(index=False),
                                         file_name=f"sweep_{sweep_result.target}.csv", mime="text/csv",
                                         key="sweep_csv_download")

        # ---- bulk uncertainty propagation across ALL targets: pick
        # uncertain inputs ONCE and run Monte Carlo for every algebraic
        # target with a SHARED seed (so every target's samples are
        # drawn from the same underlying random draws, for a
        # consistent comparison), rather than repeating the single-
        # target panel further down once per target. Only meaningful
        # when there's more than one target to begin with.
        bulk_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
        bulk_known_vars = [v for v in model.variables if v.known_value is not None]
        if len(bulk_targets) >= 2 and bulk_known_vars:
            with st.expander("🎲 Uncertainty propagation across all targets"):
                st.caption("Pick uncertain inputs once and run Monte Carlo for every target at "
                            "once, reusing the same random draws for each, rather than repeating "
                            "the single-target panel below once per target.")
                bulk_symbols = st.multiselect(
                    "Which inputs have uncertainty?", [v.symbol for v in bulk_known_vars],
                    key="bulk_mc_vars",
                )
                bulk_uncertain_vars = []
                if bulk_symbols:
                    bulk_default_rows = [
                        {"Symbol": sym, "Std (±)": abs(next(
                            v for v in bulk_known_vars if v.symbol == sym).known_value) * 0.05 or 0.1}
                        for sym in bulk_symbols
                    ]
                    bulk_edited = st.data_editor(
                        pd.DataFrame(bulk_default_rows), hide_index=True, width='stretch',
                        key=f"bulk_mc_editor_{','.join(sorted(bulk_symbols))}",
                        column_config={
                            "Symbol": st.column_config.TextColumn("Symbol", disabled=True),
                            "Std (±)": st.column_config.NumberColumn("Std (±)", min_value=0.0,
                                                                        format="%.4g"),
                        },
                    )
                    for _, row in bulk_edited.iterrows():
                        std_val = row["Std (\u00b1)"]
                        if std_val is not None and std_val > 0:
                            var = next(v for v in bulk_known_vars if v.symbol == row["Symbol"])
                            bulk_uncertain_vars.append(
                                UncertainVariable(symbol=row["Symbol"], mean=var.known_value,
                                                    std=float(std_val)))

                bulk_seed_key = "bulk_mc_seed"
                if bulk_seed_key not in st.session_state:
                    st.session_state[bulk_seed_key] = int(
                        np.random.default_rng().integers(0, 2**31 - 1))
                bulk_seed_cols = st.columns([3, 1])
                with bulk_seed_cols[0]:
                    st.number_input(
                        "Seed", min_value=0, max_value=2**31 - 1, key=bulk_seed_key,
                        help="Same seed across every target -- the same underlying random draws "
                             "are reused for each, for a consistent comparison.",
                    )
                with bulk_seed_cols[1]:
                    st.button(
                        "🎲 New seed", key="bulk_mc_randomize",
                        on_click=lambda: st.session_state.update(
                            {bulk_seed_key: int(np.random.default_rng().integers(0, 2**31 - 1))}),
                    )

                bulk_n = st.slider("Number of samples", 100, min(MC_MAX_SAMPLES, 10000), 1000,
                                     key="bulk_mc_n")

                if st.button("Run for all targets", key="bulk_mc_run") and bulk_uncertain_vars:
                    bulk_rows = []
                    with st.spinner(f"Sampling {bulk_n} times for {len(bulk_targets)} targets..."):
                        for t in bulk_targets:
                            try:
                                r = run_monte_carlo(model, t, bulk_uncertain_vars, n_samples=bulk_n,
                                                      seed=st.session_state[bulk_seed_key])
                            except ValueError as e:
                                bulk_rows.append({"target": t, "mean": None, "std": None, "p5": None,
                                                    "p95": None, "n_failed": None, "seed": None,
                                                    "error": str(e)})
                                continue
                            bulk_rows.append({"target": t, "mean": r.mean, "std": r.std, "p5": r.p5,
                                                "p95": r.p95, "n_failed": r.n_failed, "seed": r.seed,
                                                "error": None})
                    st.session_state["bulk_mc_results"] = bulk_rows

                bulk_rows = st.session_state.get("bulk_mc_results")
                if bulk_rows:
                    bulk_df = pd.DataFrame(bulk_rows)
                    st.dataframe(bulk_df, width='stretch', hide_index=True)
                    st.download_button("⬇️ Download as CSV", data=bulk_df.to_csv(index=False),
                                         file_name="uncertainty_all_targets.csv", mime="text/csv",
                                         key="bulk_mc_csv_download")



def render_interactive_plot(model: ProblemModel, edited_values, tab_explore):
    """Explore tab, second half: the live 1D/2D/contour/overlay/feasible-region plots."""
    # ---- interactive plot (algebraic equations only -- inequalities and
    # ODEs are visualized in their own dedicated sections above)
    with tab_explore:
        plottable = [e for e in model.equations if e.kind == "equation" and e.sympy_eq is not None
                     and len(plottable_free_symbols(e, set())) >= 1]
        if plottable:
            st.markdown("### Interactive plot")
            eq_choice_name = st.selectbox("Equation to plot", [e.name for e in plottable])
            eq_choice = next(e for e in plottable if e.name == eq_choice_name)
            free_syms = plottable_free_symbols(eq_choice, set())

            plot_mode = "2D line"
            if len(free_syms) >= 2:
                plot_mode = st.radio("Plot type", ["2D line", "3D surface", "Contour"], horizontal=True)

            if plot_mode == "3D surface" and len(free_syms) >= 2:
                x_symbol = st.selectbox("X-axis variable", free_syms, key="surf_x")
                y_candidates = [s for s in free_syms if s != x_symbol]
                y_symbol = st.selectbox("Y-axis variable", y_candidates, key="surf_y")
                other_syms = [s for s in free_syms if s not in (x_symbol, y_symbol)]

                param_values = {}
                if other_syms:
                    st.caption("Adjust the remaining parameters:")
                    pcols = st.columns(min(4, len(other_syms)))
                    for i, s in enumerate(other_syms):
                        default = edited_values.get(s, 1.0) or 1.0
                        with pcols[i % len(pcols)]:
                            param_values[s] = st.slider(
                                s, min_value=float(default) - 10, max_value=float(default) + 10,
                                value=float(default), key=f"surf_slider_{s}",
                            )

                x_default = edited_values.get(x_symbol, 10.0) or 10.0
                y_default = edited_values.get(y_symbol, 10.0) or 10.0
                x_range = st.slider("X-axis range", -50.0, 50.0,
                                     (min(0.0, x_default - 10), x_default + 10), key="surf_xr")
                y_range = st.slider("Y-axis range", -50.0, 50.0,
                                     (min(0.0, y_default - 10), y_default + 10), key="surf_yr")

                z_target = None
                if model.solve_for:
                    z_candidates = [t for t in model.solve_for
                                     if t not in (x_symbol, y_symbol) and target_kind(model, t) == "equation"]
                    if z_candidates:
                        z_target = st.selectbox("Z-axis target (solve equation for)", z_candidates)

                fig = build_surface_plot(eq_choice, x_symbol, y_symbol, param_values, x_range, y_range,
                                           z_target=z_target)
                st.plotly_chart(fig, width='stretch')

                surf_caption = (
                    f"Equation: {eq_choice.name} | x={x_symbol} [{x_range[0]:g}, {x_range[1]:g}] | "
                    f"y={y_symbol} [{y_range[0]:g}, {y_range[1]:g}]"
                    + (f" | z solved for: {z_target}" if z_target else "")
                    + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in param_values.items())
                       if param_values else "")
                )
                snapshot_button(
                    key=f"surface_{eq_choice.name}_{x_symbol}_{y_symbol}",
                    title=f"{eq_choice.name}: {z_target or 'residual'} vs {x_symbol}, {y_symbol}",
                    caption=surf_caption,
                    render_fn=lambda ec=eq_choice, xs=x_symbol, ys=y_symbol, pv=param_values,
                                     xr=x_range, yr=y_range, zt=z_target:
                        snapshot_surface_plot(ec, xs, ys, pv, xr, yr, z_target=zt),
                )
                format_download_button(
                    key=f"surface_{eq_choice.name}_{x_symbol}_{y_symbol}",
                    file_stem=f"{eq_choice.name}_surface",
                    render_fn=lambda fmt, ec=eq_choice, xs=x_symbol, ys=y_symbol, pv=param_values,
                                        xr=x_range, yr=y_range, zt=z_target:
                        snapshot_surface_plot(ec, xs, ys, pv, xr, yr, z_target=zt, fmt=fmt),
                )

            elif plot_mode == "Contour" and len(free_syms) >= 2:
                x_symbol = st.selectbox("X-axis variable", free_syms, key="contour_x")
                y_candidates = [s for s in free_syms if s != x_symbol]
                y_symbol = st.selectbox("Y-axis variable", y_candidates, key="contour_y")
                other_syms = [s for s in free_syms if s not in (x_symbol, y_symbol)]

                param_values = {}
                if other_syms:
                    st.caption("Adjust the remaining parameters:")
                    pcols = st.columns(min(4, len(other_syms)))
                    for i, s in enumerate(other_syms):
                        default = edited_values.get(s, 1.0) or 1.0
                        with pcols[i % len(pcols)]:
                            param_values[s] = st.slider(
                                s, min_value=float(default) - 10, max_value=float(default) + 10,
                                value=float(default), key=f"contour_slider_{s}",
                            )

                x_default = edited_values.get(x_symbol, 10.0) or 10.0
                y_default = edited_values.get(y_symbol, 10.0) or 10.0
                x_range = st.slider("X-axis range", -50.0, 50.0,
                                     (min(0.0, x_default - 10), x_default + 10), key="contour_xr")
                y_range = st.slider("Y-axis range", -50.0, 50.0,
                                     (min(0.0, y_default - 10), y_default + 10), key="contour_yr")

                z_target = None
                if model.solve_for:
                    z_candidates = [t for t in model.solve_for
                                     if t not in (x_symbol, y_symbol) and target_kind(model, t) == "equation"]
                    if z_candidates:
                        z_target = st.selectbox("Contour value (solve equation for)", z_candidates,
                                                  key="contour_z_target")

                fig = build_contour_plot(eq_choice, x_symbol, y_symbol, param_values, x_range, y_range,
                                           z_target=z_target)
                st.plotly_chart(fig, width='stretch')
                st.caption("Same relationship as the 3D surface, viewed from directly above as level "
                            "lines -- often easier to read exact values off of, with no rotation needed.")

                contour_caption = (
                    f"Equation: {eq_choice.name} | x={x_symbol} [{x_range[0]:g}, {x_range[1]:g}] | "
                    f"y={y_symbol} [{y_range[0]:g}, {y_range[1]:g}]"
                    + (f" | contours of: {z_target}" if z_target else "")
                    + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in param_values.items())
                       if param_values else "")
                )
                snapshot_button(
                    key=f"contour_{eq_choice.name}_{x_symbol}_{y_symbol}",
                    title=f"{eq_choice.name}: contours of {z_target or 'residual'} vs {x_symbol}, {y_symbol}",
                    caption=contour_caption,
                    render_fn=lambda ec=eq_choice, xs=x_symbol, ys=y_symbol, pv=param_values,
                                     xr=x_range, yr=y_range, zt=z_target:
                        snapshot_contour_plot(ec, xs, ys, pv, xr, yr, z_target=zt),
                )
                format_download_button(
                    key=f"contour_{eq_choice.name}_{x_symbol}_{y_symbol}",
                    file_stem=f"{eq_choice.name}_contour",
                    render_fn=lambda fmt, ec=eq_choice, xs=x_symbol, ys=y_symbol, pv=param_values,
                                        xr=x_range, yr=y_range, zt=z_target:
                        snapshot_contour_plot(ec, xs, ys, pv, xr, yr, z_target=zt, fmt=fmt),
                )

            else:
                x_symbol = st.selectbox("X-axis variable", free_syms, key="line_x")
                other_syms = [s for s in free_syms if s != x_symbol]

                param_values = {}
                if other_syms:
                    st.caption("Adjust the remaining parameters -- the plot updates live:")
                    pcols = st.columns(min(4, len(other_syms)))
                    for i, s in enumerate(other_syms):
                        default = edited_values.get(s, 1.0) or 1.0
                        with pcols[i % len(pcols)]:
                            param_values[s] = st.slider(
                                s, min_value=float(default) - 10, max_value=float(default) + 10,
                                value=float(default), key=f"slider_{s}",
                            )

                x_default = edited_values.get(x_symbol, 10.0) or 10.0
                x_range = st.slider("X-axis range", -50.0, 50.0,
                                     (min(0.0, x_default - 10), x_default + 10), key="line_xr")

                y_target = None
                if model.solve_for:
                    candidates = [t for t in model.solve_for
                                   if t != x_symbol and target_kind(model, t) == "equation"]
                    if candidates:
                        y_target = st.selectbox("Y-axis target (solve equation for)", candidates)

                log_cols = st.columns(2)
                with log_cols[0]:
                    x_log = st.checkbox("Log X-axis", key="line_x_log")
                with log_cols[1]:
                    y_log = st.checkbox("Log Y-axis", key="line_y_log")
                if x_log or y_log:
                    st.caption("A power-law relationship is a straight line on log-log axes; an "
                                "exponential one is a straight line with only the Y-axis logged.")

                fig = build_plot(model, eq_choice, x_symbol, param_values, x_range, y_target=y_target,
                                   x_log=x_log, y_log=y_log)
                st.plotly_chart(fig, width='stretch')

                line_caption = (
                    f"Equation: {eq_choice.name} | x={x_symbol} [{x_range[0]:g}, {x_range[1]:g}]"
                    + (f" | y solved for: {y_target}" if y_target else "")
                    + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in param_values.items())
                       if param_values else "")
                )
                snapshot_button(
                    key=f"line_{eq_choice.name}_{x_symbol}",
                    title=f"{eq_choice.name}: {y_target or 'residual'} vs {x_symbol}",
                    caption=line_caption,
                    render_fn=lambda ec=eq_choice, xs=x_symbol, pv=param_values, xr=x_range, yt=y_target,
                                     xl=x_log, yl=y_log:
                        snapshot_line_plot(ec, xs, pv, xr, y_target=yt, x_log=xl, y_log=yl),
                )
                format_download_button(
                    key=f"line_{eq_choice.name}_{x_symbol}",
                    file_stem=f"{eq_choice.name}_line",
                    render_fn=lambda fmt, ec=eq_choice, xs=x_symbol, pv=param_values, xr=x_range, yt=y_target,
                                        xl=x_log, yl=y_log:
                        snapshot_line_plot(ec, xs, pv, xr, y_target=yt, x_log=xl, y_log=yl, fmt=fmt),
                )

        # ---- feasible region (multiple inequality constraints, 2 free variables)
        inequality_eqs = [e for e in model.equations if e.kind == "inequality" and e.sympy_eq is not None]
        if len(inequality_eqs) >= 1:
            all_ineq_symbols = set()
            for e in inequality_eqs:
                all_ineq_symbols |= {s.name for s in e.sympy_eq.free_symbols}
            # only known-fixed symbols get sliders; the rest are candidate plot axes
            ineq_free_syms = sorted(all_ineq_symbols)
            if len(ineq_free_syms) >= 2:
                st.markdown("### Feasible region")
                st.caption("Shades where every selected constraint holds at once -- e.g. a budget "
                            "AND a time limit AND non-negativity, simultaneously.")
                selected_constraints = st.multiselect(
                    "Constraints to include", [e.name for e in inequality_eqs],
                    default=[e.name for e in inequality_eqs], key="region_constraints",
                )
                region_x = st.selectbox("X-axis variable", ineq_free_syms, key="region_x")
                region_y_candidates = [s for s in ineq_free_syms if s != region_x]
                region_y = st.selectbox("Y-axis variable", region_y_candidates, key="region_y")
                other_ineq_syms = [s for s in ineq_free_syms if s not in (region_x, region_y)]

                region_params = {}
                if other_ineq_syms:
                    st.caption("Fix the remaining constraint parameters:")
                    rcols = st.columns(min(4, len(other_ineq_syms)))
                    for i, s in enumerate(other_ineq_syms):
                        default = edited_values.get(s, 1.0) or 1.0
                        with rcols[i % len(rcols)]:
                            region_params[s] = st.number_input(s, value=float(default), key=f"region_param_{s}")

                rx_default = edited_values.get(region_x, 10.0) or 10.0
                ry_default = edited_values.get(region_y, 10.0) or 10.0
                region_x_range = st.slider("X-axis range", -50.0, 50.0,
                                             (min(0.0, rx_default - 10), rx_default + 10), key="region_xr")
                region_y_range = st.slider("Y-axis range", -50.0, 50.0,
                                             (min(0.0, ry_default - 10), ry_default + 10), key="region_yr")

                chosen = [e for e in inequality_eqs if e.name in selected_constraints]
                if chosen:
                    fig = build_feasible_region_plot(chosen, region_x, region_y, region_params,
                                                       region_x_range, region_y_range)
                    st.plotly_chart(fig, width='stretch')

                    region_caption = (
                        f"Constraints: {', '.join(c.name for c in chosen)} | x={region_x} "
                        f"[{region_x_range[0]:g}, {region_x_range[1]:g}] | y={region_y} "
                        f"[{region_y_range[0]:g}, {region_y_range[1]:g}]"
                        + (" | fixed: " + ", ".join(f"{k}={v:g}" for k, v in region_params.items())
                           if region_params else "")
                    )
                    snapshot_button(
                        key=f"region_{region_x}_{region_y}",
                        title="Feasible region",
                        caption=region_caption,
                        render_fn=lambda ch=chosen, rx=region_x, ry=region_y, rp=region_params,
                                         rxr=region_x_range, ryr=region_y_range:
                            snapshot_feasible_region(ch, rx, ry, rp, rxr, ryr),
                    )
