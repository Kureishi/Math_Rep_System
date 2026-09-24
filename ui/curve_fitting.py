"""
Curve/data fitting mode (CSV or pasted (x, y) -> fitted symbolic model + fit-quality + statistical inference).
"""
import streamlit as st
import sympy as sp
import numpy as np
from modules.statistical_inference import (
    regression_inference,
    residual_diagnostics,
    bayesian_linear_regression,
    compare_polynomial_degrees,
)
from modules.plotter import build_fit_plot, build_overlay_plot
from modules.plot_snapshot import snapshot_fit_plot, snapshot_overlay_plot
from modules.curve_fitting import fit_curve, best_fit, parse_xy_csv, BUILTIN_FAMILIES
from ui.common import check_upload_size, format_download_button, render_template_bar, tooltip_header


def _render_statistical_layer(xs, ys, family, degree, expr_str, param_names):
    """The statistical/data layer on top of a fit: confidence intervals
    and hypothesis tests on the parameters, residual diagnostics, and
    Bayesian regression -- see statistical_inference.py. Kept inside
    the curve-fitting tab (not a separate sidebar mode) since none of
    this means anything without the fit it's built on."""
    with st.expander("📊 Statistical inference"):
        tab_ci, tab_diag, tab_bayes = st.tabs(
            ["Confidence intervals & hypothesis tests", "Residual diagnostics", "Bayesian regression"])

        with tab_ci:
            result = regression_inference(xs, ys, family, degree=degree, expr_str=expr_str,
                                            param_names=param_names)
            if result.error:
                st.warning(result.error)
            else:
                st.caption(f"{int(result.confidence_level * 100)}% confidence intervals, "
                            f"{result.degrees_of_freedom} degrees of freedom. Each t-test's null "
                            f"hypothesis is that the parameter is exactly zero.")
                for p in result.parameters:
                    st.write(f"**{p.name}** = {p.estimate:.5g} ± {p.std_error:.4g}  "
                              f"(t={p.t_statistic:.3g}, p={p.p_value:.3g})  "
                              f"CI: [{p.ci_lower:.5g}, {p.ci_upper:.5g}]")
                if result.f_statistic is not None:
                    verdict = "significant" if result.f_p_value < 0.05 else "not significant"
                    st.info(f"Overall model F-test: F={result.f_statistic:.4g}, "
                             f"p={result.f_p_value:.3g} ({verdict} at α=0.05) -- tests whether the "
                             f"model explains significantly more than a flat mean would. "
                             f"Adjusted R² = {result.adjusted_r_squared:.5f}.")

            if family in ("linear", "polynomial"):
                st.write("---")
                st.caption("Nested-model F-test: does a higher polynomial degree explain "
                            "significantly more variance, or just fit noise?")
                col1, col2 = st.columns(2)
                with col1:
                    reduced_deg = st.number_input("Reduced degree", min_value=1, max_value=9,
                                                    value=max(1, degree), key="nested_reduced_degree")
                with col2:
                    full_deg = st.number_input("Full degree", min_value=2, max_value=10,
                                                 value=max(2, degree + 1), key="nested_full_degree")
                if st.button("Compare", key="nested_compare_button"):
                    nm = compare_polynomial_degrees(xs, ys, int(reduced_deg), int(full_deg))
                    if nm.error:
                        st.error(nm.error)
                    elif nm.significant:
                        st.success(nm.verification_detail)
                    else:
                        st.warning(nm.verification_detail)

        with tab_diag:
            diag = residual_diagnostics(xs, ys, family, degree=degree, expr_str=expr_str,
                                          param_names=param_names)
            if diag.error:
                st.warning(diag.error)
            else:
                st.caption("These check whether the confidence intervals and p-values above are "
                            "actually trustworthy -- they assume normal, independent, "
                            "constant-variance residuals, and this is where that gets checked "
                            "rather than assumed.")
                tooltip_header("Normality", "Shapiro-Wilk test: checks whether the residuals look "
                                  "like they came from a normal distribution -- the p-values and "
                                  "confidence intervals above assume this.", level="")
                (st.success if diag.normality_p_value is None or diag.normality_p_value > 0.05
                 else st.warning)(diag.normality_note)
                tooltip_header("Autocorrelation", "Durbin-Watson statistic: checks whether "
                                  "consecutive residuals are related to each other (e.g. a run of "
                                  "positive residuals followed by a run of negative ones) rather than "
                                  "independent -- a value near 2 indicates no autocorrelation.", level="")
                (st.success if 1.5 <= diag.durbin_watson <= 2.5 else st.warning)(diag.durbin_watson_note)
                tooltip_header("Heteroscedasticity", "Breusch-Pagan test: checks whether the "
                                  "residuals' spread stays roughly constant across the data, or "
                                  "instead grows/shrinks systematically (e.g. bigger errors at larger "
                                  "x values) -- the latter would undermine the standard errors above.",
                                  level="")
                (st.success if diag.heteroscedasticity_p_value is None or diag.heteroscedasticity_p_value > 0.05
                 else st.warning)(diag.heteroscedasticity_note)

        with tab_bayes:
            prior_precision = st.select_slider(
                "Prior strength (how strongly parameters are pulled toward zero)",
                options=[1e-6, 1e-3, 1e-1, 1.0, 10.0, 100.0], value=1e-6,
                format_func=lambda v: "diffuse (≈ no prior)" if v <= 1e-3 else f"precision={v:g}",
                key="bayes_prior_precision",
                help="A Bayesian 'prior' is a belief about the parameters BEFORE seeing this data -- "
                      "here, a belief that they're probably close to zero. 'Diffuse' means barely any "
                      "such belief (the result should closely match an ordinary regression fit); a "
                      "higher precision means a stronger pull toward zero, useful if you have real "
                      "reason to expect small parameter values.")
            bayes = bayesian_linear_regression(xs, ys, family, degree=degree, expr_str=expr_str,
                                                 param_names=param_names, prior_precision=prior_precision)
            if bayes.error:
                st.warning(bayes.error)
            else:
                st.caption(f"{int(bayes.credible_level * 100)}% credible intervals from a conjugate "
                            f"Normal-Inverse-Gamma posterior (closed-form, no sampling).")
                for p in bayes.parameters:
                    st.write(f"**{p.name}** posterior mean = {p.posterior_mean:.5g} "
                              f"(std = {p.posterior_std:.4g})  "
                              f"credible interval: [{p.credible_lower:.5g}, {p.credible_upper:.5g}]")
                st.caption(bayes.comparison_note)


def render_curve_fitting_tab():
    """Sibling pipeline to the word-problem solver: input is a table of
    numbers (typed in or uploaded as CSV), not LLM-extracted text, and
    the output is a fitted symbolic model plus fit-quality metrics
    (R-squared, RMSE, residuals) rather than a verified derivation."""
    st.subheader("📈 Curve / data fitting")
    st.caption("Upload or paste (x, y) data and fit a symbolic model to it.")

    upload_col, paste_col = st.columns(2)
    csv_text = None
    with upload_col:
        uploaded_csv = st.file_uploader("Upload a 2-column CSV (x, y)", type=["csv"], key="fit_csv_upload")
        if uploaded_csv is not None and check_upload_size(uploaded_csv):
            csv_text = uploaded_csv.getvalue().decode("utf-8", errors="replace")
    with paste_col:
        pasted = st.text_area("...or paste CSV text", height=100,
                               placeholder="x,y\n1,2.1\n2,3.9\n3,6.2\n4,7.8\n",
                               key="fit_csv_paste")
        if pasted.strip():
            csv_text = pasted

    if not csv_text:
        st.info("Provide data via upload or paste to fit a curve.")
        return

    try:
        xs, ys, x_label, y_label = parse_xy_csv(csv_text)
    except ValueError as e:
        st.error(f"Couldn't read the data: {e}")
        return

    st.caption(f"Parsed {len(xs)} data points ({x_label} vs {y_label}).")

    family = st.selectbox(
        "Model family", list(BUILTIN_FAMILIES) + ["custom", "best fit (try all)"],
        format_func=lambda f: {"linear": "Linear (y = a·x + b)", "polynomial": "Polynomial",
                                 "exponential": "Exponential (y = a·e^(b·x))", "power": "Power (y = a·x^b)",
                                 "logarithmic": "Logarithmic (y = a·ln(x) + b)", "custom": "Custom (linear-in-parameters)",
                                 "best fit (try all)": "Best fit -- try every built-in family"}.get(f, f),
        key="fit_family",
    )

    degree = 2
    expr_str, param_names = None, None
    if family == "polynomial":
        degree = st.number_input("Degree", min_value=2, max_value=10, value=2, step=1)
    elif family == "custom":
        render_template_bar("curve_fit_model", lambda: {
            "fit_custom_expr": st.session_state.get("fit_custom_expr", ""),
            "fit_custom_params": st.session_state.get("fit_custom_params", ""),
        })
        expr_str = st.text_input("Model expression in x and named parameters",
                                   placeholder="a*sin(x) + b*x + c", key="fit_custom_expr")
        params_raw = st.text_input("Parameter names (comma-separated)", placeholder="a, b, c",
                                     key="fit_custom_params")
        param_names = [p.strip() for p in params_raw.split(",") if p.strip()]

    if st.button("Fit", type="primary"):
        # snapshotted at click time -- everything below re-reads from this,
        # NOT from the live widget values above, so that later widget
        # interactions in this tab (the log-axis checkboxes, the nested-
        # model "Compare" button added for the statistical layer) rerun
        # the script without wiping the fit out. st.button() only returns
        # True on the exact rerun triggered by clicking it -- any OTHER
        # widget's rerun sees it as False -- so gating the whole rest of
        # this function on that value directly (as this used to do) meant
        # ANY other interaction on the page (even an unrelated checkbox)
        # made the entire fit display vanish until "Fit" was clicked again.
        st.session_state["cf_snapshot"] = dict(xs=xs, ys=ys, x_label=x_label, y_label=y_label,
                                                 family=family, degree=degree, expr_str=expr_str,
                                                 param_names=param_names)

    snapshot = st.session_state.get("cf_snapshot")
    if snapshot is None:
        return
    xs, ys, x_label, y_label = snapshot["xs"], snapshot["ys"], snapshot["x_label"], snapshot["y_label"]
    family, degree = snapshot["family"], snapshot["degree"]
    expr_str, param_names = snapshot["expr_str"], snapshot["param_names"]

    if family == "best fit (try all)":
        results = best_fit(xs, ys)
        if not results:
            st.error("No built-in family could fit this data (check for non-positive x/y values, "
                      "which rule out exponential/power/logarithmic).")
            return
        ranked = sorted(results.items(), key=lambda kv: kv[1].r_squared, reverse=True)
        st.write("Ranked by R² (higher is better):")
        for fam, res in ranked:
            st.write(f"- **{fam}**: R² = {res.r_squared:.5f}, RMSE = {res.rmse:.5g}")
        best_family, result = ranked[0]
        st.success(f"Best fit: **{best_family}**")
        stat_family, stat_degree, stat_expr_str, stat_param_names = best_family, 2, None, None

        if len(ranked) >= 2:
            with st.expander("📊 Compare every candidate fit on one plot"):
                st.caption("Every family that successfully fit the data, overlaid on the same "
                            "axes -- often makes it visually obvious why one family won, not just "
                            "which one had the higher R².")
                grid_lo, grid_hi = min(xs), max(xs)
                pad = (grid_hi - grid_lo) * 0.05 if grid_hi > grid_lo else 1.0
                x_grid = np.linspace(grid_lo - pad, grid_hi + pad, 300).tolist()
                series = [{"x": xs, "y": ys, "name": "data", "mode": "markers"}]
                for fam, res in ranked:
                    f = sp.lambdify(sp.Symbol("x"), res.expr, "numpy")
                    try:
                        y_fit = np.broadcast_to(np.asarray(f(x_grid), dtype=float), (len(x_grid),)).tolist()
                    except Exception:  # noqa: BLE001
                        continue
                    series.append({"x": x_grid, "y": y_fit, "name": f"{fam} (R²={res.r_squared:.3f})"})
                overlay_fig = build_overlay_plot(series, x_label=x_label, y_label=y_label,
                                                   title="All candidate fits vs. data")
                st.plotly_chart(overlay_fig, width='stretch')
                format_download_button(
                    key="fit_overlay", file_stem="curve_fit_comparison",
                    render_fn=lambda fmt, s=series: snapshot_overlay_plot(
                        s, x_label=x_label, y_label=y_label, title="All candidate fits vs. data", fmt=fmt),
                )
    else:
        result = fit_curve(xs, ys, family, degree=degree, expr_str=expr_str, param_names=param_names)
        stat_family, stat_degree, stat_expr_str, stat_param_names = family, degree, expr_str, param_names
        if family == "linear":
            stat_degree = 1  # "linear" is fit_curve's degree-1 special case; keep the stat layer consistent

    if result.error:
        st.error(result.error)
        return

    st.latex(f"y = {sp.latex(result.expr)}")
    m1, m2 = st.columns(2)
    m1.metric("R²", f"{result.r_squared:.5f}")
    m2.metric("RMSE", f"{result.rmse:.5g}")

    log_cols = st.columns(2)
    with log_cols[0]:
        fit_x_log = st.checkbox("Log X-axis", key="fit_x_log")
    with log_cols[1]:
        fit_y_log = st.checkbox("Log Y-axis", key="fit_y_log")
    if fit_x_log or fit_y_log:
        st.caption("A power-law fit is a straight line on log-log axes; an exponential fit is a "
                    "straight line with only the Y-axis logged -- a quick visual sanity check that "
                    "the chosen family actually matches the data's shape.")

    fig = build_fit_plot(xs, ys, result.expr, x_label, y_label, x_log=fit_x_log, y_log=fit_y_log)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Residuals"):
        for x, y, r in zip(xs, ys, result.residuals):
            st.write(f"x={x:g}, y={y:g}, residual={r:.4g}")

    _render_statistical_layer(xs, ys, stat_family, stat_degree, stat_expr_str, stat_param_names)

    format_download_button(
        key="curve_fit", file_stem="curve_fit",
        render_fn=lambda fmt: snapshot_fit_plot(xs, ys, result.expr, x_label, y_label,
                                                   x_log=fit_x_log, y_log=fit_y_log, fmt=fmt),
    )
