"""
Math Representation System -- main Streamlit app.

Run with:
    streamlit run app.py

Requires LM Studio running locally with its server started
(Developer tab -> Start Server, default port 1234).
"""
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go

from config import settings
from modules.llm_client import LMStudioClient, LLMOutputError
from modules.ocr import ocr_extract
from modules.equation_engine import extract_model, ProblemModel, target_kind, symbols_and_functions_used
from sympy.core.function import AppliedUndef
from modules.verifier import verify, VerificationReport, confidence_label, _known_substitutions
from modules.solver import compute_steps, narrate_steps
from modules.ode_utils import solve_ode
from modules.recurrence_utils import solve_recurrence, _independent_variable
from modules.optimization_utils import solve_optimization
from modules.matrix_utils import linear_system_view
from modules.unit_conversion import sweep_conversions
from modules.code_export import formula_for_target, generate_python_function, generate_python_module
from modules.vector_utils import vector_summary
from modules.scenarios import generate_alternative_scenarios
from modules.plotter import plottable_free_symbols, build_plot, build_surface_plot, build_feasible_region_plot, build_vector_plot, build_fit_plot
from modules.plot_snapshot import snapshot_line_plot, snapshot_surface_plot, snapshot_feasible_region, snapshot_ode_plot, snapshot_recurrence_plot, snapshot_vector_plot, snapshot_fit_plot
from modules.curve_fitting import fit_curve, best_fit, parse_xy_csv, BUILTIN_FAMILIES
from modules.equivalence import check_equivalence
from modules.workspace import Workspace
from modules import history
from modules.exporter import build_markdown, build_pdf_bytes, PlotSnapshot

st.set_page_config(page_title="Math Representation System", layout="wide")

# ---------------------------------------------------------------- session
client = LMStudioClient()
ws = Workspace(st.session_state)
for key, default in [("problem_text", ""), ("model", None), ("report", None),
                      ("steps", None), ("scenarios", None), ("extracted_from_image", ""),
                      ("pdf_bytes", None), ("plot_snapshots", {})]:
    st.session_state.setdefault(key, default)


def snapshot_button(key: str, title: str, caption: str, render_fn):
    """Renders a small 'include this plot in the report' control under a
    plot. render_fn is a zero-arg callable producing PNG bytes -- kept
    lazy so the (potentially slow) matplotlib re-render only happens when
    the user actually opts in, not on every script rerun."""
    existing = st.session_state["plot_snapshots"].get(key)
    if existing:
        c1, c2 = st.columns([4, 1])
        with c1:
            st.caption(f"✅ Included in the exported report as \"{existing.title}\"")
        with c2:
            if st.button("Remove", key=f"remove_snap_{key}"):
                del st.session_state["plot_snapshots"][key]
                st.session_state["pdf_bytes"] = None  # cached PDF is now stale
                st.rerun()
    else:
        if st.button("📸 Include this plot in the report", key=f"include_snap_{key}"):
            try:
                png = render_fn()
                st.session_state["plot_snapshots"][key] = PlotSnapshot(title=title, caption=caption, png_bytes=png)
                st.session_state["pdf_bytes"] = None
                st.rerun()
            except Exception as e:  # noqa: BLE001
                st.error(f"Couldn't capture a snapshot of this plot: {e}")

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
        if uploaded_csv is not None:
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
    )

    degree = 2
    expr_str, param_names = None, None
    if family == "polynomial":
        degree = st.number_input("Degree", min_value=2, max_value=10, value=2, step=1)
    elif family == "custom":
        expr_str = st.text_input("Model expression in x and named parameters",
                                   placeholder="a*sin(x) + b*x + c")
        params_raw = st.text_input("Parameter names (comma-separated)", placeholder="a, b, c")
        param_names = [p.strip() for p in params_raw.split(",") if p.strip()]

    if not st.button("Fit", type="primary"):
        return

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
    else:
        result = fit_curve(xs, ys, family, degree=degree, expr_str=expr_str, param_names=param_names)

    if result.error:
        st.error(result.error)
        return

    st.latex(f"y = {sp.latex(result.expr)}")
    m1, m2 = st.columns(2)
    m1.metric("R²", f"{result.r_squared:.5f}")
    m2.metric("RMSE", f"{result.rmse:.5g}")

    fig = build_fit_plot(xs, ys, result.expr, x_label, y_label)
    st.plotly_chart(fig, width='stretch')

    with st.expander("Residuals"):
        for x, y, r in zip(xs, ys, result.residuals):
            st.write(f"x={x:g}, y={y:g}, residual={r:.4g}")

    png = snapshot_fit_plot(xs, ys, result.expr, x_label, y_label)
    st.download_button("Download plot as PNG", data=png, file_name="curve_fit.png", mime="image/png")


def render_equivalence_tab():
    """Standalone building block: 'are these two expressions the same',
    not a representation capability of its own."""
    st.subheader("🔁 Check equivalence")
    st.caption("See whether two symbolic expressions are the same, and if not, why.")

    c1, c2 = st.columns(2)
    with c1:
        expr1_str = st.text_input("Expression 1", placeholder="sin(x)**2 + cos(x)**2")
    with c2:
        expr2_str = st.text_input("Expression 2", placeholder="1")
    extra_syms_raw = st.text_input("Extra symbol names (comma-separated, optional)", placeholder="a, b")

    if not st.button("Check", type="primary"):
        return
    if not expr1_str.strip() or not expr2_str.strip():
        st.warning("Enter both expressions.")
        return

    extra_symbols = [s.strip() for s in extra_syms_raw.split(",") if s.strip()]
    result = check_equivalence(expr1_str, expr2_str, extra_symbols)

    if result.error:
        st.error(result.error)
        return

    if result.equivalent is True:
        st.success("✅ Equivalent")
    elif result.equivalent is False:
        st.error("❌ Not equivalent")
    else:
        st.warning("⚠️ Undetermined")

    st.caption(f"Method: {result.method}")
    st.write(result.detail)
    if result.difference_simplified is not None:
        st.latex(r"\text{difference (simplified)} = " + sp.latex(result.difference_simplified))


with st.sidebar:
    st.header("LM Studio")
    ok, msg = client.is_available()
    (st.success if ok else st.error)(msg)

    loaded_models = client.list_models() if ok else []
    if ok and not loaded_models:
        st.warning("Connected, but no models are loaded. Load one in LM Studio's Developer tab.")
    elif loaded_models:
        # Fall back to whatever's actually loaded if config.py's default
        # isn't among the currently-served models, instead of silently
        # trying to call a model that doesn't exist.
        default_reasoning = settings.reasoning_model if settings.reasoning_model in loaded_models else loaded_models[0]
        settings.reasoning_model = st.selectbox(
            "Reasoning model", loaded_models,
            index=loaded_models.index(default_reasoning),
            help="Used for equation extraction, verification cross-checks, narration, and scenarios.",
        )

        default_vision = settings.vision_model if settings.vision_model in loaded_models else loaded_models[0]
        settings.vision_model = st.selectbox(
            "Vision model", loaded_models,
            index=loaded_models.index(default_vision),
            help="Used to transcribe problem statements from uploaded images. Pick a multimodal "
                 "model here -- a text-only model will error on image input; use OCR fallback instead.",
        )

    with st.expander("⚙️ Advanced settings"):
        st.caption("Tune verification strictness and generation behavior without editing config.py "
                    "or restarting the app. Changes apply to the next problem you solve.")

        settings.temperature_extraction = st.slider(
            "Extraction temperature", 0.0, 1.0, settings.temperature_extraction, 0.05,
            help="How much freedom the model has when converting text into equations. Lower is more "
                 "deterministic and faithful to the problem; raise it only if extraction feels too rigid.",
        )
        settings.temperature_narration = st.slider(
            "Narration temperature", 0.0, 1.0, settings.temperature_narration, 0.05,
            help="Controls the wording of step-by-step explanations only -- never affects the math "
                 "itself, since SymPy computes that independently.",
        )
        settings.max_verification_retries = st.number_input(
            "Max verification retries", min_value=0, max_value=5,
            value=settings.max_verification_retries, step=1,
            help="How many times to re-prompt the model with the failure reason if self-verification "
                 "fails, before giving up and showing the result with a warning.",
        )
        settings.numeric_tolerance = st.select_slider(
            "Numeric balance tolerance",
            options=[1e-9, 1e-6, 1e-3, 1e-2],
            value=settings.numeric_tolerance if settings.numeric_tolerance in (1e-9, 1e-6, 1e-3, 1e-2) else 1e-6,
            format_func=lambda x: f"{x:.0e}",
            help="How close a residual must be to zero to count as 'balances' when checking an "
                 "equation against known values. Tighter (smaller) catches more subtle errors but "
                 "may flag harmless floating-point rounding as a failure.",
        )
        cross_check_pct = st.slider(
            "Independent cross-check tolerance", 0.5, 10.0, settings.cross_check_tolerance * 100, 0.5,
            format="%.1f%%",
            help="How far apart the derived answer and the independent re-solve can be before "
                 "verification flags a disagreement. Wider tolerates more model imprecision; "
                 "narrower catches subtler derivation errors but may false-flag on rounding.",
        )
        settings.cross_check_tolerance = cross_check_pct / 100

        if st.button("Reset to defaults"):
            from config import Settings
            defaults = Settings()
            settings.temperature_extraction = defaults.temperature_extraction
            settings.temperature_narration = defaults.temperature_narration
            settings.max_verification_retries = defaults.max_verification_retries
            settings.numeric_tolerance = defaults.numeric_tolerance
            settings.cross_check_tolerance = defaults.cross_check_tolerance
            st.rerun()

    st.divider()
    st.header("Variable Workspace")
    if ws.entries:
        for name, entry in list(ws.entries.items()):
            c1, c2 = st.columns([3, 1])
            with c1:
                new_name = st.text_input(
                    "Name", value=name, key=f"wsname_{name}", label_visibility="collapsed",
                )
                st.caption(f"= {entry.value:.6g} {entry.unit or ''}  \n_{entry.source}_")
            with c2:
                if st.button("✕", key=f"rm_{name}"):
                    ws.remove(name)
                    st.rerun()
            if new_name != name:
                ok_rename, err = ws.rename(name, new_name)
                if ok_rename:
                    st.rerun()
                else:
                    st.warning(err)
        st.caption("Rename a variable above to reuse it under a new name, or reference it by its "
                    "current name in a new problem below (e.g. \"using d = ...\").")
    else:
        st.caption("No stored variables yet. Solve a problem and extract a value to reuse it here.")

    st.divider()
    st.header("History")
    recent = history.list_recent()
    if recent:
        for entry in recent:
            badge = "✅" if entry["passed"] else "⚠️"
            label = entry["problem_text"][:45] + ("..." if len(entry["problem_text"]) > 45 else "")
            c1, c2, c3 = st.columns([5, 1, 1])
            with c1:
                st.caption(f"{badge} **{entry['domain'] or '—'}** -- {entry['timestamp'][:16].replace('T', ' ')}")
                st.caption(label)
            with c2:
                if st.button("↺", key=f"load_{entry['id']}", help="Load this problem"):
                    loaded = history.load(entry["id"])
                    if loaded is not None:
                        p_text, l_model, l_report, l_steps, l_scenarios = loaded
                        st.session_state.update(
                            problem_text=p_text, model=l_model, report=l_report,
                            steps=l_steps, scenarios=l_scenarios, pdf_bytes=None, plot_snapshots={},
                        )
                        st.rerun()
            with c3:
                if st.button("✕", key=f"delhist_{entry['id']}", help="Delete from history"):
                    history.delete(entry["id"])
                    st.rerun()
    else:
        st.caption("No solved problems yet -- they'll be saved here automatically.")

st.title("🧮 Math Representation System")
st.caption("Text or image → derived equations → self-verified solution → alternative applications.")

# ---------------------------------------------------------------- mode selector
# Curve fitting and equivalence checking are sibling/standalone tools --
# gated with an early st.stop() rather than wrapping the (large,
# deeply-indented) word-problem pipeline below in its own tab block, so
# adding them doesn't require touching that pipeline's indentation at all.
mode = st.radio("Mode", ["📝 Word problem solver", "📈 Curve fitting", "🔁 Check equivalence"],
                  horizontal=True, label_visibility="collapsed")

if mode == "📈 Curve fitting":
    render_curve_fitting_tab()
    st.stop()
elif mode == "🔁 Check equivalence":
    render_equivalence_tab()
    st.stop()

# ---------------------------------------------------------------- input
tab_text, tab_image = st.tabs(["Text input", "Image input"])
problem_text = ""

with tab_text:
    problem_text = st.text_area(
        "Describe the problem or scenario",
        value=st.session_state["problem_text"],
        height=140,
        placeholder="e.g. A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. "
                    "What is its acceleration, and how far does it travel in that time?",
    )

with tab_image:
    uploaded = st.file_uploader("Upload a photo or screenshot of the problem", type=["png", "jpg", "jpeg"])
    if uploaded is not None:
        st.image(uploaded, caption="Uploaded image", width=400)
        use_vision = st.toggle("Use LM Studio vision model (falls back to Tesseract OCR if off/unavailable)",
                                value=True)
        if st.button("Extract text from image"):
            with st.spinner("Reading image..."):
                try:
                    if use_vision:
                        text = client.vision_extract(uploaded.getvalue(), mime_type=uploaded.type)
                    else:
                        text = ocr_extract(uploaded.getvalue())
                except Exception as e:  # noqa: BLE001
                    st.error(f"Extraction failed: {e}")
                    text = ""
            st.session_state["extracted_from_image"] = text

    if st.session_state["extracted_from_image"]:
        st.text_area("Extracted text preview (editable before solving)",
                      key="extracted_from_image", height=120)
        problem_text = st.session_state["extracted_from_image"]

solve_clicked = st.button("🔎 Represent & Solve", type="primary", disabled=not ok)

# ---------------------------------------------------------------- pipeline
if solve_clicked and problem_text.strip():
    st.session_state["problem_text"] = problem_text
    known_context = ws.as_context_string()
    pipeline_failed = False

    try:
        with st.spinner("Deriving equations from the problem statement..."):
            model = extract_model(client, problem_text, known_context=known_context)

        with st.spinner("Verifying the derivation..."):
            report = verify(model, client, problem_text)
            retries = 0
            while not report.passed and retries < settings.max_verification_retries:
                retries += 1
                with st.spinner(f"Verification failed -- retrying derivation ({retries}/{settings.max_verification_retries})..."):
                    model = extract_model(client, problem_text, retry_reason=report.failure_reason,
                                           known_context=known_context)
                    report = verify(model, client, problem_text)

        with st.spinner("Computing step-by-step solution..."):
            steps = compute_steps(model)
            steps = narrate_steps(client, model, steps)

        with st.spinner("Generating alternative scenarios..."):
            scenarios = generate_alternative_scenarios(client, model)

        st.session_state.update(model=model, report=report, steps=steps, scenarios=scenarios,
                                 pdf_bytes=None, plot_snapshots={})
        history.save(problem_text, model, report, steps, scenarios)

    except LLMOutputError as e:
        pipeline_failed = True
        st.error(f"⚠️ {e}")
        with st.expander("Raw model response (for debugging)"):
            st.code(e.raw_output or "(empty response)")
        st.info(
            "This usually means the model asked a clarifying question, refused, or got confused "
            "instead of returning structured JSON -- often because a referenced quantity (like a "
            "workspace variable) wasn't clearly matched. Try rephrasing, or use a stronger reasoning "
            "model for extraction."
        )

# ---------------------------------------------------------------- display
model: ProblemModel | None = st.session_state["model"]
report: VerificationReport | None = st.session_state["report"]

if model:
    st.divider()

    # ---- confidence report: an aggregated, category-grouped view over
    # the raw check list, rather than making someone scan every check to
    # get a sense of "how much should I trust this"
    cr = report.confidence_report()
    conf_label, worst_ratio = report.confidence()

    banner_cols = st.columns([1, 3])
    with banner_cols[0]:
        st.metric("Confidence", f"{cr.score:.0%}", help="1.0 = every check passed with an "
                   "essentially-exact margin. Capped below 50% if any check failed outright, "
                   "regardless of how many others passed.")
    with banner_cols[1]:
        if report.passed:
            if conf_label in ("essentially exact", "comfortable margin"):
                st.success(f"✅ Self-check passed with high confidence ({conf_label}) -- symbolic "
                            "checks and an independent re-solve agree.")
            else:
                st.warning(f"✅ Self-check passed, but confidence is only '{conf_label}' -- at least "
                            "one check came close to its tolerance. Worth a second look before "
                            "trusting the result completely.")
        else:
            st.warning(
                "⚠️ Self-check found unresolved issues after retries -- review the equations below "
                "carefully before trusting the result."
            )

    cat_cols = st.columns(min(len(cr.categories), 5) or 1)
    for i, (cat, summary) in enumerate(cr.categories.items()):
        with cat_cols[i % len(cat_cols)]:
            icon = "✅" if summary.all_passed else "❌"
            st.caption(f"{icon} {cat}")
            st.write(f"{summary.passed}/{summary.total}")

    if cr.critical_failures:
        st.error("**Critical checks that failed:** " +
                  "; ".join(f"{c.label} -- {c.detail}" for c in cr.critical_failures))

    with st.expander("Verification detail (raw check list)"):
        for c in report.checks:
            icon = "✅" if c.passed else "❌"
            margin_tag = ""
            if c.passed and c.margin_ratio is not None:
                margin_tag = f"  `confidence: {confidence_label(c.margin_ratio)}`"
            (st.write if c.passed else st.error)(f"{icon} **{c.label}**: {c.detail}{margin_tag}")
        for target, val in report.sympy_numeric_answers.items():
            st.write(f"Derived-equation answer for `{target}`: `{val:.6g}`")
        for target, val in report.llm_independent_answers.items():
            st.write(f"Independent cross-check for `{target}`: `{val:.6g}`")

    # ---- domain of validity: when does each formula's derived relation
    # actually make sense (never divide by zero, sqrt of a negative, log
    # of a non-positive value, etc.) -- shown as its own panel even for
    # restrictions that AREN'T currently violated, since knowing the
    # boundary of a formula's validity is useful on its own
    if report.domain_notes:
        with st.expander("Domain of validity", expanded=any(n.violated for n in report.domain_notes)):
            for note in report.domain_notes:
                if note.violated:
                    st.error(f"**{note.equation}** -- undefined with the given values: " +
                              "; ".join(r.description for r in note.violated))
                restrictions_ok = note.satisfied + note.pending
                if restrictions_ok:
                    descs = "; ".join(r.description for r in restrictions_ok)
                    st.write(f"**{note.equation}** requires: {descs}")

    st.subheader(f"Domain: {model.problem_domain}")

    # ---- equations + derivations
    st.markdown("### Derived equations")
    KIND_BADGES = {"equation": "🟢 equation", "inequality": "🟡 inequality",
                    "ode": "🔵 differential equation", "recurrence": "🟣 recurrence relation"}
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

    if model.assumptions:
        st.markdown("**Assumptions made:**")
        for a in model.assumptions:
            st.write(f"- {a}")

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

    # ---- step-by-step solution (one section per requested target)
    steps_by_target = st.session_state["steps"] or {}
    if steps_by_target:
        st.markdown("### Step-by-step solution")

        module_src = generate_python_module(model)
        if not module_src.startswith('"""No exportable'):
            st.download_button(
                "⬇️ Get all formulas as one Python file", data=module_src,
                file_name="formulas.py", mime="text/x-python",
            )

        for target_name, steps in steps_by_target.items():
            st.markdown(f"#### Solving for `{target_name}`")
            for i, step in enumerate(steps, start=1):
                st.markdown(f"**Step {i}: {step.description}**")
                st.latex(step.expression)
                if step.explanation:
                    st.caption(step.explanation)

            sympy_val = report.sympy_numeric_answers.get(target_name)
            if sympy_val is not None:
                if st.button(f"➕ Extract {target_name} to workspace", key=f"extract_{target_name}"):
                    unit = next((v.unit for v in model.variables if v.symbol == target_name), None)
                    ws.store(target_name, sympy_val,
                             source=f"{problem_text[:60]}...", unit=unit)
                    st.rerun()

                # ---- unit conversion sweep: offer the same numeric
                # answer in a handful of common alternate units, once its
                # declared unit is known -- purely a display convenience,
                # doesn't touch the verified value itself
                target_unit = next((v.unit for v in model.variables if v.symbol == target_name), None)
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

    # ---- interactive plot (algebraic equations only -- inequalities and
    # ODEs are visualized in their own dedicated sections above)
    plottable = [e for e in model.equations if e.kind == "equation" and e.sympy_eq is not None
                 and len(plottable_free_symbols(e, set())) >= 1]
    if plottable:
        st.markdown("### Interactive plot")
        eq_choice_name = st.selectbox("Equation to plot", [e.name for e in plottable])
        eq_choice = next(e for e in plottable if e.name == eq_choice_name)
        free_syms = plottable_free_symbols(eq_choice, set())

        plot_mode = "2D line"
        if len(free_syms) >= 2:
            plot_mode = st.radio("Plot type", ["2D line", "3D surface"], horizontal=True)

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

            fig = build_plot(model, eq_choice, x_symbol, param_values, x_range, y_target=y_target)
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
                render_fn=lambda ec=eq_choice, xs=x_symbol, pv=param_values, xr=x_range, yt=y_target:
                    snapshot_line_plot(ec, xs, pv, xr, y_target=yt),
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
