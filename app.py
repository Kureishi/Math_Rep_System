"""
Math Representation System -- main Streamlit app.

Run with:
    streamlit run app.py

Requires LM Studio running locally with its server started
(Developer tab -> Start Server, default port 1234).
"""
import streamlit as st
import sympy as sp

from config import settings
from modules.llm_client import LMStudioClient
from modules.ocr import ocr_extract
from modules.equation_engine import extract_model, ProblemModel
from modules.verifier import verify, VerificationReport
from modules.solver import compute_steps, narrate_steps
from modules.scenarios import generate_alternative_scenarios
from modules.plotter import plottable_free_symbols, build_plot
from modules.workspace import Workspace

st.set_page_config(page_title="Math Representation System", layout="wide")

# ---------------------------------------------------------------- session
client = LMStudioClient()
ws = Workspace(st.session_state)
for key, default in [("problem_text", ""), ("model", None), ("report", None),
                      ("steps", None), ("scenarios", None), ("extracted_from_image", "")]:
    st.session_state.setdefault(key, default)

# ---------------------------------------------------------------- sidebar
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
    st.divider()
    st.header("Variable Workspace")
    if ws.entries:
        for name, entry in list(ws.entries.items()):
            c1, c2 = st.columns([3, 1])
            c1.write(f"**{name}** = {entry.value:.6g} {entry.unit or ''}  \n_{entry.source}_")
            if c2.button("✕", key=f"rm_{name}"):
                ws.remove(name)
                st.rerun()
    else:
        st.caption("No stored variables yet. Solve a problem and extract a value to reuse it here.")

st.title("🧮 Math Representation System")
st.caption("Text or image → derived equations → self-verified solution → alternative applications.")

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
    with st.spinner("Deriving equations from the problem statement..."):
        model = extract_model(client, problem_text)

    with st.spinner("Verifying the derivation..."):
        report = verify(model, client, problem_text)
        retries = 0
        while not report.passed and retries < settings.max_verification_retries:
            retries += 1
            with st.spinner(f"Verification failed -- retrying derivation ({retries}/{settings.max_verification_retries})..."):
                model = extract_model(client, problem_text, retry_reason=report.failure_reason)
                report = verify(model, client, problem_text)

    with st.spinner("Computing step-by-step solution..."):
        steps = compute_steps(model)
        steps = narrate_steps(client, model, steps)

    with st.spinner("Generating alternative scenarios..."):
        scenarios = generate_alternative_scenarios(client, model)

    st.session_state.update(model=model, report=report, steps=steps, scenarios=scenarios)

# ---------------------------------------------------------------- display
model: ProblemModel | None = st.session_state["model"]
report: VerificationReport | None = st.session_state["report"]

if model:
    st.divider()

    # ---- verification banner
    if report.passed:
        st.success("✅ Self-check passed: symbolic checks and an independent re-solve agree.")
    else:
        st.warning(
            "⚠️ Self-check found unresolved issues after retries -- review the equations below "
            "carefully before trusting the result."
        )
    with st.expander("Verification detail"):
        for c in report.checks:
            (st.write if c.passed else st.error)(f"{'✅' if c.passed else '❌'} **{c.label}**: {c.detail}")
        for target, val in report.sympy_numeric_answers.items():
            st.write(f"Derived-equation answer for `{target}`: `{val:.6g}`")
        for target, val in report.llm_independent_answers.items():
            st.write(f"Independent cross-check for `{target}`: `{val:.6g}`")

    st.subheader(f"Domain: {model.problem_domain}")

    # ---- equations + derivations
    st.markdown("### Derived equations")
    for eq in model.equations:
        cols = st.columns([2, 3])
        with cols[0]:
            if eq.sympy_eq is not None:
                st.latex(sp.latex(eq.sympy_eq))
            else:
                st.error(f"Failed to parse: {eq.raw_expression}")
        with cols[1]:
            st.markdown(f"**{eq.name}**")
            st.write(eq.derivation)

    if model.assumptions:
        st.markdown("**Assumptions made:**")
        for a in model.assumptions:
            st.write(f"- {a}")

    # ---- variables (editable, modification support)
    st.markdown("### Variables")
    edited_values = {}
    var_cols = st.columns(min(4, max(1, len(model.variables))))
    for i, v in enumerate(model.variables):
        with var_cols[i % len(var_cols)]:
            default = v.known_value if v.known_value is not None else 0.0
            edited_values[v.symbol] = st.number_input(
                f"{v.symbol} — {v.meaning} ({v.unit or 'unitless'})",
                value=float(default), key=f"var_{v.symbol}",
            )

    # ---- step-by-step solution (one section per requested target)
    steps_by_target = st.session_state["steps"] or {}
    if steps_by_target:
        st.markdown("### Step-by-step solution")
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
                             source=f"Solved from: {problem_text[:60]}...", unit=unit)
                    st.rerun()

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

    # ---- interactive plot
    plottable = [e for e in model.equations if e.sympy_eq is not None
                 and len(plottable_free_symbols(e, set())) >= 1]
    if plottable:
        st.markdown("### Interactive plot")
        eq_choice_name = st.selectbox("Equation to plot", [e.name for e in plottable])
        eq_choice = next(e for e in plottable if e.name == eq_choice_name)
        free_syms = plottable_free_symbols(eq_choice, set())

        x_symbol = st.selectbox("X-axis variable", free_syms)
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
                             (min(0.0, x_default - 10), x_default + 10))

        y_target = None
        if model.solve_for:
            candidates = [t for t in model.solve_for if t != x_symbol]
            if candidates:
                y_target = st.selectbox("Y-axis target (solve equation for)", candidates)

        fig = build_plot(model, eq_choice, x_symbol, param_values, x_range, y_target=y_target)
        st.plotly_chart(fig, use_container_width=True)
