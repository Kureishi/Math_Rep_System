"""
The Practice tab of a solved problem: grade-my-work and worksheet generation.
"""
import streamlit as st
from modules.llm_client import LMStudioClient
from modules.ocr import ocr_extract
from modules.equation_engine import ProblemModel, target_kind
from modules.verifier import VerificationReport
from modules.grading import grade_work, classify_mistake
from modules.worksheet import generate_worksheet_problems, generate_targeted_worksheet_problems
from modules import history
from modules.similarity import problem_shape
from ui.common import check_upload_size


def render_practice_tab(client: LMStudioClient, model: ProblemModel, report: VerificationReport, tab_practice):
    """The Practice tab: grade-my-work, tutor-style exercises and worksheet generation."""
    # ---- grade my work: compare a student's own attempted steps
    # against the verified derivation (formula check, arithmetic check,
    # final-answer check -- see grading.py for why this is a diagnosis,
    # not a literal line-by-line diff)
    with tab_practice:
        algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
        if algebraic_targets:
            with st.expander("📝 Grade my work"):
                grade_target = st.selectbox("Which target are you solving for?", algebraic_targets,
                                              key="grade_target")

                # ---- handwritten work via photo: reuses the same
                # vision/OCR machinery as the problem-statement image
                # tab, just with a prompt tailored to transcribing
                # WORKED STEPS instead of a problem -- removes the
                # single biggest piece of friction in actually using
                # this feature: retyping work that's already on paper.
                with st.expander("📷 Or upload a photo of your handwritten work"):
                    work_input_method = st.radio(
                        "Input method", ["📁 Upload a file", "📷 Take a photo"], horizontal=True,
                        key="grade_work_input_method", label_visibility="collapsed",
                    )
                    if work_input_method == "📷 Take a photo":
                        uploaded_work = st.camera_input("Take a photo of your work",
                                                          key="grade_work_camera")
                    else:
                        uploaded_work = st.file_uploader("Upload a photo", type=["png", "jpg", "jpeg"],
                                                           key="grade_work_photo")
                    if uploaded_work is not None and check_upload_size(uploaded_work):
                        st.image(uploaded_work, caption="Uploaded work", width=300)
                        use_vision_grading = st.toggle(
                            "Use LM Studio vision model (falls back to Tesseract OCR -- much less "
                            "reliable on handwriting -- if off/unavailable)",
                            value=True, key="grade_work_use_vision",
                        )
                        if st.button("Extract work from photo", key="grade_work_extract_button"):
                            with st.spinner("Reading handwriting..."):
                                try:
                                    if use_vision_grading:
                                        extracted = client.vision_extract_work(
                                            uploaded_work.getvalue(), mime_type=uploaded_work.type)
                                    else:
                                        extracted = ocr_extract(uploaded_work.getvalue())
                                except Exception as e:  # noqa: BLE001
                                    st.error(f"Extraction failed: {e}")
                                    extracted = None
                            if extracted is not None:
                                st.session_state["grade_work_text"] = extracted
                                st.rerun()

                student_work = st.text_area(
                    "Paste your work, one step per line (or extract from a photo above)",
                    placeholder="a = (v_f - v_i) / t\na = (20 - 8) / 6\na = 2.0",
                    height=120, key="grade_work_text",
                )
                if st.button("Grade it", key="grade_button") and student_work.strip():
                    correct_val = report.sympy_numeric_answers.get(grade_target)
                    result = grade_work(model, grade_target, student_work.splitlines(), correct_val)
                    if result.error:
                        st.error(result.error)
                    else:
                        (st.success if result.final_answer_ok else
                         st.warning if result.final_answer_ok is None else st.error)(result.summary)
                        if result.formula_ok is not None:
                            st.caption(f"Formula check: {result.formula_detail}")
                        for i, lr in enumerate(result.line_results, start=1):
                            icon = "✅" if lr.arithmetic_ok else "❌" if lr.arithmetic_ok is False else "➖"
                            st.write(f"{icon} Line {i}: `{lr.raw}` -- {lr.detail}")

                        # personalized error-pattern tracking: persist this
                        # submission's classification alongside history.py's
                        # existing records, so a recurring mistake (a sign
                        # error, a wrong-formula habit, ...) can surface as
                        # an actual pattern over time rather than vanishing
                        # once this expander closes
                        classification = classify_mistake(result)
                        history.record_grading(
                            target=grade_target, domain=model.problem_domain,
                            category=classification.category, subtype=classification.subtype,
                            detail=classification.detail, equation_shapes=problem_shape(model),
                        )

                error_patterns = history.summarize_error_patterns()
                if error_patterns:
                    st.divider()
                    st.caption("📊 Patterns from your recent graded work:")
                    for p in error_patterns:
                        st.warning(p["message"])
                    st.session_state["error_pattern_messages"] = [p["message"] for p in error_patterns]

        # ---- worksheet generator: reverse generation -- new problem TEXT
        # sharing this problem's own verified equation structure, fed back
        # through the SAME extract/verify pipeline used for any problem
        # rather than trusting the generating LLM's own numbers
        with st.expander("📄 Generate worksheet variants"):
            wcols = st.columns([2, 1, 1])
            with wcols[0]:
                w_count = st.slider("How many?", 1, 5, 3, key="worksheet_count")
            with wcols[1]:
                w_difficulty = st.selectbox("Difficulty", ["similar", "easier", "harder"], key="worksheet_difficulty")
            recent_patterns = st.session_state.get("error_pattern_messages", [])
            w_targeted = False
            if recent_patterns:
                w_targeted = st.checkbox(
                    "🎯 Target my recent mistake pattern(s)", value=True, key="worksheet_targeted",
                    help="Biases the generated problems toward extra practice on: " + "; ".join(recent_patterns),
                )
            if st.button("Generate", key="worksheet_button"):
                with st.spinner("Generating worksheet problems..."):
                    if w_targeted and recent_patterns:
                        new_problems = generate_targeted_worksheet_problems(
                            client, model, recent_patterns, w_count, w_difficulty)
                    else:
                        new_problems = generate_worksheet_problems(client, model, w_count, w_difficulty)
                if not new_problems:
                    st.warning("Couldn't generate worksheet problems -- try again.")
                else:
                    st.session_state["worksheet_problems"] = new_problems
            for i, p in enumerate(st.session_state.get("worksheet_problems", []), start=1):
                st.markdown(f"**{i}.** {p}")
            if st.session_state.get("worksheet_problems"):
                st.caption("Copy any of these into the main text input above to solve and verify it "
                            "through the same pipeline as any other problem.")
