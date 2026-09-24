"""
Word-problem solver: the default page. Text/image input, the Solve button, the extract -> verify ->
retry -> steps -> scenarios pipeline (modules/pipeline.py -- the same code batch mode runs), history
save, and then the results view for whatever problem is currently loaded (just solved, or restored
from history in the sidebar).
"""
import streamlit as st

from modules import history
from modules.llm_client import LMStudioClient, LLMOutputError
from modules.ocr import ocr_extract
from modules.pipeline import run_pipeline
from modules.workspace import Workspace
from ui.common import check_upload_size
from ui.results import render_results


def render_word_problem_page(client: LMStudioClient, ws: Workspace, ok: bool) -> None:
    """`ok` is whether the LM Studio server is reachable -- it gates the Solve button."""
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
        # ---- camera vs file: a phone user taking a picture of the problem
        # in front of them right now wants the camera directly, not a file
        # picker that then makes them choose "Camera" from an OS sheet --
        # st.camera_input returns the same UploadedFile-like object
        # st.file_uploader does (.getvalue(), .type), so everything
        # downstream (size check, vision/OCR extraction) works unchanged
        # regardless of which one was used.
        image_input_method = st.radio("Input method", ["📁 Upload a file", "📷 Take a photo"],
                                        horizontal=True, key="image_input_method",
                                        label_visibility="collapsed")
        if image_input_method == "📷 Take a photo":
            uploaded = st.camera_input("Take a photo of the problem")
        else:
            uploaded = st.file_uploader("Upload a photo or screenshot of the problem",
                                          type=["png", "jpg", "jpeg"])
        if uploaded is not None and check_upload_size(uploaded):
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

        try:
            # st.spinner is the `stage` hook: run_pipeline() wraps each stage in it, so the UI
            # shows the same "Deriving... / Verifying... / Retrying (n/N)... / Computing... /
            # Generating..." progress messages without the pipeline knowing Streamlit exists.
            result = run_pipeline(client, problem_text, known_context=known_context, stage=st.spinner)
            model, report, steps, scenarios = result.model, result.report, result.steps, result.scenarios

            st.session_state.update(model=model, report=report, steps=steps, scenarios=scenarios,
                                     pdf_bytes=None, plot_snapshots={})
            saved_id = history.save(problem_text, model, report, steps, scenarios)
            st.session_state["last_saved_history_id"] = saved_id
            st.toast("Saved to history", icon="💾")

        except LLMOutputError as e:
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
    model = st.session_state["model"]
    report = st.session_state["report"]
    if model:
        render_results(client, ws, model, report, problem_text)
