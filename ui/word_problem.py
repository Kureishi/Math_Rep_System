"""
Word-problem solver: the default page. Text/image input, the Solve button, the extract -> verify ->
retry -> steps -> scenarios pipeline (modules/pipeline.py -- the same code batch mode runs), history
save, and then the results view for whatever problem is currently loaded (just solved, or restored
from history in the sidebar).
"""
from contextlib import contextmanager

import streamlit as st

from modules import history
from modules.app_logging import logger
from modules.llm_client import LMStudioClient
from modules.workspace import Workspace
from modules import provenance
from ui import view_state
from ui.common import check_upload_size
from ui.routing import render_mode_suggestion
from ui.safe import run_panel

# modules.pipeline (-> solver, verifier, scipy, ...), modules.ocr (-> PIL) and ui.results (-> pandas,
# plotly, matplotlib, ...) are imported inside the functions that use them rather than here: a fresh
# session that has not solved anything yet needs none of them, and importing them up front delays the
# first paint of the page by several seconds.


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
        run_panel("Mode suggestion", render_mode_suggestion, problem_text)        # a banner only when the text clearly belongs to another mode

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
                            from modules.ocr import ocr_extract
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
        from modules.pipeline import run_pipeline
        st.session_state["problem_text"] = problem_text
        known_context = ws.as_context_string()

        st.session_state["_pipeline_warnings"] = []
        saved_ok = False
        solve_log: list[str] = []
        try:
            # One persistent status panel for the whole pipeline, instead of a
            # sequence of st.spinner()s that each vanish once their step ends --
            # so the person sees the full procedure (derive -> verify -> retry
            # if needed -> compute -> narrate -> scenarios) as a running log
            # with a checkmark per completed stage, the way a CI pipeline or a
            # build log reads, rather than losing earlier steps as later ones
            # start. `stage` (passed to run_pipeline as its `stage` hook) is a
            # tiny context manager that updates the status label on entry and
            # appends a checkmark line on exit -- it doesn't know Streamlit
            # beyond that, matching pipeline.py's StageFn contract exactly.
            with st.status("Solving problem...", expanded=True) as status:
                @contextmanager
                def stage(label: str):
                    status.update(label=label)
                    yield
                    st.write(f"✅ {label}")
                    solve_log.append(label)

                result = run_pipeline(client, problem_text, known_context=known_context, workspace_entries=dict(ws.entries), stage=stage)
                status.update(label="Solved" if not result.warnings else "Solved (with warnings)", state="complete", expanded=False)

            model, report, steps, scenarios = result.model, result.report, result.steps, result.scenarios

            # The derivation is the valuable part: put it on screen FIRST, then do the housekeeping (history, provenance) where a failure can only
            # cost that housekeeping, never the result.
            st.session_state.update(model=model, report=report, steps=steps, scenarios=scenarios, pdf_bytes=None, plot_snapshots={},
                                     _pipeline_warnings=list(result.warnings))
            try:
                saved_id = history.save(problem_text, model, report, steps, scenarios)
                st.session_state["last_saved_history_id"] = saved_id
                view_state.start_fresh(saved_id, provenance.capture())     # default settings; record how it was solved
                saved_ok = True
                st.session_state["_after_solve"] = {"label": "Solved" if not result.warnings else "Solved (with warnings)", "lines": solve_log}
            except Exception as exc:  # noqa: BLE001
                logger.exception("could not save the solved problem to history")
                st.session_state["last_saved_history_id"] = None
                st.session_state[view_state.CURRENT_ID_KEY] = None      # so the status line does not name the PREVIOUS problem's record
                st.session_state["_pipeline_warnings"].append(
                    f"The result could not be saved to history ({type(exc).__name__}: {str(exc)[:140]}). It is shown below, but it will not be there after you leave this page.")

        except Exception as e:  # noqa: BLE001 -- every way the solve can fail is explained, never shown as a raw traceback
            from modules.llm_diagnosis import diagnose_exception
            from ui.diagnosis import render_diagnosis
            logger.exception("solve failed")
            try:
                status.update(label="Could not solve the problem", state="error", expanded=False)
            except Exception:  # noqa: BLE001 -- the status box may not exist if the failure came before it opened
                pass
            render_diagnosis(diagnose_exception(e), key="solve_failure")

        if saved_ok:
            # The sidebar was drawn at the START of this run, before the solve saved anything, so its History list is one entry behind. One rerun redraws it with the
            # new entry (the result itself is in session_state and renders as before); the "Solved" status box and the toast are re-shown once, below.
            st.rerun()

    # ---------------------------------------------------------------- display
    after = st.session_state.pop("_after_solve", None)
    if after:
        with st.status(after["label"], state="complete", expanded=False):
            for line in after["lines"]:
                st.write(f"✅ {line}")
        st.toast("Saved to history", icon="💾")
    model = st.session_state["model"]
    report = st.session_state["report"]
    if model:
        _render_status_line()
        for w in st.session_state.get("_pipeline_warnings") or []:
            st.warning(w, icon="⚠️")
        try:
            from ui.results import render_results
        except ImportError as exc:        # a missing package must be a clear instruction, not a traceback over the whole page
            from ui.safe import show_failure
            show_failure("The results view", exc, "The solved equations are kept; install the package and rerun the app.")
        else:
            render_results(client, ws, model, report, problem_text)


def _render_status_line() -> None:
    """A small line that stays while a problem is on screen. The "Solved" status box above it is a one-off (it disappears on the next interaction, which is how
    Streamlit's rerun model works), so this is what is left to say that the result is safe: where it was saved, and under which number."""
    history_id = st.session_state.get(view_state.CURRENT_ID_KEY)
    solved_here = history_id is not None and history_id == st.session_state.get("last_saved_history_id")
    warned = bool(st.session_state.get("_pipeline_warnings")) and solved_here
    try:
        still_saved = history_id is None or history.exists(history_id)
    except Exception:  # noqa: BLE001 -- a locked database must not hide the result
        still_saved = True
    text = history.status_line(history_id, solved_here, warned, still_saved)
    icon = "⚠️" if warned or history_id is None or not still_saved else ("✅" if solved_here else "🕘")
    st.caption(f"{icon} {text}")
