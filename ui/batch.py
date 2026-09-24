"""
Batch solver mode: a whole pasted/PDF problem set solved in one pass, with a combined report.
"""
import io
import streamlit as st
import pandas as pd
from modules.llm_client import LMStudioClient
from modules.batch_solver import (
    solve_batch,
    batch_summary,
    batch_results_table,
    split_batch_text,
    extract_text_from_pdf,
)
from modules.exporter import build_batch_markdown, build_batch_pdf_bytes
from ui.common import check_upload_size


def render_batch_solver_tab(client: LMStudioClient):
    """Worksheet/batch mode: solve a whole problem set in one pass --
    the kind of thing a local tool with file access can do naturally
    that a per-query web calculator can't. See batch_solver.py."""
    st.subheader("📚 Batch solver")
    st.caption("Solve a whole problem set at once and get one combined report.")

    pdf_upload = st.file_uploader("...or upload a PDF worksheet (numbered problems auto-detected)",
                                    type=["pdf"], key="batch_pdf_upload")
    pdf_extracted_text = ""
    if pdf_upload is not None and check_upload_size(pdf_upload):
        pdf_extracted_text = extract_text_from_pdf(pdf_upload.getvalue())
        if not pdf_extracted_text:
            st.warning("Couldn't extract any text from that PDF -- it may be a scanned/image-only "
                        "worksheet with no embedded text layer (OCR isn't supported here).")

    batch_text = st.text_area(
        "Paste multiple problems -- separate with a blank line, or a line containing just ---",
        value=pdf_extracted_text,
        height=200,
        placeholder="A car accelerates from 8 m/s to 20 m/s over 6 seconds. Find acceleration.\n\n"
                     "Two numbers x and y satisfy x + y = 12 and 3x - y = 8. Find both numbers.",
        key="batch_text_input",
    )
    narrate = st.checkbox("Include step narration (slower -- one extra LLM call per problem)",
                            key="batch_narrate")

    if st.button("Solve batch", type="primary", key="batch_solve_button"):
        problems = split_batch_text(batch_text)
        if not problems:
            st.warning("No problems detected -- separate them with a blank line or a '---' line.")
            return
        progress = st.progress(0.0, text=f"Solving 0/{len(problems)}...")

        def _update(done, total):
            progress.progress(done / total, text=f"Solving {done}/{total}...")

        results = solve_batch(client, problems, narrate=narrate, progress_callback=_update)
        st.session_state["batch_results"] = results
        progress.empty()

    results = st.session_state.get("batch_results")
    if not results:
        return

    summary = batch_summary(results)
    st.success(f"{summary['solved']}/{summary['total']} solved "
                f"({summary['needed_retry']} needed a retry, {summary['failed']} failed)")

    md = build_batch_markdown(results)
    dl_cols = st.columns(2)
    with dl_cols[0]:
        st.download_button("⬇️ Download combined Markdown report", data=md,
                             file_name="batch_report.md", mime="text/markdown")
    with dl_cols[1]:
        if st.button("Generate combined PDF report", key="batch_pdf_button"):
            with st.spinner("Building combined PDF..."):
                pdf_bytes = build_batch_pdf_bytes(results)
            st.download_button("⬇️ Download combined PDF report", data=pdf_bytes,
                                 file_name="batch_report.pdf", mime="application/pdf",
                                 key="batch_pdf_download")

    # ---- tabular export: a spreadsheet of (problem, target, value,
    # confidence, passed) rows, one per solved target -- what a
    # researcher running dozens of variants almost always actually
    # wants, rather than a prose report meant to be read top to bottom
    # one problem at a time. See batch_solver.batch_results_table().
    results_df = pd.DataFrame(batch_results_table(results))
    table_cols = st.columns(2)
    with table_cols[0]:
        st.download_button("⬇️ Download results as CSV", data=results_df.to_csv(index=False),
                             file_name="batch_results.csv", mime="text/csv", key="batch_csv_download")
    with table_cols[1]:
        xlsx_buf = io.BytesIO()
        results_df.to_excel(xlsx_buf, index=False, sheet_name="results")
        st.download_button(
            "⬇️ Download results as Excel", data=xlsx_buf.getvalue(),
            file_name="batch_results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="batch_xlsx_download",
        )
    with st.expander("Preview results table"):
        st.dataframe(results_df, width='stretch', hide_index=True)

    for r in results:
        preview = r.problem_text.strip().splitlines()[0][:80]
        if r.error:
            icon = "❌"
        elif r.report and r.report.passed:
            icon = "✅"
        else:
            icon = "⚠️"
        with st.expander(f"{icon} Problem {r.index + 1}: {preview}"):
            if r.error:
                st.error(r.error)
                continue
            if r.report is None:
                st.error("Could not be solved.")
                continue
            cr = r.report.confidence_report()
            st.write(f"Confidence: {cr.score:.0%} ({cr.label})")
            if r.retries:
                st.caption(f"Needed {r.retries} verification retr{'y' if r.retries==1 else 'ies'}.")
            for target, val in r.report.sympy_numeric_answers.items():
                st.write(f"**{target}** = {val:.6g}")
