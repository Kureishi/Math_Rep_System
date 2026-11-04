"""The Health check page (modules/health.py): what works, what does not, and what to do about each."""
import streamlit as st

from modules import health
from modules.llm_client import LMStudioClient

_KEY = "_health_report"


def render_health_tab(client: LMStudioClient) -> None:
    st.subheader("🩺 Health check")
    st.caption("Checks that everything the app needs is in place, and for anything that is not, says what to do. It changes "
               "nothing; the only thing it writes is a throwaway probe file to prove the data folder is writable.")
    deep = st.checkbox("Also send the model a one-word request to time it (a few seconds)", False, key="health_deep")
    if st.button("Run the health check", type="primary", key="health_run") or _KEY not in st.session_state:
        with st.spinner("Checking…"):
            st.session_state[_KEY] = health.run_checks(client, include_llm_round_trip=deep)
    report: health.HealthReport = st.session_state[_KEY]
    {"ok": st.success, "warn": st.warning, "fail": st.error}[report.overall](report.headline())
    for c in report.checks:
        st.markdown(f"{c.icon} **{c.title}**: {c.detail}" + (f"  \n<small>{c.seconds:.2f}s</small>" if c.seconds is not None and c.seconds >= 0.5 else ""),
                    unsafe_allow_html=True)
        if c.fix and c.status in ("fail", "warn"):
            st.caption(f"→ {c.fix}")
    with st.expander("Report to paste into a bug report"):
        st.code(report.text(), language=None)
        st.download_button("⬇️ Download as .txt", data=report.text(), file_name="health_report.txt", key="health_dl")
