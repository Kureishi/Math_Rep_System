"""Showing a Diagnosis (modules/llm_diagnosis.py): what happened, the likely causes with their fixes, the facts behind them, and what the model sent."""
import pandas as pd
import streamlit as st

from modules.llm_diagnosis import Diagnosis

_ICON = {"most likely": "🔴", "possible": "🟠", "less likely": "⚪"}


def render_diagnosis(d: Diagnosis, key: str = "diag", expanded_raw: bool = False) -> None:
    """The error, the ranked causes, the facts, the model's reply and the technical text, in that order."""
    (st.error if d.severity == "error" else st.warning)(f"**{d.title}.** {d.summary}", icon="⚠️")
    if d.stage:
        st.caption(f"Happened while: {d.stage}")
    if d.reply_excerpt:
        st.markdown("**What the model said** (start of its reply)")
        st.code(d.reply_excerpt, language="text")
    if d.causes:
        st.markdown("**What probably happened, and what to try**")
        for i, c in enumerate(d.causes, 1):
            st.markdown(f"{i}. {_ICON.get(c.likelihood, '🟠')} *{c.likelihood}* — {c.text}  \n   **Try:** {c.fix}")
    if d.details:
        with st.expander("The facts this is based on", expanded=False):
            st.dataframe(pd.DataFrame(d.details, columns=["", "Value"]), hide_index=True, width="stretch")
    if d.raw_output:
        with st.expander("Raw model response (for debugging)", expanded=expanded_raw):
            st.code(d.raw_output, language="text")
    if d.technical:
        with st.expander("Technical text"):
            st.code(d.technical, language="text")
