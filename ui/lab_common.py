"""Display helpers shared by the three simulation labs (waves and optics, dynamics and chaos, circuits and control)."""
import pandas as pd
import streamlit as st

from modules.quantum_common import CheckList


def render_checks(checks: CheckList, title: str = "Checks run on this result") -> None:
    """The verdict first, then every check with the numbers behind it."""
    if not checks.checks:
        return
    if checks.passed:
        st.success(f"✅ {checks.summary()}")
    else:
        st.error(f"⚠️ {checks.summary()} -- read the failures below before using this result")
    with st.expander(title, expanded=not checks.passed):
        st.dataframe(pd.DataFrame([{"": "✅" if c.passed else "❌", "Check": c.label, "What was found": c.detail} for c in checks.checks]),
                     hide_index=True, width="stretch")


def render_notes(notes: list[str]) -> None:
    for n in notes:
        st.info(n)


def num(label: str, value: float, key: str, lo: float | None = None, hi: float | None = None, **kw) -> float:
    return float(st.number_input(label, min_value=lo, max_value=hi, value=float(value), key=key, format="%.6g", **kw))
