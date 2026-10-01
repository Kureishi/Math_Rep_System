"""
Extraction-diff mode: how much does rewording a problem change the derived equations.
"""
import streamlit as st
from modules.llm_client import LMStudioClient
from modules.equation_engine import extract_model
from modules.extraction_diff import diff_extractions


def render_extraction_diff_tab(client: LMStudioClient):
    """Debugging/QA tool: given two DIFFERENT wordings of (nominally)
    the same problem, extracts both independently and shows a
    structural, side-by-side diff -- which variables/equations matched,
    which didn't -- rather than only self_consistency.py's single
    similarity score for repeated runs of the SAME wording. Answers the
    debugging question self-consistency raises but doesn't answer:
    given two SPECIFIC wordings, what EXACTLY differs? See
    extraction_diff.py."""
    st.subheader("🔬 Extraction diff")
    st.caption("Paste two different wordings of the same underlying problem and see exactly how "
                "their extractions differ -- which variables matched, which equations matched, and "
                "what changed. A debugging tool for understanding phrasing sensitivity, not a "
                "student-facing feature.")

    c1, c2 = st.columns(2)
    with c1:
        text_a = st.text_area("Wording A", height=150, key="diff_text_a",
                                placeholder="A car accelerates from 8 m/s to 20 m/s in 6 seconds. "
                                            "Find its acceleration.")
    with c2:
        text_b = st.text_area("Wording B", height=150, key="diff_text_b",
                                placeholder="A vehicle speeds up from an initial 8 m/s to a final "
                                            "20 m/s over a 6 second interval. What's the acceleration?")

    if not st.button("Extract & diff", type="primary", key="diff_run_button"):
        return
    if not text_a.strip() or not text_b.strip():
        st.warning("Enter both wordings.")
        return

    with st.spinner("Extracting both..."):
        try:
            model_a = extract_model(client, text_a)
        except Exception as e:  # noqa: BLE001
            st.error(f"Extraction A failed: {e}")
            return
        try:
            model_b = extract_model(client, text_b)
        except Exception as e:  # noqa: BLE001
            st.error(f"Extraction B failed: {e}")
            return

    diff = diff_extractions(model_a, model_b)

    m1, m2 = st.columns(2)
    m1.metric("Equation shape similarity", f"{diff.equation_shape_similarity:.0%}")
    m2.metric("Variables matched", f"{sum(1 for v in diff.variables if v.status == 'matched')}"
                                     f"/{len(diff.variables)}")

    if diff.domain_matches:
        st.success(f"Domain matches: '{model_a.problem_domain}'")
    else:
        st.warning(f"Domain differs: '{model_a.problem_domain}' vs '{model_b.problem_domain}'")
    if diff.solve_for_matches:
        st.success("solve_for target(s) match (compared by meaning, not symbol name).")
    else:
        st.warning(f"solve_for targets differ: {sorted(diff.solve_for_meanings_a)} vs "
                    f"{sorted(diff.solve_for_meanings_b)}")

    st.write("### Variables")
    icons = {"matched": "✅", "changed": "🟡", "only_in_a": "◀️", "only_in_b": "▶️"}
    for v in diff.variables:
        if v.status == "matched":
            st.write(f"{icons[v.status]} `{v.symbol_a}` (A) / `{v.symbol_b}` (B) -- matched")
        elif v.status == "changed":
            st.write(f"{icons[v.status]} `{v.symbol_a}` (A) / `{v.symbol_b}` (B) -- {v.detail}")
        elif v.status == "only_in_a":
            st.write(f"{icons[v.status]} only in A: `{v.symbol_a}` ({v.detail})")
        else:
            st.write(f"{icons[v.status]} only in B: `{v.symbol_b}` ({v.detail})")

    st.write("### Equations")
    for eq_d in diff.equations:
        if eq_d.status == "matched":
            st.write(f"✅ `{eq_d.name_a}` (A) / `{eq_d.name_b}` (B) -- same structure, ignoring "
                      "variable names")
        elif eq_d.status == "only_in_a":
            st.write(f"◀️ only in A: `{eq_d.name_a}`")
        else:
            st.write(f"▶️ only in B: `{eq_d.name_b}`")
