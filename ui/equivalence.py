"""
Equivalence-checking mode: are these two expressions the same, and if not where do they differ.
"""
import streamlit as st
import sympy as sp
from modules.proof import build_proof
from modules.equivalence import check_equivalence


def render_equivalence_tab():
    """Standalone building block: 'are these two expressions the same',
    not a representation capability of its own."""
    st.subheader("🔁 Check equivalence")
    st.caption("See whether two symbolic expressions are the same, and if not, why.")

    c1, c2 = st.columns(2)
    with c1:
        expr1_str = st.text_input("Expression 1", placeholder="sin(x)**2 + cos(x)**2", key="equiv_expr1")
    with c2:
        expr2_str = st.text_input("Expression 2", placeholder="1", key="equiv_expr2")
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

    # ---- symbolic proof mode: for a symbolically-confirmed equivalence,
    # show the actual sequence of SymPy simplification passes that
    # reduce the difference to zero, rather than just the final True --
    # see proof.py
    proof_steps = build_proof(result)
    if proof_steps:
        with st.expander("📐 Show proof"):
            for name, expr_latex in proof_steps:
                st.markdown(f"**{name}**")
                st.latex(expr_latex)
