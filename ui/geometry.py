"""
Geometry mode: triangle solving (SSS/SAS/ASA/AAS/SSA) with a labeled schematic.
"""
import streamlit as st
from modules.geometry_solver import solve_triangle, render_triangle
from ui.common import persist_on_click, render_template_bar


def render_geometry_tab():
    """Triangle solving (SSS, SAS, ASA, AAS, and the genuinely ambiguous
    SSA case) with a labeled schematic -- direct numeric input, same
    standalone pattern as render_dimensional_analysis_tab. See
    geometry_solver.py's module docstring for the solving math and why
    it's careful never to use asin() for an angle that could be obtuse."""
    st.subheader("📐 Geometry")
    st.caption("Enter exactly 3 known values (at least one side) to solve a triangle -- the rest "
                "are found via the law of sines/cosines, with a labeled schematic of the result. "
                "Leave unknown fields blank.")

    render_template_bar("triangle", lambda: {
        f"tri_{k}": st.session_state.get(f"tri_{k}", "") for k in ("a", "b", "c", "A", "B", "C")
    })

    col_sides, col_angles = st.columns(2)
    with col_sides:
        st.write("**Sides**")
        a_str = st.text_input("a (opposite A)", key="tri_a", placeholder="unknown")
        b_str = st.text_input("b (opposite B)", key="tri_b", placeholder="unknown")
        c_str = st.text_input("c (opposite C)", key="tri_c", placeholder="unknown")
    with col_angles:
        st.write("**Angles (degrees)**")
        A_str = st.text_input("A (opposite a)", key="tri_A", placeholder="unknown")
        B_str = st.text_input("B (opposite b)", key="tri_B", placeholder="unknown")
        C_str = st.text_input("C (opposite c)", key="tri_C", placeholder="unknown")

    def _collect_knowns() -> dict:
        raw = {"a": a_str, "b": b_str, "c": c_str, "A": A_str, "B": B_str, "C": C_str}
        knowns = {}
        for key, val in raw.items():
            if val.strip():
                try:
                    knowns[key] = float(val)
                except ValueError:
                    knowns[key] = None  # surfaced as a parse error below, not silently dropped
        return knowns

    knowns_preview = _collect_knowns()
    if any(v is None for v in knowns_preview.values()):
        bad = [k for k, v in knowns_preview.items() if v is None]
        st.warning(f"Could not read {', '.join(bad)} as a number.")

    result = persist_on_click(
        "Solve", "triangle_solve_button", "triangle_result",
        len(knowns_preview) == 3 and all(v is not None for v in knowns_preview.values()),
        lambda: solve_triangle(knowns_preview))

    if result is not None:
        if result.error:
            st.error(result.error)
        else:
            st.caption(f"Case: {result.case}")
            if len(result.solutions) > 1:
                st.info(f"This is the ambiguous SSA case -- {len(result.solutions)} valid triangles "
                        "match these measurements. Both are shown below.")
            for i, sol in enumerate(result.solutions):
                if len(result.solutions) > 1:
                    st.markdown(f"#### Solution {i + 1}")
                cols = st.columns(6)
                for col, (label, val) in zip(cols, [("a", sol.a), ("b", sol.b), ("c", sol.c),
                                                       ("A", sol.A), ("B", sol.B), ("C", sol.C)]):
                    with col:
                        st.metric(label, f"{val:.4g}")
                if sol.verified:
                    st.success(f"Verified: angles sum to {sol.angle_sum_check:.6g}° and the law of "
                                "cosines holds exactly for all three sides.")
                else:
                    st.warning(f"Could not fully verify (angle sum {sol.angle_sum_check:.4g}°, "
                                f"residual {sol.law_of_cosines_residual:.3g}).")
                fig = render_triangle(sol, title=f"Solution {i + 1}" if len(result.solutions) > 1
                                       else "Triangle")
                st.plotly_chart(fig, width="stretch", key=f"triangle_fig_{i}")
