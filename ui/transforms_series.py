"""
Transforms & series mode: Laplace/Fourier and Taylor/asymptotic expansions, each verified.
"""
import streamlit as st
import sympy as sp
from modules.transforms import (
    laplace_transform_expr,
    inverse_laplace_transform_expr,
    fourier_transform_expr,
    inverse_fourier_transform_expr,
)
from modules.series_asymptotics import taylor_series, asymptotic_expansion
from ui.common import live_parse_preview, persist_on_click, restore_from_query_param, sync_query_param


def render_transforms_series_tab():
    """Integral transforms (Laplace/Fourier) and series/asymptotic
    expansions -- direct symbolic-expression input, same standalone
    pattern as render_dimensional_analysis_tab. See transforms.py and
    series_asymptotics.py."""
    st.subheader("🔄 Transforms & series")
    tab_laplace, tab_fourier, tab_series = st.tabs(
        ["Laplace transform", "Fourier transform", "Series & asymptotics"])

    with tab_laplace:
        st.caption("Enter a function of t (t > 0 assumed) to transform, or a function of s to "
                    "inverse-transform. Every result is round-tripped through the opposite "
                    "transform and compared back to your input as a check.")
        direction = st.radio("Direction", ["Forward (t → s)", "Inverse (s → t)"],
                              key="laplace_direction", horizontal=True,
                              help="The Laplace transform turns a function of time t into a function "
                                    "of a complex frequency-like variable s, often making differential "
                                    "equations solvable as ordinary algebra -- 'Inverse' undoes that, "
                                    "turning an s-domain expression back into a function of t.")
        restore_from_query_param("laplace_expr")
        expr_str = st.text_input("Expression", key="laplace_expr",
                                  placeholder="exp(-2*t)*sin(3*t)" if direction.startswith("Forward")
                                  else "3/((s+2)**2+9)")
        sync_query_param("laplace_expr")
        live_parse_preview(expr_str, ["t"] if direction.startswith("Forward") else ["s"])
        result = persist_on_click(
            "Transform", "laplace_button", "laplace_transform_result", bool(expr_str.strip()),
            lambda: laplace_transform_expr(expr_str) if direction.startswith("Forward")
            else inverse_laplace_transform_expr(expr_str))
        if result is not None:
            _render_transform_result(result)

    with tab_fourier:
        st.caption("Enter a function of x to transform, or a function of k to inverse-transform.")
        direction_f = st.radio("Direction", ["Forward (x → k)", "Inverse (k → x)"],
                                key="fourier_direction", horizontal=True)
        expr_str_f = st.text_input("Expression", key="fourier_expr",
                                    placeholder="exp(-x**2)" if direction_f.startswith("Forward")
                                    else "sqrt(pi)*exp(-pi**2*k**2)")
        live_parse_preview(expr_str_f, ["x"] if direction_f.startswith("Forward") else ["k"])
        result = persist_on_click(
            "Transform", "fourier_button", "fourier_transform_result", bool(expr_str_f.strip()),
            lambda: fourier_transform_expr(expr_str_f) if direction_f.startswith("Forward")
            else inverse_fourier_transform_expr(expr_str_f))
        if result is not None:
            _render_transform_result(result)

    with tab_series:
        st.caption("Taylor/Laurent series around a finite point, or asymptotic behavior as "
                    "x → ∞ -- each numerically checked against the original function to confirm "
                    "the approximation actually improves toward the expansion point.")
        series_kind = st.radio("Kind", ["Taylor / Laurent (finite point)", "Asymptotic (x → ∞)"],
                                key="series_kind", horizontal=True,
                                help="Taylor/Laurent: a polynomial-like approximation valid NEAR a "
                                      "specific point (Laurent allows negative powers too, needed at a "
                                      "pole). Asymptotic: how the function behaves as x grows without "
                                      "bound -- a different kind of approximation, valid far away "
                                      "rather than close up.")
        expr_str_s = st.text_input("Expression (function of x)", key="series_expr",
                                    placeholder="sin(x)")
        live_parse_preview(expr_str_s, ["x"])
        col1, col2 = st.columns(2)
        point = 0.0
        with col1:
            if series_kind.startswith("Taylor"):
                point = st.number_input("Expansion point", value=0.0, key="series_point")
            order = st.slider("Order", 2, 12, 6, key="series_order",
                                help="How many terms to include -- higher order means a more accurate "
                                      "approximation near the expansion point, at the cost of a longer "
                                      "expression.")
        result = persist_on_click(
            "Expand", "series_button", "series_expand_result", bool(expr_str_s.strip()),
            lambda: taylor_series(expr_str_s, point=point, order=order) if series_kind.startswith("Taylor")
            else asymptotic_expansion(expr_str_s, order=order))
        if result is not None:
            _render_series_result(result)


def _render_transform_result(result):
    if result.error:
        st.error(result.error)
        return
    if not result.evaluated:
        st.warning(f"SymPy could not evaluate this in closed form. {result.verification_detail}")
        st.latex(sp.latex(result.output_expr))
        return
    st.latex(sp.latex(result.output_expr))
    if result.convergence_condition is not None and result.convergence_condition is not True:
        st.caption(f"Valid for: {result.convergence_condition}")
    if result.verified:
        st.success(f"Verified ({result.verification_method}): {result.verification_detail}")
    else:
        st.warning(f"Could not verify: {result.verification_detail}")


def _render_series_result(result):
    if result.error:
        st.error(result.error)
        return
    if not result.expansion_available:
        st.warning(result.verification_detail)
        return
    st.latex(sp.latex(result.truncated) + (r"+\ O(\ldots)" if result.order_term is not None else ""))
    if result.terms:
        st.write("**Terms:** " + ", ".join(f"`{t}`" for t in result.terms))
    if result.verified:
        st.success(f"Verified: {result.verification_detail}")
    else:
        st.warning(f"Could not verify: {result.verification_detail}")
