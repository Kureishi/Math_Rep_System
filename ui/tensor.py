"""
Tensor-calculus mode: Christoffel symbols, curvature, covariant derivatives and index raising/lowering for a given metric.
"""
import streamlit as st
import sympy as sp
from modules.tensor_calculus import (
    analyze_metric,
    nonzero_christoffel_symbols,
    covariant_derivative_of_vector,
)
from ui.common import live_parse_preview, persist_on_click, render_template_bar, tooltip_header


def _current_tensor_template_payload() -> dict:
    coord_str = st.session_state.get("tensor_coords", "theta, phi")
    n = len([c for c in coord_str.split(",") if c.strip()]) or 2
    payload = {"tensor_coords": coord_str}
    for i in range(n):
        for j in range(n):
            payload[f"metric_{i}_{j}"] = st.session_state.get(f"metric_{i}_{j}", "0")
    return payload


def render_tensor_calculus_tab():
    """Metric-based tensor calculus -- direct symbolic input, same
    standalone pattern as render_dimensional_analysis_tab. See
    tensor_calculus.py's module docstring for scope and the sign-
    convention note."""
    st.subheader("🧮 Tensor calculus")
    st.caption("Enter a metric tensor g_ij as a grid of expressions in your chosen coordinates. "
                "Example (2-sphere of radius R, coordinates θ, φ): rows [[R**2, 0], [0, "
                "R**2*sin(theta)**2]].")

    render_template_bar("tensor_metric", _current_tensor_template_payload)

    st.session_state.setdefault("tensor_coords", "theta, phi")
    coord_str = st.text_input("Coordinate names (comma-separated)", key="tensor_coords",
                                help="The names of the independent variables the metric is written "
                                      "in terms of -- e.g. (θ, φ) for a sphere's usual latitude/"
                                      "longitude-style angles, or (x, y) for an ordinary flat plane.")
    coord_names = [c.strip() for c in coord_str.split(",") if c.strip()]
    n_coords = len(coord_names) if coord_names else 2

    st.write("**Metric tensor g_ij:**")
    metric_rows = []
    default_metric = [["R**2", "0"], ["0", "R**2*sin(theta)**2"]]
    for i in range(n_coords):
        cols = st.columns(n_coords)
        row = []
        for j in range(n_coords):
            default_val = default_metric[i][j] if i < 2 and j < 2 and n_coords == 2 else \
                ("1" if i == j else "0")
            with cols[j]:
                st.session_state.setdefault(f"metric_{i}_{j}", default_val)
                val = st.text_input(f"g[{i}][{j}]", key=f"metric_{i}_{j}",
                                      label_visibility="collapsed")
                live_parse_preview(val, coord_names)
            row.append(val)
        metric_rows.append(row)

    result = persist_on_click("Analyze", "tensor_analyze_button", "_tensor_result", True,
                                 lambda: analyze_metric(metric_rows, coord_names))
    if result is not None and result.error:
        st.error(result.error)
    elif result is not None:
        tooltip_header("Ricci scalar (overall curvature):", "A single number summarizing how curved "
                          "this space is at a point -- zero means flat (locally indistinguishable from "
                          "ordinary Euclidean space), positive is like a sphere's surface, negative is "
                          "like a saddle. In 2D it equals 2× the Gaussian curvature.")
        st.latex(sp.latex(result.ricci_scalar))
        if result.is_flat:
            st.info("This metric is flat (zero Riemann tensor everywhere) -- geometrically "
                    "indistinguishable from ordinary Euclidean space, just in these coordinates.")

        # live parameter exploration: if the Ricci scalar still contains
        # a free symbol beyond the coordinates themselves (e.g. a sphere
        # radius R), let it be scrubbed with a slider and show the
        # curvature update immediately -- no need to retype the metric
        # or re-click Analyze to see how curvature depends on it
        extra_params = sorted(result.ricci_scalar.free_symbols - set(result.coords), key=str)
        if extra_params:
            st.write("**Explore how curvature depends on a parameter:**")
            subs = {}
            for p in extra_params:
                val = st.slider(f"{p}", 0.1, 10.0, 1.0, key=f"tensor_param_{p}")
                subs[p] = val
            try:
                numeric_ricci = complex(result.ricci_scalar.subs(subs))
                display_val = numeric_ricci.real if abs(numeric_ricci.imag) < 1e-9 else numeric_ricci
                st.metric("Ricci scalar at these values", f"{display_val:.5g}")
            except (TypeError, ValueError) as exc:
                st.caption(f"Could not evaluate numerically: {exc}")

        tooltip_header("Nonzero Christoffel symbols Γᵏ_ᵢⱼ:", "The correction terms that account for "
                          "how the coordinate basis itself twists and stretches from point to point -- "
                          "they're what makes a 'straight line' (a geodesic) look curved when written "
                          "in these coordinates, and what a covariant derivative below adds to an "
                          "ordinary derivative to compensate for.")
        symbols = nonzero_christoffel_symbols(result)
        if symbols:
            for k, i, j, val in symbols:
                st.latex(rf"\Gamma^{{{k}}}_{{{i}{j}}} = {sp.latex(val)}")
        else:
            st.caption("All Christoffel symbols are zero (this metric has constant components).")

        if result.metric_compatible:
            st.success("Metric compatibility confirmed (∇g = 0 exactly) -- a self-consistency "
                        "check on the Christoffel-symbol computation, not a property specific to "
                        "this metric.")
        else:
            st.warning("Could not confirm metric compatibility -- this would indicate a "
                        "computation error.")

        st.write("---")
        tooltip_header("Covariant derivative of a vector field (components as functions of the "
                          "coordinates):",
                          "The generalization of an ordinary derivative that accounts for the "
                          "coordinate basis changing from point to point (via the Christoffel symbols "
                          "above) -- it's what correctly measures how a vector field actually changes "
                          "along this curved space, rather than picking up spurious 'change' that's "
                          "really just an artifact of the coordinate system.")
        vec_cols = st.columns(n_coords)
        vec_components = []
        for i in range(n_coords):
            with vec_cols[i]:
                v = st.text_input(f"V^{i}", value="1" if i == 0 else "0", key=f"tensor_vec_{i}")
                live_parse_preview(v, coord_names)
            vec_components.append(v)
        nabla = persist_on_click(
            "Compute covariant derivative", "tensor_cov_button", "_tensor_cov_result", True,
            lambda: covariant_derivative_of_vector(result, vec_components))
        if nabla is not None:
            st.latex(r"\nabla_j V^i = " + sp.latex(nabla))
