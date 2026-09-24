"""
PDE solver mode: first-order PDEs, heat/wave (Dirichlet/Neumann/Robin), Laplace on a rectangle, 2D finite-difference and 2D heat.
"""
import streamlit as st
import sympy as sp
import numpy as np
import plotly.graph_objects as go
from modules.pde_utils import (
    solve_first_order_pde,
    solve_heat_equation_dirichlet,
    solve_wave_equation_dirichlet,
    solve_heat_equation_neumann,
    solve_wave_equation_neumann,
    solve_heat_equation_robin,
    solve_laplace_rectangle,
    solve_pde_finite_difference_2d,
    solve_heat_equation_2d_dirichlet,
)
from ui.common import live_parse_preview, persist_on_click


def _render_pde_time_animation(solution_expr, length: float, t_max: float, key_prefix: str,
                                 y_label: str = "u", n_frames: int = 30, n_x_points: int = 100) -> None:
    """An animated Plotly line plot of u(x, t) scrubbing through t in
    [0, t_max] via a native Plotly play button + slider (frames built
    once, animated client-side -- no Streamlit rerun needed to step
    through time), replacing what used to be no time-domain
    visualization at all for the heat/wave Fourier solutions (only the
    closed-form LaTeX was shown)."""
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    try:
        f = sp.lambdify((x, t), solution_expr, "numpy")
        xs = np.linspace(0, length, n_x_points)
        ts = np.linspace(0, t_max, n_frames)
        all_ys = [np.real(np.array([complex(f(xv, tv)) for xv in xs])) for tv in ts]
    except Exception as exc:  # noqa: BLE001
        st.caption(f"Could not build an animated view: {exc}")
        return

    y_min, y_max = float(np.min(all_ys)), float(np.max(all_ys))
    pad = 0.1 * max(abs(y_max - y_min), 1e-6)
    frames = [go.Frame(data=[go.Scatter(x=xs, y=all_ys[i], mode="lines", line=dict(width=3))],
                        name=f"{i}") for i in range(n_frames)]
    fig = go.Figure(
        data=[go.Scatter(x=xs, y=all_ys[0], mode="lines", line=dict(width=3))],
        layout=go.Layout(
            xaxis=dict(title="x", range=[0, length]),
            yaxis=dict(title=y_label, range=[y_min - pad, y_max + pad]),
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15,
                                buttons=[
                                    dict(label="▶ Play", method="animate",
                                          args=[None, {"frame": {"duration": 80, "redraw": True},
                                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                                    dict(label="⏸ Pause", method="animate",
                                          args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                                ])],
            sliders=[dict(currentvalue={"prefix": "t = "}, x=0.05, len=0.9,
                           steps=[dict(method="animate", args=[[f"{i}"],
                                        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                                        label=f"{ts[i]:.2g}") for i in range(n_frames)])],
        ),
        frames=frames,
    )
    st.plotly_chart(fig, width="stretch", key=f"{key_prefix}_anim")


def render_pde_tab():
    """PDE capabilities, split by what they can actually solve rather
    than one solver that overpromises -- see pde_utils.py's module
    docstring. Direct symbolic input, same standalone pattern as
    render_dimensional_analysis_tab."""
    st.subheader("🌡️ PDE solver")
    (tab_general, tab_heat_d, tab_heat_n, tab_heat_r, tab_wave_d, tab_wave_n,
     tab_laplace, tab_fd, tab_heat_2d) = st.tabs([
        "First-order PDE", "Heat (fixed ends)", "Heat (insulated)", "Heat (convective)",
        "Wave (fixed ends)", "Wave (free ends)", "Laplace's equation", "General (finite-difference)",
        "Heat (2D animated)"])

    with tab_general:
        st.caption("First-order (or quasilinear first-order) PDEs in two variables, via SymPy's "
                    "pdsolve. Use the same Derivative(...) syntax as the ODE mode. NOT for the "
                    "heat or wave equation -- those are second-order; use the other tabs.")
        pde_str = st.text_input("PDE", key="pde_general_expr",
                                  placeholder="Eq(Derivative(u(x,y), x) + Derivative(u(x,y), y), 2*u(x,y))")
        result = persist_on_click("Solve", "pde_general_button", "pde_general_result",
                                     bool(pde_str.strip()), lambda: solve_first_order_pde(pde_str))
        if result is not None:
            if result.error:
                st.error(result.error)
            else:
                st.latex(sp.latex(result.solution))
                if result.classification:
                    st.caption(f"Classified as: {', '.join(result.classification)}")
                if result.verified:
                    st.success(f"Verified: {result.verification_detail}")
                else:
                    st.warning(f"Could not verify: {result.verification_detail}")

    with tab_heat_d:
        st.caption("u_t = α·u_xx on [0, L] with u(0,t) = u(L,t) = 0 (both ends held at zero), "
                    "solved by separation of variables / Fourier sine series for any initial "
                    "condition f(x).")
        ic_heat = st.text_input("Initial condition f(x)", key="heat_ic", placeholder="x*(1-x)")
        live_parse_preview(ic_heat, ["x"])
        col1, col2 = st.columns(2)
        with col1:
            length_h = st.number_input("Length L", value=1.0, min_value=0.01, key="heat_length")
        with col2:
            alpha_h = st.number_input("Thermal diffusivity α", value=1.0, min_value=0.0001, key="heat_alpha",
                                help="How quickly heat spreads through the material -- larger α means "
                                      "faster diffusion. Units: length²/time.")
        result = persist_on_click(
            "Solve", "heat_button", "heat_result", bool(ic_heat.strip()),
            lambda: solve_heat_equation_dirichlet(ic_heat, length=length_h, alpha=alpha_h))
        if result is not None:
            _render_fourier_pde_result(result)
            if result.error is None:
                t_max = 3.0 / max(alpha_h * (np.pi / length_h) ** 2, 1e-6)
                _render_pde_time_animation(result.solution, length_h, t_max, "heat_d", y_label="u(x,t)")

    with tab_heat_n:
        st.caption("u_t = α·u_xx on [0, L] with INSULATED ends (u_x(0,t) = u_x(L,t) = 0, no heat "
                    "escapes) -- the rod equilibrates to the average initial temperature instead "
                    "of cooling to zero.")
        ic_heat_n = st.text_input("Initial condition f(x)", key="heat_n_ic", placeholder="x*(1-x)")
        live_parse_preview(ic_heat_n, ["x"])
        col1, col2 = st.columns(2)
        with col1:
            length_hn = st.number_input("Length L", value=1.0, min_value=0.01, key="heat_n_length")
        with col2:
            alpha_hn = st.number_input("Thermal diffusivity α", value=1.0, min_value=0.0001, key="heat_n_alpha",
                                 help="How quickly heat spreads through the material -- larger α means "
                                       "faster diffusion. Units: length²/time.")
        result = persist_on_click(
            "Solve", "heat_n_button", "heat_n_result", bool(ic_heat_n.strip()),
            lambda: solve_heat_equation_neumann(ic_heat_n, length=length_hn, alpha=alpha_hn))
        if result is not None:
            _render_fourier_pde_result(result)
            if result.error is None:
                t_max = 3.0 / max(alpha_hn * (np.pi / length_hn) ** 2, 1e-6)
                _render_pde_time_animation(result.solution, length_hn, t_max, "heat_n", y_label="u(x,t)")

    with tab_heat_r:
        st.caption("u_t = α·u_xx on [0, L] with u(0,t) = 0 and a convective (Robin) condition at "
                    "the far end, u_x(L,t) + h·u(L,t) = 0 -- e.g. heat escaping into a surrounding "
                    "medium at zero temperature. Eigenvalues found numerically (no closed form).")
        ic_heat_r = st.text_input("Initial condition f(x)", key="heat_r_ic", placeholder="x*(1-x)")
        live_parse_preview(ic_heat_r, ["x"])
        col1, col2, col3 = st.columns(3)
        with col1:
            length_hr = st.number_input("Length L", value=1.0, min_value=0.01, key="heat_r_length")
        with col2:
            alpha_hr = st.number_input("Thermal diffusivity α", value=1.0, min_value=0.0001, key="heat_r_alpha",
                                 help="How quickly heat spreads through the material -- larger α means "
                                       "faster diffusion. Units: length²/time.")
        with col3:
            robin_h = st.number_input("Convective coefficient h", value=1.0, min_value=0.0001, key="heat_r_h",
                                help="How strongly this end exchanges heat with its surroundings "
                                      "(Newton's law of cooling) -- larger h means faster heat loss "
                                      "there. A Robin (or 'mixed'/'convective') boundary condition "
                                      "blends the value AND its derivative, unlike Dirichlet (fixes "
                                      "the value) or Neumann (fixes the derivative).")
        result = persist_on_click(
            "Solve", "heat_r_button", "heat_r_result", bool(ic_heat_r.strip()),
            lambda: solve_heat_equation_robin(ic_heat_r, length=length_hr, alpha=alpha_hr,
                                                robin_coefficient=robin_h))
        if result is not None:
            _render_fourier_pde_result(result)
            if result.error is None:
                t_max = 3.0 / max(alpha_hr * (np.pi / length_hr) ** 2, 1e-6)
                _render_pde_time_animation(result.solution, length_hr, t_max, "heat_r", y_label="u(x,t)")

    with tab_wave_d:
        st.caption("u_tt = c²·u_xx on [0, L] with u(0,t) = u(L,t) = 0, given initial displacement "
                    "f(x) and initial velocity g(x), solved via Fourier sine series.")
        ic_disp = st.text_input("Initial displacement f(x)", key="wave_ic_disp", placeholder="sin(pi*x)")
        live_parse_preview(ic_disp, ["x"])
        ic_vel = st.text_input("Initial velocity g(x)", key="wave_ic_vel", value="0")
        live_parse_preview(ic_vel, ["x"])
        col1, col2 = st.columns(2)
        with col1:
            length_w = st.number_input("Length L", value=1.0, min_value=0.01, key="wave_length")
        with col2:
            speed_w = st.number_input("Wave speed c", value=1.0, min_value=0.0001, key="wave_speed",
                                help="How fast a disturbance travels along the medium -- e.g. "
                                      "√(tension/density) for a string.")
        result = persist_on_click(
            "Solve", "wave_button", "wave_result", bool(ic_disp.strip()),
            lambda: solve_wave_equation_dirichlet(ic_disp, ic_vel, length=length_w, wave_speed=speed_w))
        if result is not None:
            _render_fourier_pde_result(result)
            if result.error is None:
                t_max = 4 * length_w / max(speed_w, 1e-6)
                _render_pde_time_animation(result.solution, length_w, t_max, "wave_d", y_label="u(x,t)")

    with tab_wave_n:
        st.caption("u_tt = c²·u_xx on [0, L] with FREE ends (u_x(0,t) = u_x(L,t) = 0) -- e.g. a "
                    "string or rod not clamped at either end. A nonzero average initial velocity "
                    "produces rigid translation rather than oscillation.")
        ic_disp_n = st.text_input("Initial displacement f(x)", key="wave_n_ic_disp", placeholder="cos(pi*x)")
        live_parse_preview(ic_disp_n, ["x"])
        ic_vel_n = st.text_input("Initial velocity g(x)", key="wave_n_ic_vel", value="0")
        live_parse_preview(ic_vel_n, ["x"])
        col1, col2 = st.columns(2)
        with col1:
            length_wn = st.number_input("Length L", value=1.0, min_value=0.01, key="wave_n_length")
        with col2:
            speed_wn = st.number_input("Wave speed c", value=1.0, min_value=0.0001, key="wave_n_speed",
                                 help="How fast a disturbance travels along the medium -- e.g. "
                                       "√(tension/density) for a string.")
        result = persist_on_click(
            "Solve", "wave_n_button", "wave_n_result", bool(ic_disp_n.strip()),
            lambda: solve_wave_equation_neumann(ic_disp_n, ic_vel_n, length=length_wn, wave_speed=speed_wn))
        if result is not None:
            _render_fourier_pde_result(result)
            if result.error is None:
                t_max = 4 * length_wn / max(speed_wn, 1e-6)
                _render_pde_time_animation(result.solution, length_wn, t_max, "wave_n", y_label="u(x,t)")

    with tab_laplace:
        st.caption("u_xx + u_yy = 0 on a rectangle [0, width] × [0, height], with independent "
                    "boundary functions on each of the four sides. Sides left at '0' contribute "
                    "nothing (the default).")
        col1, col2 = st.columns(2)
        with col1:
            bc_bottom = st.text_input("Bottom (function of x)", key="laplace_bottom", value="0")
            live_parse_preview(bc_bottom, ["x"])
            bc_left = st.text_input("Left (function of y)", key="laplace_left", value="0")
            live_parse_preview(bc_left, ["y"])
        with col2:
            bc_top = st.text_input("Top (function of x)", key="laplace_top", value="0")
            live_parse_preview(bc_top, ["x"])
            bc_right = st.text_input("Right (function of y)", key="laplace_right", value="0")
            live_parse_preview(bc_right, ["y"])
        col3, col4 = st.columns(2)
        with col3:
            width_l = st.number_input("Width", value=1.0, min_value=0.01, key="laplace_width")
        with col4:
            height_l = st.number_input("Height", value=1.0, min_value=0.01, key="laplace_height")
        result = persist_on_click(
            "Solve", "laplace_button", "laplace_result", True,
            lambda: solve_laplace_rectangle(bc_bottom, bc_top, bc_left, bc_right,
                                              width=width_l, height=height_l))
        if result is not None:
            if result.error:
                st.error(result.error)
            else:
                st.latex(sp.latex(result.solution))
                st.caption(f"{result.modes_used} Fourier modes per active side.")
                if result.pde_residual_zero:
                    st.success("Laplace's equation satisfied exactly (each mode is an exact "
                                "eigenfunction solution).")
                if result.boundary_fit_ok:
                    st.success(f"All boundary conditions matched within tolerance: "
                                f"{result.boundary_fit_errors}")
                else:
                    st.warning(result.verification_detail)

                # 2D heatmap of the steady-state solution -- previously
                # only the closed-form LaTeX was shown, with no visual
                # of what the temperature/potential distribution actually
                # looks like across the rectangle
                try:
                    x_s, y_s = sp.Symbol("x", positive=True), sp.Symbol("y", positive=True)
                    f_uv = sp.lambdify((x_s, y_s), result.solution, "numpy")
                    xs_grid = np.linspace(0, width_l, 60)
                    ys_grid = np.linspace(0, height_l, 60)
                    Z = np.array([[np.real(complex(f_uv(xv, yv))) for xv in xs_grid] for yv in ys_grid])
                    heat_fig = go.Figure(data=go.Heatmap(z=Z, x=xs_grid, y=ys_grid, colorscale="Viridis"))
                    heat_fig.update_layout(title="u(x, y)", xaxis_title="x", yaxis_title="y")
                    st.plotly_chart(heat_fig, width="stretch", key="laplace_heatmap")
                except Exception as exc:  # noqa: BLE001
                    st.caption(f"Could not render a heatmap: {exc}")

    with tab_fd:
        st.caption("Numeric fallback for u_xx + u_yy = source(x,y) on ANY region (not just a "
                    "rectangle) -- describe the domain with a true/false condition on x and y, "
                    "e.g. x**2+y**2<=1 for a disk. First-order accurate at curved boundaries; "
                    "verified by comparing against a doubled-resolution solve.")
        col1, col2 = st.columns(2)
        with col1:
            source_fd = st.text_input("Source term (0 for Laplace's equation)", key="fd_source", value="0",
                                help="A nonzero source turns this into Poisson's equation "
                                      "(u_xx + u_yy = source) instead of Laplace's equation "
                                      "(source = 0) -- e.g. a charge density in electrostatics, "
                                      "or a heat source in steady-state diffusion.")
            live_parse_preview(source_fd, ["x", "y"])
            domain_fd = st.text_input("Domain (x, y) → True/False", key="fd_domain", value="x**2+y**2<=1",
                                help="Any expression that evaluates to true/false for a given "
                                      "(x,y) -- points where it's true are solved; points where "
                                      "it's false use the boundary value. The default is a disk of "
                                      "radius 1 centered at the origin.")
            live_parse_preview(domain_fd, ["x", "y"])
        with col2:
            boundary_fd = st.text_input("Boundary value (function of x, y)", key="fd_boundary", value="0")
            live_parse_preview(boundary_fd, ["x", "y"])
        col3, col4, col5, col6 = st.columns(4)
        with col3:
            x_min = st.number_input("x min", value=-1.2, key="fd_xmin")
        with col4:
            x_max = st.number_input("x max", value=1.2, key="fd_xmax")
        with col5:
            y_min = st.number_input("y min", value=-1.2, key="fd_ymin")
        with col6:
            y_max = st.number_input("y max", value=1.2, key="fd_ymax")
        grid_n = st.slider("Grid resolution", 21, 101, 41, step=10, key="fd_resolution")
        result = persist_on_click(
            "Solve", "fd_button", "fd_result", True,
            lambda: solve_pde_finite_difference_2d(
                source_fd, boundary_fd, domain_fd, (x_min, x_max), (y_min, y_max), grid_n, grid_n))
        if result is not None:
            if result.error:
                st.error(result.error)
            else:
                fig = go.Figure(data=go.Heatmap(z=result.solution_grid.T, x=result.grid_x,
                                                 y=result.grid_y, colorscale="Viridis"))
                fig.update_layout(title="Solution u(x, y)", xaxis_title="x", yaxis_title="y")
                st.plotly_chart(fig, width='stretch', key="fd_heatmap")
                if result.converged:
                    st.success(f"{result.note} (max discrepancy vs. doubled resolution: "
                                f"{result.convergence_error:.3g})")
                else:
                    st.warning(result.note)

    with tab_heat_2d:
        st.caption("The genuinely TIME-DEPENDENT counterpart to the general finite-difference tab "
                    "above, which only solves the steady state (one final heatmap, no sense of "
                    "'watching heat spread over time'). Rectangular domain, Dirichlet boundary -- "
                    "see solve_heat_equation_2d_dirichlet's docstring for why non-rectangular "
                    "domains aren't attempted here.")
        col1, col2 = st.columns(2)
        with col1:
            ic_2d = st.text_input("Initial condition (function of x, y)", key="heat2d_ic",
                                    value="sin(pi*x)*sin(pi*y)")
            live_parse_preview(ic_2d, ["x", "y"])
        with col2:
            bc_2d = st.text_input("Boundary value (function of x, y)", key="heat2d_bc", value="0")
            live_parse_preview(bc_2d, ["x", "y"])
        col3, col4, col5 = st.columns(3)
        with col3:
            width_2d = st.number_input("Width", value=1.0, min_value=0.01, key="heat2d_width")
        with col4:
            height_2d = st.number_input("Height", value=1.0, min_value=0.01, key="heat2d_height")
        with col5:
            alpha_2d = st.number_input("Thermal diffusivity α", value=0.5, min_value=0.0001,
                                         key="heat2d_alpha",
                                         help="How quickly heat spreads through the material -- "
                                               "larger α means faster diffusion.")
        grid_n_2d = st.slider("Grid resolution", 21, 61, 31, step=10, key="heat2d_resolution",
                                help="Higher resolution is more accurate but slower to compute -- "
                                      "the time-stepping scheme's stable step size shrinks faster "
                                      "than the grid gets finer.")
        result = persist_on_click(
            "Solve & animate", "heat2d_button", "heat2d_result", True,
            lambda: solve_heat_equation_2d_dirichlet(
                ic_2d, bc_2d, (0.0, width_2d), (0.0, height_2d), alpha=alpha_2d,
                nx=grid_n_2d, ny=grid_n_2d))
        if result is not None:
            if result.error:
                st.error(result.error)
            else:
                _render_2d_heat_animation(result)
                st.success(f"{len(result.frames)} frames, stable time step dt={result.dt_used:.3g} "
                            f"(computed from the 2D explicit-scheme CFL stability bound).")


def _render_2d_heat_animation(result) -> None:
    """An animated Plotly heatmap scrubbing through the solved time
    frames via a native play button + slider -- the 2D counterpart to
    _render_pde_time_animation's 1D line-plot animation."""
    frames = result.frames
    vmin, vmax = float(frames.min()), float(frames.max())
    plotly_frames = [
        go.Frame(data=[go.Heatmap(z=frames[i].T, x=result.grid_x, y=result.grid_y,
                                    colorscale="Inferno", zmin=vmin, zmax=vmax)], name=f"{i}")
        for i in range(len(frames))
    ]
    fig = go.Figure(
        data=[go.Heatmap(z=frames[0].T, x=result.grid_x, y=result.grid_y,
                           colorscale="Inferno", zmin=vmin, zmax=vmax)],
        layout=go.Layout(
            xaxis=dict(title="x"), yaxis=dict(title="y"),
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15,
                                buttons=[
                                    dict(label="▶ Play", method="animate",
                                          args=[None, {"frame": {"duration": 150, "redraw": True},
                                                          "fromcurrent": True, "transition": {"duration": 0}}]),
                                    dict(label="⏸ Pause", method="animate",
                                          args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
                                ])],
            sliders=[dict(currentvalue={"prefix": "t = "}, x=0.05, len=0.9,
                           steps=[dict(method="animate", args=[[f"{i}"],
                                        {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                                        label=f"{result.times[i]:.2g}") for i in range(len(frames))])],
        ),
        frames=plotly_frames,
    )
    st.plotly_chart(fig, width="stretch", key="heat2d_anim")


def _render_fourier_pde_result(result):
    if result.error:
        st.error(result.error)
        return
    st.latex(sp.latex(result.solution))
    st.caption(f"{result.modes_used} Fourier modes used ({result.coefficient_method}).")
    if result.pde_residual_zero:
        st.success("PDE satisfied exactly (each mode is an exact eigenfunction solution).")
    else:
        st.warning("Could not confirm the PDE is satisfied exactly.")
    if result.ic_fit_ok:
        st.success(f"Initial condition matched within {result.ic_fit_error:.2e} at sample points.")
    else:
        st.warning(result.verification_detail)
