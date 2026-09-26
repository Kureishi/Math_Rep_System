"""
Plots any derived equation/expression that has exactly one free
"plotting" variable once knowns + workspace values + slider-controlled
parameters are substituted in. Streamlit reruns the script on every
slider move, so the figure is recomputed live -- no manual callback wiring.
"""
import numpy as np
import plotly.graph_objects as go
import sympy as sp

from modules.equation_engine import Equation, ProblemModel


def plottable_free_symbols(eq: Equation, fixed_symbols: set[str]) -> list[str]:
    if eq.sympy_eq is None:
        return []
    expr = eq.sympy_eq.lhs - eq.sympy_eq.rhs
    return sorted(s.name for s in expr.free_symbols if s.name not in fixed_symbols)


def build_plot(model: ProblemModel, eq: Equation, x_symbol: str,
                param_values: dict[str, float], x_range: tuple[float, float],
                y_target: str | None = None, x_log: bool = False, y_log: bool = False) -> go.Figure:
    """
    eq:      the equation to plot
    x_symbol: which free symbol is the x-axis
    param_values: values for every OTHER free symbol (from sliders/workspace)
    x_range: (min, max) for the x-axis sweep
    y_target: which symbol to solve for and plot on the y-axis (must be one
              of model.solve_for). If None or not solvable, falls back to
              plotting the equation's residual (lhs - rhs) against x, with a
              dashed zero-line marking where the equation is satisfied.
    x_log/y_log: log-scale the respective axis -- most useful for sanity-
              checking a power-law or exponential relationship, which
              renders as a straight line on the right log axes. A log
              x-axis uses a geometric (not linear) sweep grid, floored
              just above zero, since log of a non-positive number is
              undefined.
    """
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x = sp.Symbol(x_symbol)
    if x_log:
        lo = max(x_range[0], 1e-6)
        hi = max(x_range[1], lo * 10)
        xs = np.geomspace(lo, hi, 400)
    else:
        xs = np.linspace(x_range[0], x_range[1], 400)

    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    fig = go.Figure()

    if y_target and y_target != x_symbol:
        target = sp.Symbol(y_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify(x, solved[0][target], "numpy")
            ys = f(xs)
            fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines",
                                      name=f"{y_target} vs {x_symbol}"))
            fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_target,
                                xaxis_type="log" if x_log else "linear",
                                yaxis_type="log" if y_log else "linear")
            return fig

    # fallback: plot the residual of the equation itself
    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify(x, residual, "numpy")
    ys = f(xs)
    fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines", name=eq.name))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title=x_symbol, yaxis_title=f"{eq.name} residual (0 = satisfied)",
                        xaxis_type="log" if x_log else "linear",
                        yaxis_type="log" if y_log else "linear")
    return fig


def build_surface_plot(eq: Equation, x_symbol: str, y_symbol: str,
                         param_values: dict[str, float],
                         x_range: tuple[float, float], y_range: tuple[float, float],
                         z_target: str | None = None, resolution: int = 60) -> go.Figure:
    """3D surface plot for an equation with two free plotting variables --
    e.g. distance as a function of both acceleration AND time. Solves the
    equation for z_target (if given and solvable) and evaluates it over an
    (x, y) meshgrid; falls back to plotting the equation's residual surface
    (zero-crossing = where the equation is satisfied) otherwise.
    """
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    fig = go.Figure()

    if z_target and z_target not in (x_symbol, y_symbol):
        target = sp.Symbol(z_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify((x, y), solved[0][target], "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
            fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis",
                                       colorbar=dict(title=z_target)))
            fig.update_layout(
                scene=dict(xaxis_title=x_symbol, yaxis_title=y_symbol, zaxis_title=z_target),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            return fig

    # fallback: residual surface, with a zero-plane the equation satisfies
    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify((x, y), residual, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
    fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale="RdBu",
                               colorbar=dict(title=f"{eq.name} residual")))
    fig.update_layout(
        scene=dict(xaxis_title=x_symbol, yaxis_title=y_symbol,
                    zaxis_title=f"{eq.name} residual (0 = satisfied)"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def build_feasible_region_plot(constraints: list[Equation], x_symbol: str, y_symbol: str,
                                 param_values: dict[str, float],
                                 x_range: tuple[float, float], y_range: tuple[float, float],
                                 resolution: int = 300) -> go.Figure:
    """Shades the region of the (x, y) plane where ALL given inequality
    constraints hold simultaneously -- e.g. a budget constraint AND a time
    constraint AND non-negativity, all at once. Each constraint's boolean
    Relational is lambdified directly (sympy/numpy natively evaluate
    Relational objects elementwise over arrays) and combined with a
    logical AND across the whole constraint set.
    """
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    feasible = np.ones_like(X, dtype=bool)
    skipped = []

    for c in constraints:
        if c.sympy_eq is None:
            continue
        substituted = c.sympy_eq.subs(subs)
        free = substituted.free_symbols
        if not free.issubset({x, y}):
            skipped.append(c.name)  # still has an unsubstituted symbol -- can't evaluate this one
            continue
        try:
            f = sp.lambdify((x, y), substituted, "numpy")
            mask = np.asarray(f(X, Y), dtype=bool)
            feasible &= mask
        except Exception:  # noqa: BLE001
            skipped.append(c.name)

    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        x=xs, y=ys, z=feasible.astype(int),
        colorscale=[[0, "rgba(240,240,240,0.3)"], [1, "rgba(94,234,212,0.55)"]],
        showscale=False, hoverinfo="skip",
    ))
    fig.update_layout(
        xaxis_title=x_symbol, yaxis_title=y_symbol,
        title="Shaded region = every constraint satisfied simultaneously",
    )
    if skipped:
        fig.update_layout(title=f"Shaded = feasible region (skipped: {', '.join(skipped)} -- "
                                  "still has an unresolved symbol)")
    return fig


def build_fit_plot(xs: list[float], ys: list[float], fit_expr, x_label: str = "x", y_label: str = "y",
                     x_log: bool = False, y_log: bool = False) -> go.Figure:
    """Scatter of the raw data points plus the fitted curve evaluated
    over a fine grid spanning (and slightly padding) the data's x-range.
    x_log/y_log log-scale the respective axis -- a power-law fit renders
    as a straight line on log-log axes, an exponential fit as a straight
    line on a log-y/linear-x axis, which is usually a clearer visual
    sanity check of the fit than the default linear-linear view."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", name="data", marker=dict(size=8)))
    if fit_expr is not None:
        lo, hi = min(xs), max(xs)
        pad = (hi - lo) * 0.05 if hi > lo else 1.0
        if x_log:
            grid = np.geomspace(max(lo, 1e-6), max(hi, max(lo, 1e-6) * 10), 300)
        else:
            grid = np.linspace(lo - pad, hi + pad, 300)
        f = sp.lambdify(sp.Symbol("x"), fit_expr, "numpy")
        try:
            yfit = np.broadcast_to(np.asarray(f(grid), dtype=float), grid.shape)
            fig.add_trace(go.Scatter(x=grid, y=yfit, mode="lines", name="fit"))
        except Exception:  # noqa: BLE001
            pass
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label,
                        xaxis_type="log" if x_log else "linear",
                        yaxis_type="log" if y_log else "linear")
    return fig


def build_vector_plot(vectors: list[tuple[str, list[float]]]) -> go.Figure:
    """Draws one or more vectors as arrows from the origin -- 2D (2
    components) or 3D (3 components). Mixed dimensionality isn't
    supported (all vectors passed together must share the same
    dimension); callers should group by dimension before calling this.
    """
    if not vectors:
        return go.Figure()
    dim = len(vectors[0][1])
    fig = go.Figure()

    if dim == 2:
        for name, comps in vectors:
            x, y = comps
            fig.add_trace(go.Scatter(
                x=[0, x], y=[0, y], mode="lines+markers+text",
                line=dict(width=3), marker=dict(size=[0, 8], symbol=["circle", "arrow-bar-up"]),
                text=["", name], textposition="top right", name=name,
            ))
        fig.update_layout(xaxis_title="x", yaxis_title="y",
                            xaxis=dict(zeroline=True), yaxis=dict(zeroline=True, scaleanchor="x"))
    elif dim == 3:
        for name, comps in vectors:
            x, y, z = comps
            fig.add_trace(go.Scatter3d(
                x=[0, x], y=[0, y], z=[0, z], mode="lines+markers+text",
                line=dict(width=6), marker=dict(size=[0, 4]),
                text=["", name], name=name,
            ))
        fig.update_layout(scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z"),
                            margin=dict(l=0, r=0, t=30, b=0))
    else:
        raise ValueError(f"build_vector_plot() only supports 2D or 3D vectors, got {dim}D")
    return fig


_DEP_GRAPH_COLORS = {"known": "#2ca02c", "unknown": "#d62728", "equation": "#1f77b4"}


def build_dependency_graph_plot(nodes, edges):
    """Renders a modules.dependency_graph.(nodes, edges) pair as a
    Plotly figure: three columns (known inputs / equations / unknowns),
    colored by node kind, with directed edges drawn as plain lines
    (Plotly has no built-in arrowheads on Scatter lines; the fixed
    left-to-right column layout makes direction visually obvious
    without needing them)."""
    fig = go.Figure()
    by_id = {n.id: n for n in nodes}
    for edge in edges:
        src, dst = by_id[edge.source], by_id[edge.target]
        fig.add_trace(go.Scatter(x=[src.x, dst.x], y=[src.y, dst.y], mode="lines",
                                   line=dict(color="rgba(120,120,120,0.5)", width=1.5),
                                   showlegend=False, hoverinfo="skip"))

    for kind, color in _DEP_GRAPH_COLORS.items():
        kind_nodes = [n for n in nodes if n.kind == kind]
        if not kind_nodes:
            continue
        fig.add_trace(go.Scatter(
            x=[n.x for n in kind_nodes], y=[n.y for n in kind_nodes], mode="markers+text",
            text=[n.label for n in kind_nodes], textposition="middle center",
            marker=dict(size=36, color=color, line=dict(width=1, color="white")),
            textfont=dict(color="white", size=11),
            name={"known": "Known", "unknown": "Unknown", "equation": "Equation"}[kind],
        ))
    fig.update_layout(
        xaxis=dict(visible=False, range=[-0.5, 2.5]),
        yaxis=dict(visible=False, autorange="reversed"),
        showlegend=True,
    )
    return fig


def build_tornado_chart(entries):
    """Interactive counterpart of plot_snapshot.snapshot_tornado_chart():
    a horizontal bar per input, largest swing at top."""
    fig = go.Figure()
    labels = [e.symbol for e in entries]
    lows = [min(e.low_target, e.high_target) for e in entries]
    highs = [max(e.low_target, e.high_target) for e in entries]
    widths = [h - l for l, h in zip(lows, highs)]
    fig.add_trace(go.Bar(
        y=labels, x=widths, base=lows, orientation="h",
        hovertext=[f"{e.symbol}: {e.low_target:.4g} to {e.high_target:.4g}" for e in entries],
        hoverinfo="text",
    ))
    fig.update_layout(xaxis_title="target value across each input's swept range",
                        yaxis=dict(autorange="reversed"))
    return fig


def build_sweep_chart(sweep_result):
    """Interactive counterpart of plot_snapshot.snapshot_sweep_chart()."""
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=sweep_result.values, y=sweep_result.target_values,
                               mode="lines", name="swept"))
    if sweep_result.nominal_target is not None:
        fig.add_trace(go.Scatter(x=[sweep_result.nominal_input], y=[sweep_result.nominal_target],
                                   mode="markers", marker=dict(size=10, color="red"), name="nominal"))
    fig.update_layout(xaxis_title=sweep_result.symbol, yaxis_title="target value")
    return fig


def build_contour_plot(eq: Equation, x_symbol: str, y_symbol: str,
                         param_values: dict[str, float],
                         x_range: tuple[float, float], y_range: tuple[float, float],
                         z_target: str | None = None, resolution: int = 80) -> go.Figure:
    """2D contour/level-set counterpart of build_surface_plot() -- the
    same underlying (x, y) -> z evaluation, but as labeled contour lines
    on a flat plane rather than a rotatable 3D surface. Often more
    readable for a research figure (no viewing angle to fight with) and
    the standard way to show where a two-variable relationship is
    constant, e.g. reading off exactly which (x, y) combinations give a
    particular z value."""
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    fig = go.Figure()

    z_label = z_target or f"{eq.name} residual"
    if z_target and z_target not in (x_symbol, y_symbol):
        target = sp.Symbol(z_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify((x, y), solved[0][target], "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
            fig.add_trace(go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis",
                                       contours=dict(showlabels=True),
                                       colorbar=dict(title=z_target)))
            fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_symbol)
            return fig

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify((x, y), residual, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
    fig.add_trace(go.Contour(
        x=xs, y=ys, z=Z, colorscale="RdBu", contours=dict(showlabels=True),
        colorbar=dict(title=f"{eq.name} residual"),
    ))
    fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_symbol,
                        title=f"Contours of {z_label} (0 = equation satisfied)")
    return fig


def build_overlay_plot(series: list[dict], x_label: str = "x", y_label: str = "y",
                         title: str | None = None) -> go.Figure:
    """Generic multi-series overlay -- plots several (x, y) curves/traces
    together on shared axes for direct visual comparison, rather than
    only ever seeing one result at a time. Each entry in `series` is
    {"x": [...], "y": [...], "name": str, optionally "mode": "lines" |
    "markers" | "lines+markers"}. Used for e.g. overlaying every
    candidate fit family against the same data (curve_fitting.best_fit's
    result set), or any other "compare several curves at once" need --
    kept generic rather than tied to one specific caller."""
    fig = go.Figure()
    for s in series:
        fig.add_trace(go.Scatter(x=s["x"], y=s["y"], mode=s.get("mode", "lines"), name=s.get("name", "")))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label, title=title)
    return fig


def build_chain_sweep_plot(sweep_rows: list[dict], swept_symbol: str,
                             step_labels: dict[int, str]) -> go.Figure:
    """Plots the result of chains.sweep_step_binding(): one line per
    downstream chain step, x = the swept literal input's value, y = that
    step's resolved output at each swept value. `sweep_rows` is the list
    chains.sweep_step_binding() returns: [{"value": float,
    "outputs": {position: float|None}}, ...]. `step_labels` maps a
    step's position -> a display name (typically "step N: output_symbol").
    A step whose output failed to solve at a given swept value (None) is
    simply gapped in its line rather than plotted as zero."""
    fig = go.Figure()
    x_values = [row["value"] for row in sweep_rows]
    positions = sorted({pos for row in sweep_rows for pos in row["outputs"]})
    for pos in positions:
        y_values = [row["outputs"].get(pos) for row in sweep_rows]
        fig.add_trace(go.Scatter(x=x_values, y=y_values, mode="lines+markers",
                                   name=step_labels.get(pos, f"step {pos + 1}"),
                                   connectgaps=False))
    fig.update_layout(xaxis_title=swept_symbol, yaxis_title="step output value",
                        title=f"Downstream outputs as {swept_symbol} is swept")
    return fig


def build_spread_plot(values: list[float], target_symbol: str, labels: list[str] | None = None) -> go.Figure:
    """Strip/box plot of a small set of numeric answers for the SAME
    target that came from different runs -- built for
    self_consistency.py's per-run numeric answers, to make visible not
    just THAT repeated runs disagree but by HOW MUCH. Shows every
    individual point (so a lone outlier run is visible, not averaged
    away) plus the box's mean/quartile summary."""
    fig = go.Figure()
    fig.add_trace(go.Box(
        y=values, boxpoints="all", jitter=0.4, pointpos=0, name=target_symbol,
        text=labels or [f"run {i + 1}" for i in range(len(values))],
        hoverinfo="y+text",
    ))
    fig.update_layout(yaxis_title=target_symbol, showlegend=False,
                        title=f"{target_symbol} across independent re-extraction runs")
    return fig


def build_histogram_plot(samples: list[float], target_symbol: str,
                           mean: float | None = None, p5: float | None = None,
                           p95: float | None = None) -> go.Figure:
    """Histogram of Monte Carlo output samples (see monte_carlo.py), with
    vertical reference lines for the mean and the 5th/95th percentile
    band -- the propagated-uncertainty counterpart of a single point
    answer: instead of "the answer is 4.2", this shows the whole spread
    of answers implied by the input uncertainties, e.g. "4.2 ± 0.3,
    5-95% band 3.7 to 4.7"."""
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=samples, nbinsx=min(60, max(10, len(samples) // 20)),
                                 marker=dict(color="#2a9d8f")))
    if mean is not None:
        fig.add_vline(x=mean, line_color="black", line_width=2,
                       annotation_text="mean", annotation_position="top")
    if p5 is not None:
        fig.add_vline(x=p5, line_dash="dash", line_color="gray",
                       annotation_text="5th pct", annotation_position="top left")
    if p95 is not None:
        fig.add_vline(x=p95, line_dash="dash", line_color="gray",
                       annotation_text="95th pct", annotation_position="top right")
    fig.update_layout(xaxis_title=target_symbol, yaxis_title="count",
                        title=f"Monte Carlo distribution of {target_symbol} ({len(samples)} samples)")
    return fig


def build_sweep_heatmap(x_values: list, y_values: list, z_matrix, x_label: str, y_label: str,
                          target_symbol: str) -> go.Figure:
    """A 2D parameter sweep's result as a heatmap -- x/y are the two
    swept variables, color is the target value at each grid point. The
    natural visualization for parameter_sweep.py's 2-variable case,
    since a results table alone doesn't make the SHAPE of a sensitivity
    surface (a ridge, a saddle, a monotonic gradient) visible at a
    glance the way a heatmap does."""
    fig = go.Figure(data=go.Heatmap(
        x=x_values, y=y_values, z=z_matrix, colorscale="Viridis",
        colorbar=dict(title=target_symbol), hoverongaps=False,
    ))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label,
                        title=f"{target_symbol} across {x_label} \u00d7 {y_label}")
    return fig


# ================================================================== animated / dynamical-systems views
#
# Everything above this line is a snapshot of a single static relationship.
# The four functions below instead show something CHANGING -- a coupled
# system's direction field and trajectory, a recurrence converging (or not)
# to a fixed point, a Monte Carlo estimate stabilizing as samples accumulate,
# and an optimizer's descent toward a critical point -- using Plotly's
# frames + updatemenus play/pause mechanism already established by
# ui/pde.py's PDE time-evolution animation, so all of the app's animations
# share one interaction pattern rather than each inventing its own.

def build_phase_portrait(dx_dt, dy_dt, x_range: tuple[float, float], y_range: tuple[float, float],
                           x_label: str = "x", y_label: str = "y",
                           trajectory: tuple[np.ndarray, np.ndarray] | None = None,
                           resolution: int = 16) -> go.Figure:
    """A 2D phase portrait for a coupled first-order system dx/dt = dx_dt(x, y),
    dy/dt = dy_dt(x, y): a quiver direction field (via
    plotly.figure_factory.create_quiver) showing the qualitative flow
    everywhere in the plane, with the actual solved trajectory -- if one is
    given -- drawn over it as a highlighted path from its start (circle) to
    its end (star). Uses create_quiver rather than create_streamline: a
    streamline plot integrates field lines via internal RK4 stepping, which
    divides by the local speed and raises on any grid point sitting exactly
    on an equilibrium (velocity = 0, a NaN "cannot convert float NaN to
    integer" crash) -- and an equilibrium is usually the single most
    important point a phase portrait exists to show, not a rare edge case
    to design around. A quiver plot draws one arrow per grid point directly
    from the field with no integration, so a zero-length arrow at an
    equilibrium is just a dot, not a crash. `dx_dt`/`dy_dt` are plain
    numpy-vectorized callables (e.g. from sp.lambdify((x, y), expr,
    "numpy")), not sympy expressions -- this function does no symbolic work
    itself. `trajectory` is (xs, ys): the same two arrays either axis of a
    solved (x(t), y(t)) pair evaluates to over the plotted time range.
    """
    import plotly.figure_factory as ff
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    U = np.asarray(dx_dt(X, Y), dtype=float)
    V = np.asarray(dy_dt(X, Y), dtype=float)
    U = np.nan_to_num(U, nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(V, nan=0.0, posinf=0.0, neginf=0.0)
    # normalize arrow length so a wildly varying speed across the grid
    # doesn't make slow regions invisible or fast regions overlap --
    # direction is what a phase portrait needs to show, not magnitude
    speed = np.sqrt(U ** 2 + V ** 2)
    speed[speed == 0] = 1.0
    step = max((x_range[1] - x_range[0]) / resolution, (y_range[1] - y_range[0]) / resolution)
    Un, Vn = (U / speed) * step * 0.8, (V / speed) * step * 0.8

    fig = ff.create_quiver(X.flatten(), Y.flatten(), Un.flatten(), Vn.flatten(),
                             scale=1, arrow_scale=0.35,
                             line=dict(color="rgba(100,110,130,0.6)", width=1.4))
    fig.update_traces(showlegend=False)
    if trajectory is not None:
        tx, ty = trajectory
        fig.add_trace(go.Scatter(x=tx, y=ty, mode="lines", name="trajectory",
                                    line=dict(color="#2E5EAA", width=3)))
        fig.add_trace(go.Scatter(x=[tx[0]], y=[ty[0]], mode="markers", name="start",
                                    marker=dict(symbol="circle", size=11, color="#1E7E34")))
        fig.add_trace(go.Scatter(x=[tx[-1]], y=[ty[-1]], mode="markers", name="end",
                                    marker=dict(symbol="star", size=14, color="#C0392B")))
    fig.update_layout(xaxis_title=x_label, yaxis_title=y_label,
                        title=f"Phase portrait: {x_label}\u2013{y_label}")
    return fig


def build_cobweb_plot(g, x0: float, x_range: tuple[float, float], n_steps: int = 25,
                        x_label: str = "a(n)") -> go.Figure:
    """A cobweb (staircase) diagram for a first-order recurrence
    a(n+1) = g(a(n)): the curve y = g(x), the diagonal y = x (every fixed
    point of the recurrence is where the two cross), and the zig-zag path
    -- vertical from (x, x) up/down to (x, g(x)), horizontal across to
    (g(x), g(x)), repeated -- that makes convergence, oscillation, or
    divergence visually obvious in a way a plain "value vs. step index"
    plot doesn't. `g` is a plain numpy-vectorized callable. Animated: each
    frame reveals one more zig-zag segment, via the same
    frames + play/pause pattern as the other functions in this section.
    """
    curve_xs = np.linspace(x_range[0], x_range[1], 300)
    curve_ys = np.asarray(g(curve_xs), dtype=float)

    xs = [x0]
    for _ in range(n_steps):
        xs.append(float(np.real(g(xs[-1]))))
    # the staircase as one continuous polyline: (x0,x0) -> (x0,g(x0)) -> (g(x0),g(x0)) -> ...
    path_x, path_y = [xs[0]], [xs[0]]
    for i in range(n_steps):
        path_x += [xs[i], xs[i + 1]]
        path_y += [xs[i + 1], xs[i + 1]]

    base = [
        go.Scatter(x=curve_xs, y=curve_ys, mode="lines", name="y = g(x)",
                    line=dict(color="#2E5EAA", width=2.5)),
        go.Scatter(x=curve_xs, y=curve_xs, mode="lines", name="y = x",
                    line=dict(color="rgba(100,100,100,0.5)", dash="dash")),
        go.Scatter(x=[], y=[], mode="lines", name="path", line=dict(color="#C0392B", width=2)),
    ]
    frames = [go.Frame(data=[go.Scatter(x=path_x[:2 * i + 1], y=path_y[:2 * i + 1])],
                         traces=[2], name=f"{i}") for i in range(n_steps + 1)]
    fig = go.Figure(
        data=base,
        layout=go.Layout(
            xaxis=dict(title=x_label, range=x_range), yaxis=dict(title=f"g({x_label})", range=x_range),
            title=f"Cobweb diagram (x0 = {x0:g}, {n_steps} steps)",
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15, buttons=[
                dict(label="\u25b6 Play", method="animate",
                      args=[None, {"frame": {"duration": 250, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="\u23f8 Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ])],
            sliders=[dict(currentvalue={"prefix": "step "}, x=0.05, len=0.9, steps=[
                dict(method="animate", args=[[f"{i}"],
                      {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                      label=f"{i}") for i in range(n_steps + 1)])],
        ),
        frames=frames,
    )
    return fig


def build_monte_carlo_convergence_plot(samples: list[float], target_symbol: str,
                                          n_frames: int = 40) -> go.Figure:
    """Animates a Monte Carlo estimate stabilizing as samples accumulate --
    the running mean (with a running ±1 std band) plotted against sample
    count, playing from the first handful of samples up to the full set.
    The complementary view to build_histogram_plot's static end-state
    snapshot: this one makes visible HOW the estimate settled down (or
    didn't, within the sample budget used), which is the point a
    fixed-in-time histogram can't make on its own.
    """
    arr = np.asarray(samples, dtype=float)
    n = len(arr)
    running_mean = np.cumsum(arr) / np.arange(1, n + 1)
    # running (population) std via the running sum-of-squares identity --
    # avoids an O(n^2) recompute of np.std(arr[:k]) for every k
    running_sq_mean = np.cumsum(arr ** 2) / np.arange(1, n + 1)
    running_var = np.maximum(running_sq_mean - running_mean ** 2, 0.0)
    running_std = np.sqrt(running_var)

    checkpoints = np.unique(np.linspace(1, n, min(n_frames, n)).astype(int))
    idx = np.arange(1, n + 1)
    y_min = float(np.min(running_mean - running_std))
    y_max = float(np.max(running_mean + running_std))
    pad = 0.1 * max(y_max - y_min, 1e-9)

    def frame_data(k):
        upper = (running_mean[:k] + running_std[:k]).tolist()
        lower = (running_mean[:k] - running_std[:k]).tolist()
        return [
            go.Scatter(x=idx[:k].tolist() + idx[:k].tolist()[::-1], y=upper + lower[::-1],
                        fill="toself", fillcolor="rgba(46,94,170,0.15)",
                        line=dict(color="rgba(0,0,0,0)"), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=idx[:k], y=running_mean[:k], mode="lines", name="running mean",
                        line=dict(color="#2E5EAA", width=3)),
        ]

    frames = [go.Frame(data=frame_data(k), name=f"{k}") for k in checkpoints]
    fig = go.Figure(
        data=frame_data(int(checkpoints[0])),
        layout=go.Layout(
            xaxis=dict(title="samples used", range=[1, n]),
            yaxis=dict(title=target_symbol, range=[y_min - pad, y_max + pad]),
            title=f"Convergence of the {target_symbol} estimate ({n} samples)",
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15, buttons=[
                dict(label="\u25b6 Play", method="animate",
                      args=[None, {"frame": {"duration": 60, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="\u23f8 Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ])],
            sliders=[dict(currentvalue={"prefix": "n = "}, x=0.05, len=0.9, steps=[
                dict(method="animate", args=[[f"{k}"],
                      {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                      label=f"{k}") for k in checkpoints])],
        ),
        frames=frames,
    )
    return fig


def build_descent_path_plot(f, path: list[tuple[float, float]], x_range: tuple[float, float],
                              y_range: tuple[float, float], x_label: str = "x", y_label: str = "y",
                              resolution: int = 60) -> go.Figure:
    """An objective function's contour map with a numerical optimization
    path (see modules.optimization_utils.gradient_descent_path) animated
    walking from its starting point to the critical point already found
    symbolically -- turns "here is the answer" into "here is how an
    iterative method would arrive at it", which the app's actual (direct
    calculus / Lagrange) solve doesn't produce on its own since it jumps
    straight to the critical point with no iteration to show. `f` is a
    plain numpy-vectorized callable of two arguments.
    """
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    Z = np.asarray(f(X, Y), dtype=float)
    px = [p[0] for p in path]
    py = [p[1] for p in path]

    base = [
        go.Contour(x=xs, y=ys, z=Z, colorscale="Blues", showscale=False,
                    contours=dict(coloring="fill"), opacity=0.85),
        go.Scatter(x=[], y=[], mode="lines+markers", name="descent path",
                    line=dict(color="#C0392B", width=2), marker=dict(size=5, color="#C0392B")),
        go.Scatter(x=[px[-1]], y=[py[-1]], mode="markers", name="critical point",
                    marker=dict(symbol="star", size=16, color="#1E7E34")),
    ]
    frames = [go.Frame(data=[go.Scatter(x=px[:i + 1], y=py[:i + 1])], traces=[1], name=f"{i}")
              for i in range(len(path))]
    fig = go.Figure(
        data=base,
        layout=go.Layout(
            xaxis=dict(title=x_label, range=x_range), yaxis=dict(title=y_label, range=y_range),
            title="Convergence to the critical point",
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15, buttons=[
                dict(label="\u25b6 Play", method="animate",
                      args=[None, {"frame": {"duration": 150, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="\u23f8 Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ])],
            sliders=[dict(currentvalue={"prefix": "iteration "}, x=0.05, len=0.9, steps=[
                dict(method="animate", args=[[f"{i}"],
                      {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                      label=f"{i}") for i in range(len(path))])],
        ),
        frames=frames,
    )
    return fig


def build_motion_diagram(t_values: np.ndarray, x_values: np.ndarray, v_values: np.ndarray,
                           x_label: str = "position", x_unit: str = "", t_unit: str = "",
                           n_frames: int = 50) -> go.Figure:
    """A classic kinematics motion diagram: a dot moving along a 1D track
    with a velocity vector attached (top panel), synced to the same dot
    tracing out position-vs-time underneath (bottom panel). Built from a
    modules.motion_diagram.MotionTrajectory's three arrays -- this function
    does no physics of its own, same separation as every other function in
    this module. The top-panel view is the actual point of this function:
    a position-vs-time GRAPH (the bottom panel, which build_plot could
    already produce for any two related quantities) doesn't by itself show
    the intro-physics idea of "an object moving through space with a
    velocity" nearly as directly as watching a dot move along a line does.
    """
    from plotly.subplots import make_subplots

    n = len(t_values)
    idx = np.unique(np.linspace(0, n - 1, min(n_frames, n)).astype(int))
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    x_pad = 0.18 * max(x_max - x_min, 1.0)
    v_max = max(abs(float(np.max(v_values))), abs(float(np.min(v_values))), 1e-9)
    v_scale = 0.12 * max(x_max - x_min, 1.0) / v_max
    x_unit_sfx = f" ({x_unit})" if x_unit else ""
    t_unit_sfx = f" ({t_unit})" if t_unit else ""

    fig = make_subplots(rows=2, cols=1, row_heights=[0.28, 0.72], vertical_spacing=0.16,
                          subplot_titles=("motion", f"{x_label}{x_unit_sfx} vs. time{t_unit_sfx}"))

    def frame_traces(i: int):
        x, v = float(x_values[i]), float(v_values[i])
        arrow_end = x + v * v_scale
        arrow_symbol = "triangle-right" if v >= 0 else "triangle-left"
        return [
            go.Scatter(x=[x_min - x_pad, x_max + x_pad], y=[0, 0], mode="lines",
                        line=dict(color="rgba(150,150,150,0.35)", width=2), showlegend=False,
                        hoverinfo="skip"),
            go.Scatter(x=[x, arrow_end], y=[0, 0], mode="lines+markers",
                        line=dict(color="#C0392B", width=3),
                        marker=dict(size=[0, 11], symbol=["circle", arrow_symbol], color="#C0392B"),
                        name="velocity", showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[x], y=[0], mode="markers", marker=dict(size=18, color="#2E5EAA"),
                        name="object", showlegend=False),
            go.Scatter(x=t_values[:i + 1], y=x_values[:i + 1], mode="lines",
                        line=dict(color="#2E5EAA", width=2), showlegend=False, hoverinfo="skip"),
            go.Scatter(x=[t_values[i]], y=[x], mode="markers", marker=dict(size=11, color="#C0392B"),
                        showlegend=False),
        ]

    base = frame_traces(0)
    fig.add_trace(base[0], row=1, col=1)
    fig.add_trace(base[1], row=1, col=1)
    fig.add_trace(base[2], row=1, col=1)
    fig.add_trace(base[3], row=2, col=1)
    fig.add_trace(base[4], row=2, col=1)

    frames = [go.Frame(data=frame_traces(int(i)), traces=[0, 1, 2, 3, 4], name=f"{int(i)}") for i in idx]
    fig.update_layout(
        height=520, margin=dict(t=60, b=40),
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.14, buttons=[
            dict(label="\u25b6 Play", method="animate",
                  args=[None, {"frame": {"duration": 60, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="\u23f8 Pause", method="animate",
                  args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
        ])],
        sliders=[dict(currentvalue={"prefix": "t = "}, x=0.05, len=0.9, steps=[
            dict(method="animate", args=[[f"{int(i)}"],
                  {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                  label=f"{t_values[int(i)]:.2g}") for i in idx])],
    )
    fig.update_yaxes(visible=False, row=1, col=1)
    fig.update_xaxes(title=f"position{x_unit_sfx}", row=1, col=1)
    fig.update_xaxes(title=f"time{t_unit_sfx}", row=2, col=1)
    fig.update_yaxes(title=f"{x_label}{x_unit_sfx}", row=2, col=1)
    fig.frames = frames
    return fig
