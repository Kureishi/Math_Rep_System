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
from modules.plot_params import solve_for_target, residual_expression


def plottable_free_symbols(eq: Equation, fixed_symbols: set[str]) -> list[str]:
    if eq.sympy_eq is None:
        return []
    expr = eq.sympy_eq.lhs - eq.sympy_eq.rhs
    return sorted(s.name for s in expr.free_symbols if s.name not in fixed_symbols)


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
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

    fig = go.Figure()

    if y_target and y_target != x_symbol:
        solved_expr = solve_for_target(eq, y_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify(x, solved_expr, "numpy")
            ys = f(xs)
            fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines",
                                      name=f"{y_target} vs {x_symbol}"))
            fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_target,
                                xaxis_type="log" if x_log else "linear",
                                yaxis_type="log" if y_log else "linear")
            return fig

    # fallback: plot the residual of the equation itself
    residual = residual_expression(eq, param_values, {x_symbol})
    f = sp.lambdify(x, residual, "numpy")
    ys = f(xs)
    fig.add_trace(go.Scatter(x=xs, y=np.real(ys), mode="lines", name=eq.name))
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(xaxis_title=x_symbol, yaxis_title=f"{eq.name} residual (0 = satisfied)",
                        xaxis_type="log" if x_log else "linear",
                        yaxis_type="log" if y_log else "linear")
    return fig


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
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

    fig = go.Figure()

    if z_target and z_target not in (x_symbol, y_symbol):
        solved_expr = solve_for_target(eq, z_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify((x, y), solved_expr, "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
            fig.add_trace(go.Surface(x=xs, y=ys, z=Z, colorscale="Viridis",
                                       colorbar=dict(title=z_target)))
            fig.update_layout(
                scene=dict(xaxis_title=x_symbol, yaxis_title=y_symbol, zaxis_title=z_target),
                margin=dict(l=0, r=0, t=30, b=0),
            )
            return fig

    # fallback: residual surface, with a zero-plane the equation satisfies
    residual = residual_expression(eq, param_values, {x_symbol, y_symbol})
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


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
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

    fig = go.Figure()

    z_label = z_target or f"{eq.name} residual"
    if z_target and z_target not in (x_symbol, y_symbol):
        solved_expr = solve_for_target(eq, z_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify((x, y), solved_expr, "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex)) if not np.isscalar(f(X, Y)) else np.full_like(X, f(X, Y))
            fig.add_trace(go.Contour(x=xs, y=ys, z=Z, colorscale="Viridis",
                                       contours=dict(showlabels=True),
                                       colorbar=dict(title=z_target)))
            fig.update_layout(xaxis_title=x_symbol, yaxis_title=y_symbol)
            return fig

    residual = residual_expression(eq, param_values, {x_symbol, y_symbol})
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


def add_camera_rotation(fig: go.Figure, n_frames: int = 60, elevation_deg: float = 25.0,
                          radius: float = 1.9) -> go.Figure:
    """Adds a slow orbit-around animation (frames + play/pause, same
    pattern as every other function in this section) to an EXISTING 3D
    Plotly figure, by sweeping the camera eye position in a circle at a
    fixed elevation and distance. Works on any 3D scene, not just
    build_surface_plot's output -- it only touches layout.scene.camera,
    never the figure's data -- so a tensor-calculus or other 3D view could
    reuse this exactly the same way. A static 3D surface is notoriously
    hard to read from a single fixed angle (which bump is a peak and which
    is a dip isn't always obvious without depth cues from motion); slowly
    orbiting it is a cheap, well-known fix for that, not just a decoration.
    """
    import math as _math
    thetas = [i * 2 * _math.pi / n_frames for i in range(n_frames)]
    elev_rad = _math.radians(elevation_deg)

    def eye(theta: float) -> dict:
        return dict(x=radius * _math.cos(elev_rad) * _math.cos(theta),
                     y=radius * _math.cos(elev_rad) * _math.sin(theta),
                     z=radius * _math.sin(elev_rad))

    frames = [go.Frame(layout=go.Layout(scene_camera=dict(eye=eye(t))), name=f"{i}")
               for i, t in enumerate(thetas)]
    fig.update_layout(
        scene_camera=dict(eye=eye(thetas[0])),
        updatemenus=list(fig.layout.updatemenus or []) + [dict(
            type="buttons", showactive=False, x=0.02, y=0.02, buttons=[
                dict(label="\U0001f504 Rotate", method="animate",
                      args=[None, {"frame": {"duration": 40, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0},
                                     "mode": "immediate"}]),
                dict(label="\u23f8 Stop", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ])],
    )
    fig.frames = frames
    return fig


# ---------------------------------------------------------------------------
# Time-resolved views: error over time, animated phase trails, a time cursor
# linking a time series to its phase plane, and series partial-sum animation.
# All four use the same Play/Pause + slider mechanism as the cobweb diagram
# above; their matplotlib GIF/PNG counterparts live in plot_snapshot.py.
# ---------------------------------------------------------------------------
_PALETTE = ["#2E5EAA", "#C0392B", "#1E7E34", "#8E44AD", "#D68910", "#117A8B"]


def _frame_indices(n: int, max_frames: int) -> list[int]:
    """At most `max_frames` evenly spaced indices into a length-`n` series,
    always including the first and last -- keeps an animation's payload
    bounded however finely the underlying data is sampled."""
    if n <= 0:
        return []
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, min(n, max_frames))})


def _play_layout(labels: list[str], prefix: str, duration_ms: int = 90) -> dict:
    """updatemenus (Play/Pause) + a slider over frames named "0".."n-1"."""
    return dict(
        updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.15, buttons=[
            dict(label="\u25b6 Play", method="animate",
                  args=[None, {"frame": {"duration": duration_ms, "redraw": True},
                                 "fromcurrent": True, "transition": {"duration": 0}}]),
            dict(label="\u23f8 Pause", method="animate",
                  args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
        ])],
        sliders=[dict(currentvalue={"prefix": prefix}, x=0.05, len=0.9, steps=[
            dict(method="animate",
                  args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                  label=label) for i, label in enumerate(labels)])],
    )


def _rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    return f"rgba({int(h[0:2], 16)},{int(h[2:4], 16)},{int(h[4:6], 16)},{alpha:.2f})"


def build_ode_error_plot(comparison) -> go.Figure:
    """The closed-form ODE solution against an independent numerical
    integration, over time, with the pointwise relative error underneath on
    a log axis against the same tolerance numerical_cross_check uses for its
    verdict. `comparison` is a modules.ode_trajectories.TrajectoryComparison
    with applicable=True."""
    from plotly.subplots import make_subplots
    if not comparison.applicable:
        raise ValueError(comparison.reason or "No comparison available.")
    verdict = "agree" if comparison.ok else "DISAGREE"
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.58, 0.42], vertical_spacing=0.1,
                         subplot_titles=("Closed form vs. numerical integration",
                                         "Relative error (log scale)"))
    t = comparison.t
    marker_every = max(1, len(t) // 25)
    for i, name in enumerate(comparison.names):
        color = _PALETTE[i % len(_PALETTE)]
        fig.add_trace(go.Scatter(x=t, y=comparison.symbolic[i], mode="lines", name=f"{name} (closed form)",
                                   line=dict(color=color, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t[::marker_every], y=comparison.numeric[i][::marker_every], mode="markers",
                                   name=f"{name} (numerical)",
                                   marker=dict(color=color, size=7, symbol="circle-open", line=dict(width=2))),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=np.maximum(comparison.rel_error[i], 1e-16), mode="lines",
                                   name=f"{name} error", line=dict(color=color, width=2), showlegend=False),
                      row=2, col=1)
    fig.add_hline(y=comparison.tolerance, line=dict(color="#C0392B", dash="dash"), row=2, col=1,
                   annotation_text=f"tolerance {comparison.tolerance:g}", annotation_position="top left")
    fig.update_yaxes(type="log", row=2, col=1, title_text="|numeric \u2212 closed form| / |closed form|")
    fig.update_xaxes(title_text="t", row=2, col=1)
    fig.update_layout(title=f"Numerical integration and closed form {verdict} "
                              f"(max relative error {comparison.max_rel_error:.2e})",
                        height=560)
    return fig


def build_phase_trail_animation(dx_f, dy_f, x_range: tuple[float, float], y_range: tuple[float, float],
                                  trajectories: list[tuple[np.ndarray, np.ndarray]], times: np.ndarray,
                                  x_label: str = "x", y_label: str = "y", trail_length: int = 25,
                                  max_frames: int = 60, resolution: int = 16,
                                  highlight: int = 0) -> go.Figure:
    """The phase portrait's direction field with one or more points MOVING
    along their paths, each dragging a fading trail -- showing the flow, not
    just the one solved path. `trajectories` are equal-length (xs, ys)
    arrays sampled at `times` (NaN where a path has left the finite range or
    couldn't be integrated: those points are simply not drawn); trajectory
    `highlight` (the actual solved path) is drawn in blue, the rest in
    amber. Axis ranges are fixed so the view doesn't rescale mid-animation."""
    n = len(times)
    base = build_phase_portrait(dx_f, dy_f, x_range, y_range, x_label, y_label, resolution=resolution)
    fig = go.Figure(data=list(base.data))

    gap = np.array([np.nan])
    all_x = np.concatenate([np.concatenate([tx, gap]) for tx, _ in trajectories]) if trajectories else gap
    all_y = np.concatenate([np.concatenate([ty, gap]) for _, ty in trajectories]) if trajectories else gap
    fig.add_trace(go.Scatter(x=all_x, y=all_y, mode="lines", showlegend=False, hoverinfo="skip",
                               line=dict(color="rgba(120,120,120,0.35)", width=1.5)))

    def colour(j: int) -> str:
        return _PALETTE[0] if j == highlight else _PALETTE[4]

    def dynamic(i: int) -> list[go.Scatter]:
        trail_x: list[float] = []
        trail_y: list[float] = []
        trail_c: list[str] = []
        trail_s: list[float] = []
        head_x: list[float] = []
        head_y: list[float] = []
        head_c: list[str] = []
        for j, (tx, ty) in enumerate(trajectories):
            lo = max(0, i - trail_length + 1)
            seg_x, seg_y = tx[lo:i + 1], ty[lo:i + 1]
            ok = np.isfinite(seg_x) & np.isfinite(seg_y)
            m = int(ok.sum())
            if m:
                alphas = np.linspace(0.12, 0.85, m)
                trail_x += seg_x[ok].tolist()
                trail_y += seg_y[ok].tolist()
                trail_c += [_rgba(colour(j), a) for a in alphas]
                trail_s += np.linspace(3, 8, m).tolist()
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                head_x.append(float(tx[i]))
                head_y.append(float(ty[i]))
                head_c.append(colour(j))
        return [
            go.Scatter(x=trail_x, y=trail_y, mode="markers", showlegend=False, hoverinfo="skip",
                        marker=dict(color=trail_c, size=trail_s)),
            go.Scatter(x=head_x, y=head_y, mode="markers", showlegend=False,
                        marker=dict(color=head_c, size=13, line=dict(color="white", width=1.5))),
        ]

    idx = _frame_indices(n, max_frames)
    first = len(fig.data)
    fig.add_traces(dynamic(idx[0]))
    fig.frames = [go.Frame(data=dynamic(i), traces=[first, first + 1], name=str(k)) for k, i in enumerate(idx)]
    fig.update_layout(
        xaxis=dict(title=x_label, range=list(x_range)), yaxis=dict(title=y_label, range=list(y_range)),
        title=f"Phase portrait flow: {x_label}\u2013{y_label}", showlegend=False,
        **_play_layout([f"{times[i]:.3g}" for i in idx], "t = "))
    return fig


def build_time_linked_view(ts: np.ndarray, series: list[tuple[str, np.ndarray]], field=None,
                             max_frames: int = 60, resolution: int = 14) -> go.Figure:
    """Two panels sharing ONE time cursor: the left plots every function of
    time with a vertical cursor line, the right plots the first two against
    each other (the phase plane, optionally over its direction field) with a
    marker at the same instant -- so the slider scrubs both at once and the
    title reads off the values. `field` is an optional (dx_f, dy_f) pair of
    numpy-vectorized callables for the direction field."""
    from plotly.subplots import make_subplots
    if len(series) < 2:
        raise ValueError("The linked time view needs at least two functions (for the phase plane).")
    (name_x, xs), (name_y, ys) = series[0], series[1]
    n = len(ts)

    def padded(values: np.ndarray, frac: float = 0.2) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        lo, hi = float(finite.min()), float(finite.max())
        pad = frac * max(hi - lo, 1.0)
        return lo - pad, hi + pad

    x_range, y_range = padded(xs), padded(ys)
    all_vals = np.concatenate([v for _, v in series])
    v_lo, v_hi = padded(all_vals, 0.1)

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1,
                         subplot_titles=("Time series", f"Phase plane: {name_x}\u2013{name_y}"))
    for k, (name, vals) in enumerate(series):
        fig.add_trace(go.Scatter(x=ts, y=vals, mode="lines", name=name,
                                   line=dict(color=_PALETTE[k % len(_PALETTE)], width=2.5)), row=1, col=1)
    if field is not None:
        quiver = build_phase_portrait(field[0], field[1], x_range, y_range, name_x, name_y,
                                        resolution=resolution)
        for tr in quiver.data:
            fig.add_trace(tr, row=1, col=2)
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", showlegend=False, hoverinfo="skip",
                               line=dict(color="rgba(46,94,170,0.55)", width=3)), row=1, col=2)

    def dynamic(i: int) -> list[go.Scatter]:
        t_i = float(ts[i])
        return [
            go.Scatter(x=[t_i, t_i], y=[v_lo, v_hi], mode="lines", showlegend=False, hoverinfo="skip",
                        line=dict(color="rgba(60,60,60,0.7)", dash="dash", width=1.5)),
            go.Scatter(x=[t_i] * len(series), y=[float(v[i]) for _, v in series], mode="markers",
                        showlegend=False, marker=dict(size=11, color=[_PALETTE[k % len(_PALETTE)]
                                                                       for k in range(len(series))],
                                                       line=dict(color="white", width=1.5))),
            go.Scatter(x=[float(xs[i])], y=[float(ys[i])], mode="markers", showlegend=False,
                        marker=dict(size=13, color=_PALETTE[1], line=dict(color="white", width=1.5))),
        ]

    def title(i: int) -> str:
        return f"t = {ts[i]:.3g}   {name_x} = {xs[i]:.4g}   {name_y} = {ys[i]:.4g}"

    idx = _frame_indices(n, max_frames)
    first = len(fig.data)
    # the cursor line and the series markers live in the left panel, the phase marker in the right
    for tr, col in zip(dynamic(idx[0]), (1, 1, 2)):
        fig.add_trace(tr, row=1, col=col)
    fig.frames = [go.Frame(data=dynamic(i), traces=[first, first + 1, first + 2], name=str(k),
                             layout=go.Layout(title_text=title(i))) for k, i in enumerate(idx)]
    fig.update_xaxes(title_text="t", row=1, col=1)
    fig.update_yaxes(title_text="value", range=[v_lo, v_hi], row=1, col=1)
    fig.update_xaxes(title_text=name_x, range=list(x_range), row=1, col=2)
    fig.update_yaxes(title_text=name_y, range=list(y_range), row=1, col=2)
    fig.update_layout(title=title(idx[0]), **_play_layout([f"{ts[i]:.3g}" for i in idx], "t = "))
    return fig


def build_partial_sum_animation(partial_sums, title: str = "") -> go.Figure:
    """A function (black) with its series partial sums (red) added one term
    -- or harmonic -- at a time. `partial_sums` is a
    modules.series_animation.PartialSums. Axes are fixed from the TRUE
    function so a diverging polynomial runs off the plot instead of
    rescaling it; the title of each frame reports that frame's RMS error."""
    if partial_sums.error:
        raise ValueError(partial_sums.error)
    ps = partial_sums
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ps.xs, y=ps.target, mode="lines", name="function",
                               line=dict(color="#222222", width=3)))
    if ps.center is not None:
        fig.add_trace(go.Scatter(x=[ps.center, ps.center], y=list(ps.y_range), mode="lines",
                                   name="expansion point", line=dict(color="rgba(100,100,100,0.5)", dash="dot")))
    dyn = len(fig.data)
    fig.add_trace(go.Scatter(x=ps.xs, y=ps.sums[0], mode="lines", name="partial sum",
                               line=dict(color="#C0392B", width=2.5)))

    def frame_title(k: int) -> str:
        return f"{title + ' \u2014 ' if title else ''}{ps.labels[k]}   (RMS error {ps.errors[k]:.3g} {ps.error_region})"

    fig.frames = [go.Frame(data=[go.Scatter(x=ps.xs, y=ps.sums[k])], traces=[dyn], name=str(k),
                             layout=go.Layout(title_text=frame_title(k))) for k in range(len(ps.sums))]
    fig.update_layout(xaxis=dict(title="x", range=[float(ps.xs[0]), float(ps.xs[-1])]),
                        yaxis=dict(title="y", range=list(ps.y_range)), title=frame_title(0),
                        **_play_layout([str(k + 1) for k in range(len(ps.sums))], "step ", duration_ms=700))
    return fig


# ---------------------------------------------------------------------------
# Uncertainty over time, parameter morph, bifurcation diagram, space-time view
# ---------------------------------------------------------------------------
def build_uncertainty_fan(fan) -> go.Figure:
    """A fan chart from a modules.time_uncertainty.FanResult: for each
    function, the 5-95% and 25-75% bands of the sampled solutions around
    their median, the nominal curve (every input at its mean) dashed, and --
    if it was computed -- the guaranteed interval-arithmetic envelope dotted."""
    from plotly.subplots import make_subplots
    if not fan.applicable:
        raise ValueError(fan.reason or "No uncertainty fan available.")
    n = len(fan.names)
    fig = make_subplots(rows=n, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                         subplot_titles=fan.names if n > 1 else None)
    t = fan.t
    for i, name in enumerate(fan.names):
        row, colour, first = i + 1, _PALETTE[i % len(_PALETTE)], i == 0
        if fan.envelope_lo is not None and fan.envelope_hi is not None:
            for edge, label in ((fan.envelope_hi[i], f"guaranteed envelope (\u00b1{fan.envelope_sigmas:g}\u03c3)"),
                                (fan.envelope_lo[i], None)):
                fig.add_trace(go.Scatter(x=t, y=edge, mode="lines", name=label or "envelope",
                                           showlegend=bool(label) and first,
                                           line=dict(color="rgba(80,80,80,0.7)", width=1.5, dash="dot")),
                              row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.p95[i], mode="lines", line=dict(width=0), showlegend=False,
                                   hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.p5[i], mode="lines", line=dict(width=0), fill="tonexty",
                                   fillcolor=_rgba(colour, 0.18), name="5\u201395% of samples",
                                   showlegend=first), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.p75[i], mode="lines", line=dict(width=0), showlegend=False,
                                   hoverinfo="skip"), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.p25[i], mode="lines", line=dict(width=0), fill="tonexty",
                                   fillcolor=_rgba(colour, 0.35), name="25\u201375% of samples",
                                   showlegend=first), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.median[i], mode="lines", name="median", showlegend=first,
                                   line=dict(color=colour, width=2.5)), row=row, col=1)
        fig.add_trace(go.Scatter(x=t, y=fan.nominal[i], mode="lines", name="nominal (inputs at their means)",
                                   showlegend=first, line=dict(color="#222222", width=1.8, dash="dash")),
                      row=row, col=1)
        fig.update_yaxes(title_text=name, row=row, col=1)
    fig.update_xaxes(title_text="t", row=n, col=1)
    fig.update_layout(title=f"Uncertainty over time ({fan.n_samples} samples, seed {fan.seed})",
                        height=380 + 230 * (n - 1))
    return fig


def build_parameter_morph(morph) -> go.Figure:
    """The family of solution curves from a modules.parameter_morph.MorphResult
    as an animation: every curve faint in the background, the current
    parameter value's curve bold, the title naming the value and how many
    turning points that curve has."""
    if not morph.applicable:
        raise ValueError(morph.reason or "No morph available.")
    t, n = morph.t, len(morph.values)
    gap = np.array([np.nan])
    ghost_x = np.concatenate([np.concatenate([t, gap]) for _ in range(n)])
    ghost_y = np.concatenate([np.concatenate([row, gap]) for row in morph.curves])
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ghost_x, y=ghost_y, mode="lines", name="all values", hoverinfo="skip",
                               line=dict(color="rgba(120,120,120,0.28)", width=1.2)))
    if morph.nominal_curve is not None:
        fig.add_trace(go.Scatter(x=t, y=morph.nominal_curve, mode="lines",
                                   name=f"current setting ({morph.param_name} = {morph.nominal_value:g})",
                                   line=dict(color="#222222", width=2, dash="dash")))
    dyn = len(fig.data)
    fig.add_trace(go.Scatter(x=t, y=morph.curves[0], mode="lines", name=f"{morph.function}(t)",
                               line=dict(color="#C0392B", width=3.5)))

    def title(i: int) -> str:
        tp = morph.turning_points[i]
        shape = "monotone" if tp == 0 else f"{tp} turning point{'s' if tp != 1 else ''}"
        return f"{morph.param_name} = {morph.values[i]:.4g}  \u2014  {shape}"

    idx = _frame_indices(n, 60)
    fig.frames = [go.Frame(data=[go.Scatter(x=t, y=morph.curves[i])], traces=[dyn], name=str(k),
                             layout=go.Layout(title_text=title(i))) for k, i in enumerate(idx)]
    fig.update_layout(xaxis=dict(title="t"), yaxis=dict(title=morph.function, range=list(morph.y_range)),
                        title=title(idx[0]),
                        **_play_layout([f"{morph.values[i]:.3g}" for i in idx], f"{morph.param_name} = ",
                                       duration_ms=120))
    return fig


def build_bifurcation_plot(result, marker: float | None = None) -> go.Figure:
    """A modules.bifurcation.BifurcationResult as the classic diagram: the
    long-run values of the map against the parameter. `marker` draws a line
    at the parameter's current value (where the cobweb diagram is drawn);
    dotted lines mark where the period changes through the 2, 4, 8, 16
    doubling cascade."""
    if not result.applicable:
        raise ValueError(result.reason or "No bifurcation diagram available.")
    xs = np.tile(result.params, result.values.shape[0]).astype(np.float32)
    ys = result.values.ravel().astype(np.float32)
    keep = np.isfinite(ys)
    fig = go.Figure(go.Scattergl(x=xs[keep], y=ys[keep], mode="markers", name="long-run values",
                                   marker=dict(size=1.6, color="rgba(46,94,170,0.45)")))
    for p, period in result.transitions:
        if period in (2, 4, 8, 16):
            fig.add_vline(x=p, line=dict(color="rgba(192,57,43,0.55)", dash="dot", width=1),
                           annotation_text=f"period {period}", annotation_position="top",
                           annotation_font_size=9)
    if marker is not None and result.params[0] <= marker <= result.params[-1]:
        fig.add_vline(x=marker, line=dict(color="#1E7E34", dash="dash", width=2),
                       annotation_text="current", annotation_position="bottom right")
    fig.update_layout(xaxis=dict(title=result.param_name, range=[float(result.params[0]), float(result.params[-1])]),
                        yaxis=dict(title="long-run value"), showlegend=False,
                        title=f"Bifurcation diagram (x\u2080 = {result.x0:g})")
    return fig


def build_space_time_view(field, y_label: str = "u") -> go.Figure:
    """A modules.pde_field.PDEField as two linked panels sharing the x axis:
    the profile u(x, t) animated through time on top, and the whole
    evolution as a heatmap below with a cursor line at the current time."""
    from plotly.subplots import make_subplots
    if field.error:
        raise ValueError(field.error)
    xs, ts, u = field.xs, field.ts, field.u
    pad = 0.08 * max(field.u_max - field.u_min, 1e-9)
    signed = field.u_min < 0 < field.u_max
    bound = max(abs(field.u_min), abs(field.u_max))
    heat = dict(colorscale="RdBu", zmin=-bound, zmax=bound, zmid=0) if signed else \
        dict(colorscale="Inferno", zmin=field.u_min, zmax=field.u_max)

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.42, 0.58], vertical_spacing=0.1,
                         subplot_titles=(f"Profile {y_label}(x, t)", "Whole evolution"))
    fig.add_trace(go.Scatter(x=xs, y=u[0], mode="lines", line=dict(color="#C0392B", width=3), showlegend=False),
                  row=1, col=1)
    fig.add_trace(go.Heatmap(x=xs, y=ts, z=u, colorbar=dict(title=y_label, len=0.55, y=0.28), **heat),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=[float(xs[0]), float(xs[-1])], y=[float(ts[0])] * 2, mode="lines",
                               line=dict(color="white", width=2, dash="dash"), showlegend=False,
                               hoverinfo="skip"), row=2, col=1)

    def title(i: int) -> str:
        return (f"t = {ts[i]:.3g}   max|{y_label}| = {field.max_abs[i]:.4g}   "
                f"\u222b{y_label} dx = {field.integral[i]:.4g}")

    idx = _frame_indices(len(ts), 60)
    fig.frames = [go.Frame(data=[go.Scatter(x=xs, y=u[i]),
                                  go.Scatter(x=[float(xs[0]), float(xs[-1])], y=[float(ts[i])] * 2)],
                             traces=[0, 2], name=str(k), layout=go.Layout(title_text=title(i)))
                  for k, i in enumerate(idx)]
    fig.update_yaxes(title_text=y_label, range=[field.u_min - pad, field.u_max + pad], row=1, col=1)
    fig.update_yaxes(title_text="t", row=2, col=1)
    fig.update_xaxes(title_text="x", row=2, col=1)
    fig.update_layout(title=title(idx[0]), height=620,
                        **_play_layout([f"{ts[i]:.3g}" for i in idx], "t = ", duration_ms=90))
    return fig
