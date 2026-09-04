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
