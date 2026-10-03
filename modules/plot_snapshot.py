"""
Renders static PNG snapshots of a plot for embedding in exported reports
(Markdown/PDF), as an alternative to the interactive Plotly figures in
plotter.py.

Deliberately uses matplotlib rather than Plotly's fig.to_image(): as of
Plotly's current release, static image export requires the `kaleido`
package, and kaleido>=1.0 in turn requires a separately-installed Chrome
browser -- a real portability regression for an app whose whole design
goal is "pip install and go, no extra binaries" (the same reason equation
LaTeX is rendered via matplotlib's mathtext rather than requiring a system
LaTeX install). matplotlib is already a required dependency for that, so
reusing it here avoids adding kaleido+Chrome as a second, more fragile
path to the same kind of output.

These are static re-renders of the same underlying data, not literal
screenshots of the Plotly figure -- so they won't be pixel-identical to
what's on screen, but they show the same numbers.
"""
import io

import numpy as np
import sympy as sp
import matplotlib
from typing import cast
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection

from modules.equation_engine import Equation
from modules.plot_params import solve_for_target, residual_expression


def _finish(fig, fmt: str = "png") -> bytes:
    """fmt: "png" (default, raster -- what gets embedded in exported
    Markdown/PDF reports), "svg" or "pdf" (vector -- for dropping a
    figure directly into a paper/slide deck without the pixelation a
    raster image gets when scaled up). All three come for free from
    matplotlib's own savefig() -- no extra dependency needed, unlike
    Plotly's static export path (see this module's own docstring)."""
    if fmt not in ("png", "svg", "pdf"):
        raise ValueError(f"Unsupported format '{fmt}' -- use 'png', 'svg', or 'pdf'.")
    buf = io.BytesIO()
    fig.savefig(buf, format=fmt, dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
def snapshot_line_plot(eq: Equation, x_symbol: str, param_values: dict[str, float],
                         x_range: tuple[float, float], y_target: str | None = None,
                         x_log: bool = False, y_log: bool = False,
                       fmt: str = "png") -> bytes:
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x = sp.Symbol(x_symbol)
    if x_log:
        lo = max(x_range[0], 1e-6)
        xs = np.geomspace(lo, max(x_range[1], lo * 10), 400)
    else:
        xs = np.linspace(x_range[0], x_range[1], 400)

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    if y_target and y_target != x_symbol:
        solved_expr = solve_for_target(eq, y_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify(x, solved_expr, "numpy")
            ys = np.real(np.array(f(xs), dtype=complex))
            ax.plot(xs, ys, color="#2a9d8f", linewidth=2)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_target)
            ax.grid(alpha=0.3)
            return _finish(fig, fmt)

    residual = residual_expression(eq, param_values, {x_symbol})
    f = sp.lambdify(x, residual, "numpy")
    ys = np.real(np.array(f(xs), dtype=complex))
    ax.plot(xs, ys, color="#2a9d8f", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(f"{eq.name} residual (0 = satisfied)")
    ax.grid(alpha=0.3)
    return _finish(fig, fmt)


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
def snapshot_surface_plot(eq: Equation, x_symbol: str, y_symbol: str,
                            param_values: dict[str, float],
                            x_range: tuple[float, float], y_range: tuple[float, float],
                            z_target: str | None = None, resolution: int = 50,
                       fmt: str = "png") -> bytes:
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    if z_target and z_target not in (x_symbol, y_symbol):
        solved_expr = solve_for_target(eq, z_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify((x, y), solved_expr, "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex))
            if Z.shape != X.shape:
                Z = np.full_like(X, float(Z))
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_symbol)
            ax.set_zlabel(z_target)
            return _finish(fig, fmt)

    residual = residual_expression(eq, param_values, {x_symbol, y_symbol})
    f = sp.lambdify((x, y), residual, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex))
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))
    ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_zlabel(f"{eq.name} residual")
    return _finish(fig, fmt)


def snapshot_feasible_region(constraints: list[Equation], x_symbol: str, y_symbol: str,
                               param_values: dict[str, float],
                               x_range: tuple[float, float], y_range: tuple[float, float],
                               resolution: int = 300,
                       fmt: str = "png") -> bytes:
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    subs = {sp.Symbol(k): v for k, v in param_values.items()}

    feasible = np.ones_like(X, dtype=bool)
    for c in constraints:
        if c.sympy_eq is None:
            continue
        substituted = c.sympy_eq.subs(subs)
        if not substituted.free_symbols.issubset({x, y}):
            continue
        try:
            f = sp.lambdify((x, y), substituted, "numpy")
            feasible &= np.asarray(f(X, Y), dtype=bool)
        except Exception:  # noqa: BLE001
            continue

    fig, ax = plt.subplots(figsize=(6.5, 6))
    ax.imshow(feasible.astype(int), extent=(x_range[0], x_range[1], y_range[0], y_range[1]),
               origin="lower", aspect="auto", cmap="Greens", alpha=0.6)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_title("Shaded = every selected constraint satisfied simultaneously", fontsize=10)
    return _finish(fig, fmt)


def snapshot_ode_plot(func_name: str, indep_symbol: sp.Symbol, rhs_expr: sp.Expr,
                        t_range: tuple[float, float],
                       fmt: str = "png") -> bytes:
    xs = np.linspace(t_range[0], t_range[1], 300)
    f = sp.lambdify(indep_symbol, rhs_expr, "numpy")
    ys = np.real(np.array(f(xs), dtype=complex))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, color="#5e60ce", linewidth=2)
    ax.set_xlabel(str(indep_symbol))
    ax.set_ylabel(func_name)
    ax.grid(alpha=0.3)
    return _finish(fig, fmt)


def snapshot_recurrence_plot(func_name: str, indep_symbol: sp.Symbol, closed_form: sp.Expr,
                               n_range: tuple[int, int],
                       fmt: str = "png") -> bytes:
    """Discrete markers (a stem plot), not a connected line -- a recurrence
    is only defined at integer indices, so drawing a continuous curve
    through the points would visually imply values exist in between that
    the problem never actually defines."""
    ns = np.arange(n_range[0], n_range[1] + 1)
    f = sp.lambdify(indep_symbol, closed_form, "numpy")
    ys = np.real(np.array([complex(f(n)) for n in ns]))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.stem(ns, ys, basefmt=" ")
    ax.set_xlabel(str(indep_symbol))
    ax.set_ylabel(func_name)
    ax.grid(alpha=0.3)
    return _finish(fig, fmt)


def snapshot_vector_plot(vectors: list[tuple[str, list[float]]],
                       fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_vector_plot() -- arrows from
    the origin, 2D via matplotlib.quiver or 3D via Axes3D.quiver. All
    vectors passed together must share the same dimension (2 or 3)."""
    if not vectors:
        fig, ax = plt.subplots(figsize=(5, 5))
        return _finish(fig, fmt)
    dim = len(vectors[0][1])

    if dim == 2:
        fig, ax = plt.subplots(figsize=(5.5, 5.5))
        xs = [c[0] for _, c in vectors]
        ys = [c[1] for _, c in vectors]
        colors = cast(list, cast(ListedColormap, plt.get_cmap("tab10")).colors)
        for i, (name, comps) in enumerate(vectors):
            x, y = comps
            ax.quiver(0, 0, x, y, angles="xy", scale_units="xy", scale=1,
                       color=colors[i % len(colors)], label=name)
        span = max(1.0, max(abs(v) for v in xs + ys) * 1.3)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.axhline(0, color="gray", linewidth=0.8)
        ax.axvline(0, color="gray", linewidth=0.8)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.legend()
        return _finish(fig, fmt)

    if dim == 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        colors = cast(list, cast(ListedColormap, plt.get_cmap("tab10")).colors)
        all_vals = [c for _, comps in vectors for c in comps]
        span = max(1.0, max(abs(v) for v in all_vals) * 1.3)
        for i, (name, comps) in enumerate(vectors):
            x, y, z = comps
            ax.quiver(0, 0, 0, x, y, z, color=colors[i % len(colors)], label=name)
        ax.set_xlim(-span, span)
        ax.set_ylim(-span, span)
        ax.set_zlim(-span, span)
        ax.legend()
        return _finish(fig, fmt)

    raise ValueError(f"snapshot_vector_plot() only supports 2D or 3D vectors, got {dim}D")


def snapshot_fit_plot(xs: list[float], ys: list[float], fit_expr, x_label: str = "x", y_label: str = "y",
                       x_log: bool = False, y_log: bool = False,
                       fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_fit_plot()."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")
    ax.scatter(xs, ys, label="data", zorder=3)
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
            ax.plot(grid, yfit, color="C1", label="fit")
        except Exception:  # noqa: BLE001
            pass
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, fmt)


def snapshot_dependency_graph(nodes, edges,
                       fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_dependency_graph_plot()."""
    colors = {"known": "tab:green", "unknown": "tab:red", "equation": "tab:blue"}
    by_id = {n.id: n for n in nodes}
    fig, ax = plt.subplots(figsize=(6, max(3, 0.8 * max((n.y for n in nodes), default=0) + 2)))
    for edge in edges:
        src, dst = by_id[edge.source], by_id[edge.target]
        ax.plot([src.x, dst.x], [src.y, dst.y], color="0.7", linewidth=1.2, zorder=1)
    for kind, color in colors.items():
        kind_nodes = [n for n in nodes if n.kind == kind]
        if not kind_nodes:
            continue
        ax.scatter([n.x for n in kind_nodes], [n.y for n in kind_nodes],
                    s=900, color=color, zorder=2, label=kind.capitalize())
        for n in kind_nodes:
            ax.annotate(n.label, (n.x, n.y), ha="center", va="center", color="white",
                         fontsize=9, zorder=3)
    ax.invert_yaxis()
    ax.axis("off")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.08), ncol=3, frameon=False)
    return _finish(fig, fmt)


def snapshot_tornado_chart(entries,
                       fmt: str = "png") -> bytes:
    """Static tornado chart: horizontal bars showing the target's swing
    across each input's swept range, largest at top. `entries` is a
    list of sensitivity.TornadoEntry."""
    fig, ax = plt.subplots(figsize=(7, max(2.5, 0.5 * len(entries) + 1)))
    labels = [e.symbol for e in entries]
    lows = [min(e.low_target, e.high_target) for e in entries]
    highs = [max(e.low_target, e.high_target) for e in entries]
    y_pos = list(range(len(entries)))[::-1]
    for y, lo, hi in zip(y_pos, lows, highs):
        ax.barh(y, hi - lo, left=lo, height=0.6, color="C0", alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Target value across each input's swept range")
    ax.grid(alpha=0.3, axis="x")
    return _finish(fig, fmt)


def snapshot_sweep_chart(sweep_result,
                       fmt: str = "png") -> bytes:
    """Static counterpart of a single-input sweep line: target value vs
    the swept input's value, with the nominal point marked."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sweep_result.values, sweep_result.target_values, color="C0")
    if sweep_result.nominal_target is not None:
        ax.scatter([sweep_result.nominal_input], [sweep_result.nominal_target],
                    color="C1", zorder=3, label="nominal", s=60)
        ax.legend()
    ax.set_xlabel(sweep_result.symbol)
    ax.set_ylabel("target value")
    ax.grid(alpha=0.3)
    return _finish(fig, fmt)


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
def snapshot_contour_plot(eq: Equation, x_symbol: str, y_symbol: str,
                            param_values: dict[str, float],
                            x_range: tuple[float, float], y_range: tuple[float, float],
                            z_target: str | None = None, resolution: int = 60,
                            fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_contour_plot()."""
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if z_target and z_target not in (x_symbol, y_symbol):
        solved_expr = solve_for_target(eq, z_target, param_values)
        if solved_expr is not None:
            f = sp.lambdify((x, y), solved_expr, "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex))
            if Z.shape != X.shape:
                Z = np.full_like(X, float(Z))
            cs = ax.contour(X, Y, Z, cmap="viridis")
            ax.clabel(cs, inline=True, fontsize=8)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_symbol)
            return _finish(fig, fmt)

    residual = residual_expression(eq, param_values, {x_symbol, y_symbol})
    f = sp.lambdify((x, y), residual, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex))
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))
    cs = ax.contour(X, Y, Z, cmap="RdBu")
    ax.clabel(cs, inline=True, fontsize=8)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_title(f"Contours of {eq.name} residual (0 = satisfied)", fontsize=10)
    return _finish(fig, fmt)


def snapshot_histogram_plot(samples: list[float], target_symbol: str,
                              mean: float | None = None, p5: float | None = None,
                              p95: float | None = None,
                              fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_histogram_plot() -- the
    exportable version of a Monte Carlo uncertainty-propagation result."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(samples, bins=min(60, max(10, len(samples) // 20)), color="#2a9d8f", alpha=0.85)
    if mean is not None:
        ax.axvline(mean, color="black", linewidth=2, label="mean")
    if p5 is not None:
        ax.axvline(p5, color="gray", linestyle="--", linewidth=1, label="5th/95th pct")
    if p95 is not None:
        ax.axvline(p95, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(target_symbol)
    ax.set_ylabel("count")
    ax.grid(alpha=0.3)
    if mean is not None or p5 is not None:
        ax.legend()
    return _finish(fig, fmt)


def snapshot_spread_plot(values: list[float], target_symbol: str, fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_spread_plot() -- individual
    per-run points (jittered so overlapping values stay visible) plus a
    box summarizing the spread, for self_consistency.py's per-run
    numeric answers."""
    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(values, showfliers=False)
    jitter = (np.random.rand(len(values)) - 0.5) * 0.08
    ax.scatter(1 + jitter, values, color="#e76f51", zorder=3)
    ax.set_ylabel(target_symbol)
    ax.set_xticks([])
    ax.set_title(f"{target_symbol} across independent re-extraction runs", fontsize=10)
    ax.grid(alpha=0.3, axis="y")
    return _finish(fig, fmt)


def snapshot_overlay_plot(series: list[dict], x_label: str = "x", y_label: str = "y",
                            title: str | None = None, fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_overlay_plot() -- each entry
    in `series` is {"x": [...], "y": [...], "name": str}."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for s in series:
        ax.plot(s["x"], s["y"], label=s.get("name", ""))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, fmt)


def snapshot_chain_sweep_plot(sweep_rows: list[dict], swept_symbol: str,
                                step_labels: dict[int, str], fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_chain_sweep_plot()."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x_values = [row["value"] for row in sweep_rows]
    positions = sorted({pos for row in sweep_rows for pos in row["outputs"]})
    for pos in positions:
        y_values = [row["outputs"].get(pos) for row in sweep_rows]
        # matplotlib doesn't auto-gap on None the way Plotly's connectgaps=False
        # does -- mask them out explicitly instead of letting a None break the plot
        xy = [(x, y) for x, y in zip(x_values, y_values) if y is not None]
        if xy:
            xs_plot, ys_plot = zip(*xy)
            ax.plot(xs_plot, ys_plot, marker="o", label=step_labels.get(pos, f"step {pos + 1}"))
    ax.set_xlabel(swept_symbol)
    ax.set_ylabel("step output value")
    ax.grid(alpha=0.3)
    ax.legend()
    return _finish(fig, fmt)


def snapshot_sweep_heatmap(x_values: list, y_values: list, z_matrix, x_label: str, y_label: str,
                             target_symbol: str, fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_sweep_heatmap()."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    im = ax.imshow(z_matrix, aspect="auto", origin="lower", cmap="viridis",
                     extent=(min(x_values), max(x_values), min(y_values), max(y_values)))
    fig.colorbar(im, ax=ax, label=target_symbol)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{target_symbol} across {x_label} \u00d7 {y_label}", fontsize=10)
    return _finish(fig, fmt)


def _finish_gif(fig, update_fn, n_frames: int, fps: int = 12) -> bytes:
    """Renders an animated GIF via matplotlib.animation.FuncAnimation +
    PillowWriter. Pillow is already a hard dependency of this module (see
    its own docstring for why kaleido/Chrome is avoided for static export)
    and PillowWriter needs nothing beyond it -- no ffmpeg, no browser,
    which is exactly the "no extra binaries" bar the rest of this module
    holds to. `update_fn(frame_index)` mutates `fig`'s own Axes in place
    for that frame (typically via ax.clear() + redraw) and returns
    nothing; matplotlib redraws the whole figure each frame rather than
    blitting individual artists, which is simpler and plenty fast enough
    at the frame counts these animations actually use (tens, not hundreds).

    Writes to a temporary file rather than an in-memory buffer: matplotlib's
    Animation.save() resolves its `outfile` argument as a filesystem path
    (Path(outfile).parent.resolve(...)) before PillowWriter ever touches
    it, so a BytesIO -- which has no filesystem parent -- fails there
    before any actual writing happens. The temp file is read back into
    bytes and removed immediately after, so callers still see this as a
    plain in-memory bytes-producing function.
    """
    from matplotlib.animation import FuncAnimation, PillowWriter
    import tempfile
    import os
    anim = FuncAnimation(fig, lambda i: update_fn(i), frames=n_frames, blit=False)
    fd, path = tempfile.mkstemp(suffix=".gif")
    os.close(fd)
    try:
        anim.save(path, writer=PillowWriter(fps=fps))
        with open(path, "rb") as f:
            data = f.read()
    finally:
        plt.close(fig)
        os.remove(path)
    return data


@np.errstate(divide="ignore", invalid="ignore")  # a solved curve like a = 12/t is infinite at t=0; Plotly/matplotlib
# drop non-finite points cleanly, so numpy's warning is just log noise on every slider drag
def snapshot_rotating_surface_gif(eq: Equation, x_symbol: str, y_symbol: str,
                                     param_values: dict[str, float],
                                     x_range: tuple[float, float], y_range: tuple[float, float],
                                     z_target: str | None = None, resolution: int = 40,
                                     n_frames: int = 36, fps: int = 12) -> bytes:
    """The animated counterpart to snapshot_surface_plot() above: the same
    surface, orbited through a full revolution -- matplotlib's own
    equivalent of plotter.add_camera_rotation()'s Plotly camera sweep, for
    dropping into a report rather than only being interactive on-screen."""
    if eq.sympy_eq is None:
        raise ValueError(f"Equation {eq.name!r} has no parsed sympy expression to plot.")
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)

    z_label = z_target or f"{eq.name} residual"
    expr = None
    if z_target and z_target not in (x_symbol, y_symbol):
        expr = solve_for_target(eq, z_target, param_values)
    if expr is None:
        expr = residual_expression(eq, param_values, {x_symbol, y_symbol})

    f = sp.lambdify((x, y), expr, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex))
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))

    fig = plt.figure(figsize=(6.5, 5.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_zlabel(z_label)

    def update(i):
        ax.view_init(elev=25, azim=i * 360 / n_frames)

    return _finish_gif(fig, update, n_frames, fps)


def snapshot_motion_diagram_gif(t_values, x_values, v_values, x_label: str = "position",
                                   x_unit: str = "", t_unit: str = "",
                                   n_frames: int = 40, fps: int = 15, a_values=None,
                                   n_strobes: int = 0) -> bytes:
    """The animated counterpart to modules.plotter.build_motion_diagram():
    a dot moving along a track with a velocity arrow, synced to a
    position-vs-time trace underneath, as a GIF for a report rather than
    only interactive on-screen. `a_values` adds an acceleration arrow and
    `n_strobes` leaves faint ghost positions behind, each with its own
    velocity (and acceleration) arrow -- see build_motion_diagram."""
    t_values = np.asarray(t_values, dtype=float)
    x_values = np.asarray(x_values, dtype=float)
    v_values = np.asarray(v_values, dtype=float)
    n = len(t_values)
    idx = np.unique(np.linspace(0, n - 1, min(n_frames, n)).astype(int))
    x_min, x_max = float(np.min(x_values)), float(np.max(x_values))
    pad = 0.18 * max(x_max - x_min, 1.0)
    v_max = max(abs(float(np.max(v_values))), abs(float(np.min(v_values))), 1e-9)
    v_scale = 0.12 * max(x_max - x_min, 1.0) / v_max
    x_unit_sfx = f" ({x_unit})" if x_unit else ""
    t_unit_sfx = f" ({t_unit})" if t_unit else ""
    show_a = a_values is not None
    a_arr = np.asarray(a_values, dtype=float) if show_a else None
    a_scale = (0.12 * max(x_max - x_min, 1.0) / max(float(np.max(np.abs(a_arr))), 1e-9)) if a_arr is not None else 0.0
    strobes = [int(j) for j in (np.unique(np.linspace(0, n - 1, min(n_strobes, n)).astype(int))
                                if n_strobes > 0 else [])]
    offset = 0.42            # arrow height above/below the track (the track axis runs -1..1)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6.5, 6), gridspec_kw={"height_ratios": [1, 3]})

    def arrow(ax, x0, value, scale, y, colour, width=2):
        ax.annotate("", xy=(x0 + value * scale, y), xytext=(x0, y),
                     arrowprops=dict(arrowstyle="-|>", color=colour, linewidth=width))

    def update(k):
        i = int(idx[k])
        ax1.clear()
        ax2.clear()
        xi, vi = float(x_values[i]), float(v_values[i])
        ax1.axhline(0, color="lightgray", linewidth=2)
        for j in strobes:
            if j <= i:
                xj = float(x_values[j])
                ax1.plot(xj, 0, "o", color="#2E5EAA", alpha=0.3, markersize=12, zorder=2)
                arrow(ax1, xj, float(v_values[j]), v_scale, offset, "#C0392B", 1.4)
                if a_arr is not None:
                    arrow(ax1, xj, float(a_arr[j]), a_scale, -offset, "#1E7E34", 1.4)
        ax1.plot(xi, 0, "o", color="#2E5EAA", markersize=16, zorder=3)
        arrow(ax1, xi, vi, v_scale, 0, "#C0392B")
        if a_arr is not None:
            arrow(ax1, xi, float(a_arr[i]), a_scale, -offset, "#1E7E34", 2.4)
        ax1.set_xlim(x_min - pad, x_max + pad)
        ax1.set_ylim(-1, 1)
        ax1.set_yticks([])
        ax1.set_xlabel(f"position{x_unit_sfx}")
        if show_a or strobes:
            ax1.set_title("red: velocity" + (", green: acceleration" if show_a else "")
                           + "  (each scaled separately)", fontsize=8)

        ax2.plot(t_values[: i + 1], x_values[: i + 1], color="#2E5EAA", linewidth=2)
        ax2.plot(t_values[i], x_values[i], "o", color="#C0392B", markersize=9, zorder=3)
        ax2.set_xlim(float(t_values[0]), float(t_values[-1]))
        ax2.set_ylim(x_min - pad, x_max + pad)
        ax2.set_xlabel(f"time{t_unit_sfx}")
        ax2.set_ylabel(f"{x_label}{x_unit_sfx}")
        ax2.grid(alpha=0.3)
        fig.tight_layout()

    return _finish_gif(fig, update, len(idx), fps)


def snapshot_cobweb_gif(g, x0: float, x_range: tuple[float, float], n_steps: int = 25,
                           x_label: str = "a(n)", fps: int = 6) -> bytes:
    """The animated counterpart to modules.plotter.build_cobweb_plot(): the
    staircase drawn one zig-zag segment at a time, as a GIF."""
    curve_xs = np.linspace(x_range[0], x_range[1], 300)
    curve_ys = np.asarray(g(curve_xs), dtype=float)

    xs = [x0]
    for _ in range(n_steps):
        xs.append(float(np.real(g(xs[-1]))))
    path_x, path_y = [xs[0]], [xs[0]]
    for i in range(n_steps):
        path_x += [xs[i], xs[i + 1]]
        path_y += [xs[i + 1], xs[i + 1]]

    fig, ax = plt.subplots(figsize=(5.5, 5.5))

    def update(k):
        ax.clear()
        ax.plot(curve_xs, curve_ys, color="#2E5EAA", linewidth=2, label="y = g(x)")
        ax.plot(curve_xs, curve_xs, color="gray", linestyle="--", linewidth=1.3, label="y = x")
        ax.plot(path_x[: 2 * k + 1], path_y[: 2 * k + 1], color="#C0392B", linewidth=1.6)
        ax.set_xlim(x_range)
        ax.set_ylim(x_range)
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"g({x_label})")
        ax.legend(loc="upper left", fontsize=8)
        ax.grid(alpha=0.3)
        fig.tight_layout()

    return _finish_gif(fig, update, n_steps + 1, fps)


# ---------------------------------------------------------------------------
# Time-resolved views (matplotlib counterparts of plotter.build_ode_error_plot,
# build_phase_trail_animation, build_time_linked_view and
# build_partial_sum_animation -- PNG for the error plot, GIF for the three
# animations, same no-extra-binaries approach as the GIFs above).
# ---------------------------------------------------------------------------
_TIME_PALETTE = ["#2E5EAA", "#C0392B", "#1E7E34", "#8E44AD", "#D68910", "#117A8B"]


def _frame_picks(n: int, n_frames: int) -> list[int]:
    """At most `n_frames` evenly spaced indices into a length-`n` series."""
    return sorted({int(round(v)) for v in np.linspace(0, n - 1, min(n, n_frames))})


def _draw_field(ax, dx_f, dy_f, x_range, y_range, resolution: int = 16) -> None:
    """A normalized direction field -- every arrow the same length, so only
    direction (not speed) is shown, matching plotter.build_phase_portrait."""
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    U = np.nan_to_num(np.asarray(dx_f(X, Y), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    V = np.nan_to_num(np.asarray(dy_f(X, Y), dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    speed = np.sqrt(U ** 2 + V ** 2)
    speed[speed == 0] = 1.0
    step = max((x_range[1] - x_range[0]) / resolution, (y_range[1] - y_range[0]) / resolution)
    ax.quiver(X, Y, U / speed, V / speed, color="#9AA3B2", angles="xy", scale_units="xy",
               scale=1.0 / (step * 0.8), pivot="mid", width=0.003)


@np.errstate(divide="ignore", invalid="ignore")
def snapshot_ode_error_plot(comparison, fmt: str = "png") -> bytes:
    """Closed form vs numerical integration, with relative error underneath
    (log scale) against the verification tolerance."""
    if not comparison.applicable:
        raise ValueError(comparison.reason or "No comparison available.")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(7.5, 6.2), sharex=True,
                                     gridspec_kw={"height_ratios": [1.3, 1]})
    t = comparison.t
    every = max(1, len(t) // 25)
    for i, name in enumerate(comparison.names):
        colour = _TIME_PALETTE[i % len(_TIME_PALETTE)]
        ax1.plot(t, comparison.symbolic[i], color=colour, linewidth=2.2, label=f"{name} (closed form)")
        ax1.plot(t[::every], comparison.numeric[i][::every], linestyle="none", marker="o", markersize=6,
                  markerfacecolor="none", markeredgecolor=colour, markeredgewidth=1.6,
                  label=f"{name} (numerical)")
        ax2.semilogy(t, np.maximum(comparison.rel_error[i], 1e-16), color=colour, linewidth=1.8)
    ax2.axhline(comparison.tolerance, color="#C0392B", linestyle="--", linewidth=1.3)
    ax2.text(t[0], comparison.tolerance / 1.6, f"tolerance {comparison.tolerance:g}", color="#C0392B",
              fontsize=8, va="top")
    ax1.set_ylabel("value")
    ax1.legend(fontsize=8, loc="best")
    ax1.grid(alpha=0.3)
    ax2.set_xlabel("t")
    ax2.set_ylabel("relative error")
    ax2.grid(alpha=0.3, which="both")
    verdict = "agree" if comparison.ok else "DISAGREE"
    fig.suptitle(f"Numerical integration and closed form {verdict} "
                  f"(max relative error {comparison.max_rel_error:.2e})", fontsize=10)
    fig.tight_layout()
    return _finish(fig, fmt)


def snapshot_phase_trail_gif(dx_f, dy_f, x_range: tuple[float, float], y_range: tuple[float, float],
                               trajectories: list[tuple[np.ndarray, np.ndarray]], times: np.ndarray,
                               x_label: str = "x", y_label: str = "y", trail_length: int = 25,
                               n_frames: int = 40, fps: int = 10, highlight: int = 0) -> bytes:
    """GIF of plotter.build_phase_trail_animation: points moving along their
    paths over the direction field, each with a fading trail."""
    from matplotlib.colors import to_rgba
    picks = _frame_picks(len(times), n_frames)
    fig, ax = plt.subplots(figsize=(5.8, 5.5))

    def update(k):
        i = picks[k]
        ax.clear()
        _draw_field(ax, dx_f, dy_f, x_range, y_range)
        for j, (tx, ty) in enumerate(trajectories):
            colour = _TIME_PALETTE[0] if j == highlight else _TIME_PALETTE[4]
            ax.plot(tx, ty, color="gray", alpha=0.3, linewidth=1.2)
            lo = max(0, i - trail_length + 1)
            seg_x, seg_y = tx[lo:i + 1], ty[lo:i + 1]
            ok = np.isfinite(seg_x) & np.isfinite(seg_y)
            m = int(ok.sum())
            if m:
                rgba = [to_rgba(colour, a) for a in np.linspace(0.12, 0.85, m)]
                ax.scatter(seg_x[ok], seg_y[ok], s=np.linspace(8, 40, m), c=rgba, zorder=3)
            if np.isfinite(tx[i]) and np.isfinite(ty[i]):
                ax.scatter([tx[i]], [ty[i]], s=90, color=colour, edgecolors="white", linewidths=1.2, zorder=4)
        ax.set_xlim(x_range)
        ax.set_ylim(y_range)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_title(f"Phase portrait flow    t = {times[i]:.3g}", fontsize=10)
        ax.grid(alpha=0.25)
        fig.tight_layout()

    return _finish_gif(fig, update, len(picks), fps)


def snapshot_time_linked_gif(ts: np.ndarray, series: list[tuple[str, np.ndarray]], field=None,
                               n_frames: int = 40, fps: int = 10) -> bytes:
    """GIF of plotter.build_time_linked_view: one moving instant shown on the
    time series (vertical cursor) and the phase plane (marker) together."""
    if len(series) < 2:
        raise ValueError("The linked time view needs at least two functions (for the phase plane).")
    (name_x, xs), (name_y, ys) = series[0], series[1]
    picks = _frame_picks(len(ts), n_frames)

    def padded(values: np.ndarray, frac: float) -> tuple[float, float]:
        finite = values[np.isfinite(values)]
        lo, hi = float(finite.min()), float(finite.max())
        pad = frac * max(hi - lo, 1.0)
        return lo - pad, hi + pad

    x_range, y_range = padded(xs, 0.2), padded(ys, 0.2)
    v_lo, v_hi = padded(np.concatenate([v for _, v in series]), 0.1)
    fig, (ax_t, ax_p) = plt.subplots(1, 2, figsize=(10, 4.8), gridspec_kw={"width_ratios": [1.2, 1]})

    def update(k):
        i = picks[k]
        ax_t.clear()
        ax_p.clear()
        for n, (name, vals) in enumerate(series):
            colour = _TIME_PALETTE[n % len(_TIME_PALETTE)]
            ax_t.plot(ts, vals, color=colour, linewidth=2, label=name)
            ax_t.scatter([ts[i]], [vals[i]], s=70, color=colour, edgecolors="white", zorder=4)
        ax_t.axvline(ts[i], color="#444444", linestyle="--", linewidth=1.2)
        ax_t.set_ylim(v_lo, v_hi)
        ax_t.set_xlabel("t")
        ax_t.set_ylabel("value")
        ax_t.legend(fontsize=8, loc="best")
        ax_t.grid(alpha=0.3)
        if field is not None:
            _draw_field(ax_p, field[0], field[1], x_range, y_range, resolution=14)
        ax_p.plot(xs, ys, color=_TIME_PALETTE[0], alpha=0.55, linewidth=2.5)
        ax_p.scatter([xs[i]], [ys[i]], s=90, color=_TIME_PALETTE[1], edgecolors="white", zorder=4)
        ax_p.set_xlim(x_range)
        ax_p.set_ylim(y_range)
        ax_p.set_xlabel(name_x)
        ax_p.set_ylabel(name_y)
        ax_p.grid(alpha=0.3)
        fig.suptitle(f"t = {ts[i]:.3g}    {name_x} = {xs[i]:.4g}    {name_y} = {ys[i]:.4g}", fontsize=10)
        fig.tight_layout()

    return _finish_gif(fig, update, len(picks), fps)


def snapshot_partial_sum_gif(partial_sums, title: str = "", fps: int = 2) -> bytes:
    """GIF of plotter.build_partial_sum_animation: the function with its
    series partial sums added one term (or harmonic) at a time."""
    if partial_sums.error:
        raise ValueError(partial_sums.error)
    ps = partial_sums
    fig, ax = plt.subplots(figsize=(7, 4.6))

    def update(k):
        ax.clear()
        ax.plot(ps.xs, ps.target, color="#222222", linewidth=2.6, label="function")
        if ps.center is not None:
            ax.axvline(ps.center, color="gray", linestyle=":", linewidth=1.2)
        ax.plot(ps.xs, ps.sums[k], color="#C0392B", linewidth=2.2, label="partial sum")
        ax.set_xlim(float(ps.xs[0]), float(ps.xs[-1]))
        ax.set_ylim(ps.y_range)
        ax.set_xlabel("x")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(alpha=0.3)
        head = f"{title} \u2014 " if title else ""
        ax.set_title(f"{head}{ps.labels[k]}\nRMS error {ps.errors[k]:.3g} {ps.error_region}", fontsize=9)
        fig.tight_layout()

    return _finish_gif(fig, update, len(ps.sums), fps)


# ---------------------------------------------------------------------------
# Matplotlib counterparts of plotter.build_uncertainty_fan / build_parameter_morph /
# build_bifurcation_plot / build_space_time_view
# ---------------------------------------------------------------------------
@np.errstate(invalid="ignore")
def snapshot_uncertainty_fan(fan, fmt: str = "png") -> bytes:
    """PNG (or SVG/PDF) of a modules.time_uncertainty.FanResult."""
    if not fan.applicable:
        raise ValueError(fan.reason or "No uncertainty fan available.")
    n = len(fan.names)
    fig, axes = plt.subplots(n, 1, figsize=(7.5, 3.4 * n), sharex=True, squeeze=False)
    for i, name in enumerate(fan.names):
        ax, colour = axes[i][0], _TIME_PALETTE[i % len(_TIME_PALETTE)]
        if fan.envelope_lo is not None and fan.envelope_hi is not None:
            ax.plot(fan.t, fan.envelope_hi[i], color="#555555", linestyle=":", linewidth=1.3,
                     label=f"guaranteed envelope (\u00b1{fan.envelope_sigmas:g}\u03c3)")
            ax.plot(fan.t, fan.envelope_lo[i], color="#555555", linestyle=":", linewidth=1.3)
        ax.fill_between(fan.t, fan.p5[i], fan.p95[i], color=colour, alpha=0.18, linewidth=0,
                         label="5\u201395% of samples")
        ax.fill_between(fan.t, fan.p25[i], fan.p75[i], color=colour, alpha=0.35, linewidth=0,
                         label="25\u201375% of samples")
        ax.plot(fan.t, fan.median[i], color=colour, linewidth=2.2, label="median")
        ax.plot(fan.t, fan.nominal[i], color="#222222", linestyle="--", linewidth=1.5,
                 label="nominal (inputs at their means)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.3)
        if i == 0:
            ax.legend(fontsize=8, loc="best")
    axes[-1][0].set_xlabel("t")
    fig.suptitle(f"Uncertainty over time ({fan.n_samples} samples, seed {fan.seed})", fontsize=10)
    fig.tight_layout()
    return _finish(fig, fmt)


def snapshot_parameter_morph_gif(morph, n_frames: int = 40, fps: int = 8) -> bytes:
    """GIF of plotter.build_parameter_morph: the family of curves, one
    parameter value's curve highlighted at a time."""
    if not morph.applicable:
        raise ValueError(morph.reason or "No morph available.")
    picks = _frame_picks(len(morph.values), n_frames)
    fig, ax = plt.subplots(figsize=(7, 4.4))

    def update(k):
        i = picks[k]
        ax.clear()
        for row in morph.curves:
            ax.plot(morph.t, row, color="gray", alpha=0.2, linewidth=1)
        if morph.nominal_curve is not None:
            ax.plot(morph.t, morph.nominal_curve, color="#222222", linestyle="--", linewidth=1.6,
                     label=f"current setting ({morph.param_name} = {morph.nominal_value:g})")
            ax.legend(fontsize=8, loc="upper right")
        ax.plot(morph.t, morph.curves[i], color="#C0392B", linewidth=3)
        ax.set_ylim(morph.y_range)
        ax.set_xlabel("t")
        ax.set_ylabel(morph.function)
        ax.grid(alpha=0.3)
        tp = morph.turning_points[i]
        shape = "monotone" if tp == 0 else f"{tp} turning point{'s' if tp != 1 else ''}"
        ax.set_title(f"{morph.param_name} = {morph.values[i]:.4g}  \u2014  {shape}", fontsize=10)
        fig.tight_layout()

    return _finish_gif(fig, update, len(picks), fps)


def snapshot_bifurcation_plot(result, marker: float | None = None, fmt: str = "png") -> bytes:
    """PNG (or SVG/PDF) of a modules.bifurcation.BifurcationResult."""
    if not result.applicable:
        raise ValueError(result.reason or "No bifurcation diagram available.")
    xs = np.tile(result.params, result.values.shape[0])
    ys = result.values.ravel()
    keep = np.isfinite(ys)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(xs[keep], ys[keep], s=0.4, color="#2E5EAA", alpha=0.45, linewidths=0, rasterized=True)
    for p, period in result.transitions:
        if period in (2, 4, 8, 16):
            ax.axvline(p, color="#C0392B", linestyle=":", linewidth=0.9, alpha=0.7)
            ax.text(p, ax.get_ylim()[1], f"{period}", color="#C0392B", fontsize=7, ha="center", va="bottom")
    if marker is not None and result.params[0] <= marker <= result.params[-1]:
        ax.axvline(marker, color="#1E7E34", linestyle="--", linewidth=1.8, label="current")
        ax.legend(fontsize=8, loc="upper left")
    ax.set_xlim(float(result.params[0]), float(result.params[-1]))
    ax.set_xlabel(result.param_name)
    ax.set_ylabel("long-run value")
    ax.set_title(f"Bifurcation diagram (x\u2080 = {result.x0:g})", fontsize=10)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    return _finish(fig, fmt)


def snapshot_space_time_gif(field, y_label: str = "u", n_frames: int = 40, fps: int = 10) -> bytes:
    """GIF of plotter.build_space_time_view: the profile on top and the
    whole evolution below, with a cursor at the current time."""
    if field.error:
        raise ValueError(field.error)
    picks = _frame_picks(len(field.ts), n_frames)
    pad = 0.08 * max(field.u_max - field.u_min, 1e-9)
    signed = field.u_min < 0 < field.u_max
    bound = max(abs(field.u_min), abs(field.u_max))
    fig, (ax_p, ax_h) = plt.subplots(2, 1, figsize=(7, 6.4), sharex=True, gridspec_kw={"height_ratios": [1, 1.3]})
    extent = [float(field.xs[0]), float(field.xs[-1]), float(field.ts[0]), float(field.ts[-1])]
    image = ax_h.imshow(field.u, origin="lower", aspect="auto", extent=extent,
                         cmap="RdBu" if signed else "inferno",
                         vmin=-bound if signed else field.u_min, vmax=bound if signed else field.u_max)
    fig.colorbar(image, ax=[ax_p, ax_h], label=y_label, fraction=0.04)
    ax_h.set_xlabel("x")
    ax_h.set_ylabel("t")
    cursor = ax_h.axhline(float(field.ts[0]), color="white", linestyle="--", linewidth=1.6)

    def update(k):
        i = picks[k]
        ax_p.clear()
        ax_p.plot(field.xs, field.u[i], color="#C0392B", linewidth=2.6)
        ax_p.set_ylim(field.u_min - pad, field.u_max + pad)
        ax_p.set_ylabel(y_label)
        ax_p.grid(alpha=0.3)
        cursor.set_ydata([field.ts[i], field.ts[i]])
        ax_p.set_title(f"t = {field.ts[i]:.3g}   max|{y_label}| = {field.max_abs[i]:.4g}   "
                       f"\u222b{y_label} dx = {field.integral[i]:.4g}", fontsize=9)

    return _finish_gif(fig, update, len(picks), fps)


# ---------------------------------------------------------------------------
# Solve-order replay and chain value flow (matplotlib GIFs)
# ---------------------------------------------------------------------------
_CURRENT_RING = "#F39C12"      # the stage being shown (matches plotter._CURRENT_RING)
_BLOCKED_RING = "#C0392B"


def snapshot_solve_order_gif(nodes, edges, frames, fps: int = 1) -> bytes:
    """GIF of plotter.build_solve_order_replay: the dependency graph lit up
    stage by stage in the order its quantities get determined."""
    from matplotlib.colors import to_rgba
    if not frames:
        raise ValueError("Nothing to replay.")
    base = {"known": "#1E7E34", "unknown": "#C0392B", "equation": "#2E5EAA"}
    alpha = {"known": 1.0, "done": 1.0, "current": 1.0, "pending": 0.16, "blocked": 0.45}
    by_id = {n.id: n for n in nodes}
    fig, ax = plt.subplots(figsize=(7, max(3.4, 0.9 * max((n.y for n in nodes), default=0) + 2.2)))

    def update(k):
        frame = frames[k]
        ax.clear()
        for e in edges:
            a, b = by_id[e.source], by_id[e.target]
            ax.plot([a.x, b.x], [a.y, b.y], color="0.8", linewidth=1.2, zorder=1)
        for s, t in frame.active_edges:
            ax.plot([by_id[s].x, by_id[t].x], [by_id[s].y, by_id[t].y], color=_CURRENT_RING, linewidth=4.5, zorder=2)
        for n in nodes:
            state = frame.states[n.id]
            ring = _CURRENT_RING if state == "current" else _BLOCKED_RING if state == "blocked" else "white"
            ax.scatter([n.x], [n.y], s=1500 if state == "current" else 1000,
                        color=to_rgba(base[n.kind], alpha[state]), edgecolors=ring,
                        linewidths=3.5 if state in ("current", "blocked") else 1, zorder=3)
            ax.annotate(n.label, (n.x, n.y), ha="center", va="center", color="white", fontsize=9, zorder=4)
        ax.set_xlim(-0.5, 2.5)
        ax.invert_yaxis()
        ax.axis("off")
        ax.set_title(frame.title, fontsize=9, wrap=True)
        fig.tight_layout()

    return _finish_gif(fig, update, len(frames), fps)


def snapshot_chain_flow_gif(flow, fps: int = 1) -> bytes:
    """GIF of plotter.build_chain_flow_plot: the chain's steps appearing one
    at a time with the values they carry forward."""
    from modules.chain_flow import step_caption
    if not flow.steps:
        raise ValueError("This chain has no steps.")
    pos = {s.position: s for s in flow.steps}
    order = sorted(pos)
    colour = {"ok": "#2E5EAA", "error": "#C0392B", "stale": "#8A8F98"}
    xs = [s.x for s in flow.steps]
    ys = [s.y for s in flow.steps] + [lit.y for lit in flow.literals]
    fig, ax = plt.subplots(figsize=(max(6.0, 1.9 * len(order)), 4.4))

    def update(i):
        k = order[i]
        ax.clear()
        for e in flow.edges:
            if e.target > k or e.source not in pos:
                continue
            a, b = pos[e.source], pos[e.target]
            ax.annotate("", xy=(b.x, b.y), xytext=(a.x, a.y), zorder=1,
                         arrowprops=dict(arrowstyle="-|>", color="#C0392B" if e.broken else "#2E5EAA",
                                         linewidth=2.6, linestyle="--" if e.broken else "-", shrinkA=26, shrinkB=26))
            ax.text((a.x + b.x) / 2, (a.y + b.y) / 2 + 0.22,
                    f"{e.symbol}: {e.reason}" if e.broken else f"{e.symbol} = {e.carried:.4g}",
                    ha="center", fontsize=8, color="#1B2A41")
        for lit in flow.literals:
            if lit.step <= k:
                s = pos[lit.step]
                ax.plot([lit.x, s.x], [lit.y, s.y], color="0.7", linewidth=1.2, zorder=1)
                ax.scatter([lit.x], [lit.y], marker="s", s=70, color="#8A8F98", zorder=2)
                ax.text(lit.x, lit.y + 0.12, f"{lit.symbol} = {lit.value:.4g}", ha="center", fontsize=8)
        for s in flow.steps:
            if s.position <= k:
                ring = _CURRENT_RING if s.position == k else "white"
                ax.scatter([s.x], [s.y], s=2800, color=colour.get(s.status, "#8A8F98"), edgecolors=ring,
                            linewidths=4 if s.position == k else 1, zorder=3)
                text = f"Step {s.position + 1}\n" + (f"{s.symbol} = {s.value:.4g}"
                                                      if s.status == "ok" and s.value is not None else s.symbol)
                ax.annotate(text, (s.x, s.y), ha="center", va="center", color="white", fontsize=8, zorder=4)
        ax.set_xlim(min(xs) - 1.4, max(xs) + 1.4)
        ax.set_ylim(min(ys) - 0.9, max(ys) + 0.9)
        ax.axis("off")
        ax.set_title(step_caption(flow, k), fontsize=8, wrap=True)
        fig.tight_layout()

    return _finish_gif(fig, update, len(order), fps)
