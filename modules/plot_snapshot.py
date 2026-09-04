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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 -- registers 3d projection

from modules.equation_engine import Equation


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


def snapshot_line_plot(eq: Equation, x_symbol: str, param_values: dict[str, float],
                         x_range: tuple[float, float], y_target: str | None = None,
                         x_log: bool = False, y_log: bool = False,
                       fmt: str = "png") -> bytes:
    x = sp.Symbol(x_symbol)
    if x_log:
        lo = max(x_range[0], 1e-6)
        xs = np.geomspace(lo, max(x_range[1], lo * 10), 400)
    else:
        xs = np.linspace(x_range[0], x_range[1], 400)
    subs = {sp.Symbol(k): v for k, v in param_values.items()}

    fig, ax = plt.subplots(figsize=(7, 4.2))
    if x_log:
        ax.set_xscale("log")
    if y_log:
        ax.set_yscale("log")

    if y_target and y_target != x_symbol:
        target = sp.Symbol(y_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify(x, solved[0][target], "numpy")
            ys = np.real(np.array(f(xs), dtype=complex))
            ax.plot(xs, ys, color="#2a9d8f", linewidth=2)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_target)
            ax.grid(alpha=0.3)
            return _finish(fig, fmt)

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify(x, residual, "numpy")
    ys = np.real(np.array(f(xs), dtype=complex))
    ax.plot(xs, ys, color="#2a9d8f", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(f"{eq.name} residual (0 = satisfied)")
    ax.grid(alpha=0.3)
    return _finish(fig, fmt)


def snapshot_surface_plot(eq: Equation, x_symbol: str, y_symbol: str,
                            param_values: dict[str, float],
                            x_range: tuple[float, float], y_range: tuple[float, float],
                            z_target: str | None = None, resolution: int = 50,
                       fmt: str = "png") -> bytes:
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    subs = {sp.Symbol(k): v for k, v in param_values.items()}

    fig = plt.figure(figsize=(7, 5.5))
    ax = fig.add_subplot(111, projection="3d")

    if z_target and z_target not in (x_symbol, y_symbol):
        target = sp.Symbol(z_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify((x, y), solved[0][target], "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex))
            if Z.shape != X.shape:
                Z = np.full_like(X, float(Z))
            ax.plot_surface(X, Y, Z, cmap="viridis", edgecolor="none", alpha=0.9)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_symbol)
            ax.set_zlabel(z_target)
            return _finish(fig, fmt)

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
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
        colors = plt.cm.tab10.colors
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
        colors = plt.cm.tab10.colors
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


def snapshot_contour_plot(eq: Equation, x_symbol: str, y_symbol: str,
                            param_values: dict[str, float],
                            x_range: tuple[float, float], y_range: tuple[float, float],
                            z_target: str | None = None, resolution: int = 60,
                            fmt: str = "png") -> bytes:
    """Static counterpart to plotter.build_contour_plot()."""
    x, y = sp.Symbol(x_symbol), sp.Symbol(y_symbol)
    xs = np.linspace(x_range[0], x_range[1], resolution)
    ys = np.linspace(y_range[0], y_range[1], resolution)
    X, Y = np.meshgrid(xs, ys)
    subs = {sp.Symbol(k): v for k, v in param_values.items()}

    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    if z_target and z_target not in (x_symbol, y_symbol):
        target = sp.Symbol(z_target)
        try:
            solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
        except Exception:  # noqa: BLE001
            solved = []
        if solved:
            f = sp.lambdify((x, y), solved[0][target], "numpy")
            Z = np.real(np.array(f(X, Y), dtype=complex))
            if Z.shape != X.shape:
                Z = np.full_like(X, float(Z))
            cs = ax.contour(X, Y, Z, cmap="viridis")
            ax.clabel(cs, inline=True, fontsize=8)
            ax.set_xlabel(x_symbol)
            ax.set_ylabel(y_symbol)
            return _finish(fig, fmt)

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
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
