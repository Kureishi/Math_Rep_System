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


def _finish(fig) -> bytes:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


def snapshot_line_plot(eq: Equation, x_symbol: str, param_values: dict[str, float],
                         x_range: tuple[float, float], y_target: str | None = None) -> bytes:
    x = sp.Symbol(x_symbol)
    xs = np.linspace(x_range[0], x_range[1], 400)
    subs = {sp.Symbol(k): v for k, v in param_values.items()}

    fig, ax = plt.subplots(figsize=(7, 4.2))

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
            return _finish(fig)

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify(x, residual, "numpy")
    ys = np.real(np.array(f(xs), dtype=complex))
    ax.plot(xs, ys, color="#2a9d8f", linewidth=2)
    ax.axhline(0, color="gray", linestyle="--", linewidth=1)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(f"{eq.name} residual (0 = satisfied)")
    ax.grid(alpha=0.3)
    return _finish(fig)


def snapshot_surface_plot(eq: Equation, x_symbol: str, y_symbol: str,
                            param_values: dict[str, float],
                            x_range: tuple[float, float], y_range: tuple[float, float],
                            z_target: str | None = None, resolution: int = 50) -> bytes:
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
            return _finish(fig)

    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    f = sp.lambdify((x, y), residual, "numpy")
    Z = np.real(np.array(f(X, Y), dtype=complex))
    if Z.shape != X.shape:
        Z = np.full_like(X, float(Z))
    ax.plot_surface(X, Y, Z, cmap="coolwarm", edgecolor="none", alpha=0.9)
    ax.set_xlabel(x_symbol)
    ax.set_ylabel(y_symbol)
    ax.set_zlabel(f"{eq.name} residual")
    return _finish(fig)


def snapshot_feasible_region(constraints: list[Equation], x_symbol: str, y_symbol: str,
                               param_values: dict[str, float],
                               x_range: tuple[float, float], y_range: tuple[float, float],
                               resolution: int = 300) -> bytes:
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
    return _finish(fig)


def snapshot_ode_plot(func_name: str, indep_symbol: sp.Symbol, rhs_expr: sp.Expr,
                        t_range: tuple[float, float]) -> bytes:
    xs = np.linspace(t_range[0], t_range[1], 300)
    f = sp.lambdify(indep_symbol, rhs_expr, "numpy")
    ys = np.real(np.array(f(xs), dtype=complex))

    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(xs, ys, color="#5e60ce", linewidth=2)
    ax.set_xlabel(str(indep_symbol))
    ax.set_ylabel(func_name)
    ax.grid(alpha=0.3)
    return _finish(fig)


def snapshot_recurrence_plot(func_name: str, indep_symbol: sp.Symbol, closed_form: sp.Expr,
                               n_range: tuple[int, int]) -> bytes:
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
    return _finish(fig)


def snapshot_vector_plot(vectors: list[tuple[str, list[float]]]) -> bytes:
    """Static counterpart to plotter.build_vector_plot() -- arrows from
    the origin, 2D via matplotlib.quiver or 3D via Axes3D.quiver. All
    vectors passed together must share the same dimension (2 or 3)."""
    if not vectors:
        fig, ax = plt.subplots(figsize=(5, 5))
        return _finish(fig)
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
        return _finish(fig)

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
        return _finish(fig)

    raise ValueError(f"snapshot_vector_plot() only supports 2D or 3D vectors, got {dim}D")
