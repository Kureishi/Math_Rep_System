"""
A PDE solution u(x, t) sampled on a grid: the data behind a space-time view.

The heat and wave solvers in modules.pde_utils return u(x, t) as a symbolic
(truncated Fourier series) expression. This evaluates it over a whole x-by-t
grid in one vectorised call and adds two summaries that change in time:
the largest |u| and the integral of u over the rod (total heat, for the
heat equation) -- so the evolution can be seen as a picture, and measured.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

MAX_POINTS = 600_000


@dataclass
class PDEField:
    error: str | None = None
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    ts: np.ndarray = field(default_factory=lambda: np.empty(0))
    u: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))         # (n_t, n_x)
    u_min: float = 0.0
    u_max: float = 0.0
    max_abs: np.ndarray = field(default_factory=lambda: np.empty(0))       # max |u| at each time
    integral: np.ndarray = field(default_factory=lambda: np.empty(0))      # integral of u dx at each time


def evaluate_field(solution: sp.Expr, length: float, t_max: float, n_x: int = 200, n_t: int = 120) -> PDEField:
    """Evaluates `solution` (a formula in the symbols named x and t) on
    x in [0, length], t in [0, t_max]."""
    if not (length > 0 and t_max > 0):
        return PDEField(error="The length and the end time must both be positive.")
    if n_x < 2 or n_t < 2 or n_x * n_t > MAX_POINTS:
        return PDEField(error=f"The grid must be at least 2x2 and at most {MAX_POINTS} points.")

    symbols = {s.name: s for s in solution.free_symbols}
    extra = sorted(set(symbols) - {"x", "t"})
    if extra:
        return PDEField(error=f"The solution contains {', '.join(extra)} besides x and t, so it can't be drawn.")
    x_sym = symbols.get("x", sp.Symbol("x"))
    t_sym = symbols.get("t", sp.Symbol("t"))

    xs = np.linspace(0.0, length, n_x)
    ts = np.linspace(0.0, t_max, n_t)
    try:
        f = sp.lambdify((x_sym, t_sym), solution, "numpy")
        with np.errstate(all="ignore"):
            grid = np.asarray(f(xs[None, :], ts[:, None]), dtype=complex)
    except Exception as exc:  # noqa: BLE001
        return PDEField(error=f"Could not evaluate the solution: {exc}")
    grid = np.broadcast_to(grid, (n_t, n_x))
    if np.any(np.abs(grid.imag) > 1e-9 * (1.0 + np.abs(grid.real))) or not np.all(np.isfinite(grid.real)):
        return PDEField(error="The solution isn't real and finite everywhere on this grid.")

    u = np.array(grid.real)
    return PDEField(xs=xs, ts=ts, u=u, u_min=float(u.min()), u_max=float(u.max()),
                     max_abs=np.max(np.abs(u), axis=1), integral=trapezoid(u, xs, axis=1))
