"""
Bifurcation diagram of a one-parameter first-order map x -> g(x; p).

For each value of the parameter p the map is iterated from a starting value,
the first `n_transient` iterates are discarded (so the orbit has settled onto
its long-run behaviour), and the next `n_keep` are recorded. Plotted against
p, the recorded values show a fixed point, then a cycle that doubles
(period 2, 4, 8, ...), then chaos -- the classic picture for the logistic map
x -> r x (1 - x). All parameter values are iterated at once as one numpy
array, so a diagram of hundreds of parameters costs only
n_transient + n_keep vectorised steps.

Works from the numeric map alone, so it doesn't need a closed-form solution
of the recurrence (the logistic map has none).
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

MAX_PARAMS = 4000
MAX_PERIOD = 16          # a tail with more distinct values than this counts as aperiodic


@dataclass
class BifurcationResult:
    applicable: bool
    reason: str = ""
    param_name: str = ""
    params: np.ndarray = field(default_factory=lambda: np.empty(0))
    values: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))    # (n_keep, n_params); NaN once an orbit escapes
    periods: list[int | None] = field(default_factory=list)                 # per parameter; None = aperiodic/escaped
    transitions: list[tuple[float, int | None]] = field(default_factory=list)   # (first p of each new regime, its period)
    x0: float = 0.0


def attractor_period(tail: np.ndarray, tol: float) -> int | None:
    """Period of the settled orbit `tail`: how many distinct values (up to
    `tol`) it visits. 1 = fixed point, 2 = a 2-cycle, ... None if it visits
    more than MAX_PERIOD (chaotic or quasi-periodic) or contains a
    non-finite value (the orbit escaped to infinity)."""
    if not np.all(np.isfinite(tail)):
        return None
    ordered = np.sort(tail)
    clusters = 1 + int(np.sum(np.diff(ordered) > tol))
    return clusters if clusters <= MAX_PERIOD else None


def regime_changes(params: np.ndarray, periods: list[int | None], min_run: int = 3
                    ) -> list[tuple[float, int | None]]:
    """Where the long-run behaviour changes: the first parameter of each run
    of at least `min_run` consecutive parameters sharing a period, after the
    first run. A shorter run is a blip between regimes (a parameter sitting
    right at a bifurcation), not a regime of its own."""
    runs: list[tuple[int, int | None, int]] = []
    for i, period in enumerate(periods):
        if runs and runs[-1][1] == period:
            runs[-1] = (runs[-1][0], period, runs[-1][2] + 1)
        else:
            runs.append((i, period, 1))
    solid = [r for r in runs if r[2] >= min_run]
    changes: list[tuple[float, int | None]] = []
    previous = solid[0][1] if solid else None
    for start, period, _ in solid[1:]:
        if period != previous:
            changes.append((float(params[start]), period))
        previous = period
    return changes


def bifurcation_diagram(g: Callable[..., float], param_name: str, p_range: tuple[float, float],
                         x0: float, n_params: int = 500, n_transient: int = 600, n_keep: int = 64,
                         escape: float = 1e6) -> BifurcationResult:
    """`g(x, p)` is the map, accepting numpy arrays for both. `escape` is the
    magnitude beyond which an orbit counts as having diverged (and is
    dropped, rather than overflowing into NaN noise)."""
    lo, hi = p_range
    if not (np.isfinite(lo) and np.isfinite(hi)) or not hi > lo:
        return BifurcationResult(applicable=False,
                                  reason=f"The range for {param_name} must run from a lower to a higher value.")
    if not 2 <= n_params <= MAX_PARAMS:
        return BifurcationResult(applicable=False, reason=f"The number of parameters must be 2 to {MAX_PARAMS}.")
    if n_transient < 0 or n_keep < 2:
        return BifurcationResult(applicable=False, reason="Need at least 2 recorded iterates and no negative transient.")

    params = np.linspace(lo, hi, n_params)
    x = np.full(n_params, float(x0))
    kept = np.empty((n_keep, n_params))
    with np.errstate(all="ignore"):
        for step in range(n_transient + n_keep):
            x = np.asarray(g(x, params), dtype=float)
            x = np.where(np.abs(x) > escape, np.nan, x)           # an escaped orbit stays escaped (NaN propagates)
            if step >= n_transient:
                kept[step - n_transient] = x

    finite = kept[np.isfinite(kept)]
    scale = float(np.max(np.abs(finite))) if finite.size else 1.0
    tol = 1e-4 * max(scale, 1e-12)
    periods = [attractor_period(kept[:, j], tol) for j in range(n_params)]
    return BifurcationResult(applicable=True, param_name=param_name, params=params, values=kept, periods=periods,
                              transitions=regime_changes(params, periods), x0=float(x0))


def lambdify_map(g_expr: sp.Expr, x_sym: sp.Symbol, param: sp.Symbol, fixed: dict[sp.Symbol, float]
                  ) -> Callable[..., float] | str:
    """Turns the symbolic one-step map into g(x, p) for bifurcation_diagram
    with every symbol other than the state and the swept parameter replaced
    by its value -- or a string naming the symbols that have none."""
    left_over = sorted(s.name for s in g_expr.free_symbols - {x_sym, param} - set(fixed))
    if left_over:
        return f"No value for {', '.join(left_over)}; give every other symbol a value first."
    if param not in g_expr.free_symbols:
        return f"The map doesn't depend on {param.name}."
    f = sp.lambdify((x_sym, param), g_expr.subs({s: v for s, v in fixed.items() if s != param}), "numpy")
    return f
