"""
Time-resolved views of an ODE solution.

modules.ode_utils.numerical_cross_check answers one question -- "does the
symbolic closed form agree with an independent numerical integration?" -- as
a single verdict at five sample points. This module computes the same
comparison as a full curve over time (compare_trajectories), and integrates
the autonomous system dx/dt = f(x, y), dy/dt = g(x, y) from many starting
points at once (phase_flow) so a phase portrait can show the flow itself
rather than only the one solved path. Pure computation, no plotting.
"""
from dataclasses import dataclass, field
from collections.abc import Callable
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp

from modules.equation_engine import Equation, ProblemModel
from modules.ode_utils import (
    NumericalCrossCheckResult, prepare_ivp, _CROSS_CHECK_RELATIVE_TOLERANCE,
)
import sympy as sp
from modules.timeout_utils import run_with_timeout


@dataclass
class TrajectoryComparison:
    applicable: bool
    reason: str = ""                          # why not applicable; empty otherwise
    names: list[str] = field(default_factory=list)
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    numeric: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))     # (n_funcs, n_times)
    symbolic: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    abs_error: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    rel_error: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    tolerance: float = _CROSS_CHECK_RELATIVE_TOLERANCE
    max_rel_error: float = 0.0
    ok: bool = False                          # max_rel_error is within tolerance everywhere shown
    t0: float = 0.0


def compare_trajectories(model: ProblemModel, group: list[Equation], solutions: dict[str, sp.Eq],
                          t_end: float | None = None, n_points: int = 300) -> TrajectoryComparison:
    """Integrates the ORIGINAL differential equation(s) numerically and
    evaluates the symbolic closed form on the same dense time grid, returning
    both and their pointwise error. Same scope and same relative-error rule
    (|numeric - symbolic| / max(|symbolic|, 1e-9)) as numerical_cross_check,
    so a plot of rel_error against `tolerance` shows exactly what that
    verdict was computed from -- and WHERE in time a disagreement starts,
    which a pass/fail at five points can't. `t_end` overrides the
    automatically chosen window (it must lie after the initial time)."""
    setup = prepare_ivp(model, group, solutions)
    if isinstance(setup, NumericalCrossCheckResult):
        return TrajectoryComparison(applicable=False, reason=setup.reason)

    end = setup.t0 + setup.window if t_end is None else float(t_end)
    if not end > setup.t0:
        return TrajectoryComparison(
            applicable=False,
            reason=f"The end time ({end:g}) must be after the initial time ({setup.t0:g}).")

    try:
        ivp = run_with_timeout(solve_ivp, setup.rhs, (setup.t0, end), setup.y0,
                                label="ode_trajectory_comparison", dense_output=True,
                                rtol=1e-8, atol=1e-10, method="RK45")
    except Exception as exc:  # noqa: BLE001
        return TrajectoryComparison(applicable=False, reason=f"Numerical integration failed: {exc}")
    if not ivp.success:
        return TrajectoryComparison(applicable=False,
                                      reason=f"Numerical integration did not converge: {ivp.message}")

    ts = np.linspace(setup.t0, end, n_points)
    try:
        numeric = np.asarray(ivp.sol(ts), dtype=float)
        # a closed form that doesn't depend on t at all (a constant solution)
        # lambdifies to a bare scalar -- broadcast it to the time grid
        symbolic = np.array([
            np.broadcast_to(np.real(np.asarray(sf(ts), dtype=complex)), ts.shape)
            for sf in setup.sol_funcs])
    except Exception as exc:  # noqa: BLE001
        return TrajectoryComparison(applicable=False, reason=f"Could not evaluate for comparison: {exc}")

    abs_error = np.abs(numeric - symbolic)
    rel_error = abs_error / np.maximum(np.abs(symbolic), 1e-9)
    max_rel = float(np.max(rel_error))
    return TrajectoryComparison(
        applicable=True, names=list(setup.func_order), t=ts, numeric=numeric, symbolic=symbolic,
        abs_error=abs_error, rel_error=rel_error, max_rel_error=max_rel,
        ok=max_rel < _CROSS_CHECK_RELATIVE_TOLERANCE, t0=setup.t0)


def ring_of_starts(center: tuple[float, float], x_range: tuple[float, float],
                    y_range: tuple[float, float], n: int, fraction: float = 0.3) -> list[tuple[float, float]]:
    """`n` starting points spaced evenly on an ellipse around `center`,
    scaled to `fraction` of the view box so they land inside it whatever
    the axes' units are. Empty for n <= 0."""
    if n <= 0:
        return []
    rx = fraction * (x_range[1] - x_range[0]) / 2
    ry = fraction * (y_range[1] - y_range[0]) / 2
    angles = 2 * np.pi * np.arange(n) / n
    return [(center[0] + rx * float(np.cos(a)), center[1] + ry * float(np.sin(a))) for a in angles]


def phase_flow(dx_f: Callable[..., Any], dy_f: Callable[..., Any],
                starts: list[tuple[float, float]], duration: float, n_points: int = 150,
                blowup: float = 1e6) -> list[tuple[np.ndarray, np.ndarray]]:
    """Integrates the autonomous system dx/dt = dx_f(x, y), dy/dt = dy_f(x, y)
    from each starting point over `duration`, all sampled at the SAME
    `n_points` times so the paths can be animated in lockstep. Always returns
    one (xs, ys) pair per start, in order: a path that leaves the finite
    range (|value| > `blowup`, e.g. x' = x^2 reaching infinity in finite
    time) is cut off there and NaN afterwards, and a start whose integration
    fails outright is all NaN -- never dropped, so the caller's index into
    `starts` stays valid."""
    ts = np.linspace(0.0, duration, n_points)

    def rhs(_t, state):
        return [float(np.real(dx_f(state[0], state[1]))), float(np.real(dy_f(state[0], state[1])))]

    paths: list[tuple[np.ndarray, np.ndarray]] = []
    for x0, y0 in starts:
        nan_path = (np.full(n_points, np.nan), np.full(n_points, np.nan))
        try:
            ivp = run_with_timeout(solve_ivp, rhs, (0.0, duration), [x0, y0], label="phase flow",
                                    t_eval=ts, rtol=1e-7, atol=1e-9, method="RK45")
        except Exception:  # noqa: BLE001
            paths.append(nan_path)
            continue
        xs = np.full(n_points, np.nan)
        ys = np.full(n_points, np.nan)
        got = ivp.y.shape[1]                      # solve_ivp stops early if it can't continue
        xs[:got], ys[:got] = ivp.y[0], ivp.y[1]
        bad = ~np.isfinite(xs) | ~np.isfinite(ys) | (np.abs(xs) > blowup) | (np.abs(ys) > blowup)
        if bad.any():
            first = int(np.argmax(bad))
            xs[first:], ys[first:] = np.nan, np.nan
        paths.append((xs, ys))
    return paths
