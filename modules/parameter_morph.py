"""
Parameter morph: a whole FAMILY of solution curves as one parameter varies.

A static sweep chart overlays a handful of curves and hides how the shape
changes in between; this evaluates the closed-form solution for many values
of one parameter so the family can be played as an animation. Along the way
it counts each curve's turning points, so the point where a solution stops
decaying monotonically and starts to oscillate (a damping ratio crossing 1,
say) is visible as a number, not just something to spot by eye.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.series_animation import _fixed_y_range

MAX_VALUES = 120


@dataclass
class MorphResult:
    applicable: bool
    reason: str = ""
    function: str = ""
    param_name: str = ""
    values: np.ndarray = field(default_factory=lambda: np.empty(0))     # the swept parameter
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    curves: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))  # (n_values, n_t)
    turning_points: list[int] = field(default_factory=list)             # per curve
    y_range: tuple[float, float] = (-1.0, 1.0)
    nominal_value: float | None = None
    nominal_curve: np.ndarray | None = None                             # the curve at the nominal value


def count_turning_points(y: np.ndarray, min_swing: float = 0.01) -> int:
    """Number of VISIBLE local maxima/minima of the sampled curve `y`: a
    turning point counts only once the curve has moved at least `min_swing`
    of its own range away from it (a zigzag filter). A strongly damped
    oscillator technically keeps turning forever, but with an amplitude far
    below anything a plot could show; counting those would label a curve
    that looks monotone as having six turning points. NaN samples are
    skipped."""
    y = y[np.isfinite(y)]
    if len(y) < 3:
        return 0
    threshold = min_swing * float(np.max(y) - np.min(y))
    if threshold <= 0:
        return 0
    direction = 0                                  # 0 = not yet moved far enough, 1 = rising, -1 = falling
    low = high = extreme = float(y[0])
    turning = 0
    for v in map(float, y[1:]):
        if direction == 0:
            low, high = min(low, v), max(high, v)
            if v - low >= threshold:
                direction, extreme = 1, v
            elif high - v >= threshold:
                direction, extreme = -1, v
        elif direction == 1:
            if v > extreme:
                extreme = v
            elif extreme - v >= threshold:
                turning, direction, extreme = turning + 1, -1, v     # a maximum is now confirmed
        else:
            if v < extreme:
                extreme = v
            elif v - extreme >= threshold:
                turning, direction, extreme = turning + 1, 1, v      # a minimum is now confirmed
    return turning


def first_regime_change(morph: MorphResult) -> tuple[float, float, int, int] | None:
    """Where the family's shape first changes character: the first pair of
    neighbouring parameter values whose curves have a different number of
    visible turning points, as (value_before, value_after, turning_before,
    turning_after). None if every curve has the same count."""
    counts = morph.turning_points
    for i in range(1, len(counts)):
        if counts[i] != counts[i - 1]:
            return float(morph.values[i - 1]), float(morph.values[i]), counts[i - 1], counts[i]
    return None


def _family_y_range(curves: np.ndarray) -> tuple[float, float]:
    """Y limits that show every curve in full: the family's true minimum and
    maximum, padded. Only when a few values dwarf the rest (a family that
    diverges, e.g. through a pole) does it fall back to percentile limits
    so those can't squash everything else into a line."""
    finite = curves[np.isfinite(curves)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(finite.min()), float(finite.max())
    typical = float(np.percentile(finite, 98) - np.percentile(finite, 2))
    if hi - lo > 50 * max(typical, 1e-12):
        return _fixed_y_range(finite, pad=0.1)
    pad = 0.05 * max(hi - lo, 1e-9)
    return (lo - pad, hi + pad)


def morph_family(expr: sp.Expr, t_sym: sp.Symbol, param: sp.Symbol, lo: float, hi: float,
                  t_range: tuple[float, float], fixed: dict[sp.Symbol, float], function: str = "y",
                  n_values: int = 40, n_points: int = 300, nominal_value: float | None = None,
                  label: str | None = None) -> MorphResult:
    """The solution `expr` (a formula in `t_sym`, `param` and the symbols in
    `fixed`) for `n_values` values of `param` between `lo` and `hi`, over
    `t_range`. `nominal_value`, if given, also yields the curve at that
    value (for marking "where you currently are" on the family). `label` is
    what a person should see for the parameter in titles and messages when
    that differs from the symbol -- an initial value's placeholder symbol is
    `_ic_N_0_`, but a person knows it as N(0)."""
    name = label or param.name
    if not 2 <= n_values <= MAX_VALUES:
        return MorphResult(applicable=False, reason=f"The number of values must be between 2 and {MAX_VALUES}.")
    if not (np.isfinite(lo) and np.isfinite(hi)) or not hi > lo:
        return MorphResult(applicable=False, reason=f"The range for {name} must run from a lower to a higher value.")
    if not t_range[1] > t_range[0]:
        return MorphResult(applicable=False, reason="The end time must be after the start time.")
    if param not in expr.free_symbols:
        return MorphResult(applicable=False, reason=f"This solution doesn't depend on {name}.")
    left_over = sorted(s.name for s in expr.free_symbols - {t_sym, param} - set(fixed))
    if left_over:
        return MorphResult(applicable=False,
                            reason=f"No value for {', '.join(left_over)}; give every other symbol a value first.")

    values = np.linspace(lo, hi, n_values)
    ts = np.linspace(t_range[0], t_range[1], n_points)
    body = expr.subs({s: v for s, v in fixed.items() if s != param})
    f = sp.lambdify((t_sym, param), body, "numpy")
    with np.errstate(all="ignore"):
        raw = np.asarray(f(ts[None, :], values[:, None]), dtype=complex)
    raw = np.broadcast_to(raw, (n_values, n_points))
    ok = np.abs(raw.imag) <= 1e-9 * (1.0 + np.abs(raw.real))
    curves = np.where(ok & np.isfinite(raw.real), raw.real, np.nan)

    nominal_curve = None
    if nominal_value is not None:
        with np.errstate(all="ignore"):
            single = np.asarray(f(ts, float(nominal_value)), dtype=complex)
        single = np.broadcast_to(single, ts.shape)
        nominal_curve = np.where((np.abs(single.imag) <= 1e-9 * (1.0 + np.abs(single.real)))
                                  & np.isfinite(single.real), single.real, np.nan)

    return MorphResult(
        applicable=True, function=function, param_name=name, values=values, t=ts, curves=curves,
        turning_points=[count_turning_points(row) for row in curves], y_range=_family_y_range(curves),
        nominal_value=nominal_value, nominal_curve=nominal_curve)
