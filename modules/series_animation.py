"""
Partial-sum views of a series: the data behind "watch the approximation
converge" animations.

taylor_partial_sums takes an already-computed (and already-verified)
SeriesResult from modules.series_asymptotics and splits it into cumulative
partial sums, one per added nonzero term. fourier_partial_sums computes the
Fourier series of a function on [-L, L] and returns its partial sums by
harmonic. Both return a PartialSums: the target function and every partial
sum on a common x grid, plus the error of each -- pure computation, no
plotting (see modules.plotter.build_partial_sum_animation).
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid

from modules.series_asymptotics import SeriesResult, _parse


@dataclass
class PartialSums:
    kind: str                                  # "taylor" | "fourier"
    error: str | None = None                   # set => nothing below is meaningful
    xs: np.ndarray = field(default_factory=lambda: np.empty(0))
    target: np.ndarray = field(default_factory=lambda: np.empty(0))     # the true function (NaN where undefined)
    sums: list[np.ndarray] = field(default_factory=list)                # cumulative partial sum, one per frame
    labels: list[str] = field(default_factory=list)                     # human label per frame
    errors: list[float] = field(default_factory=list)                   # RMS error per frame (see `error_region`)
    error_region: str = ""                     # what the RMS error is measured over
    y_range: tuple[float, float] = (-1.0, 1.0)  # fixed y limits so the animation doesn't rescale
    center: float | None = None                # Taylor expansion point (marked on the plot); None for Fourier
    monotone: bool | None = None               # Fourier only: error never increased (a property of the series)
    coefficients: list[tuple[float, float]] = field(default_factory=list)   # Fourier (a_n, b_n), n = 0.. (b_0 = 0)


def _fixed_y_range(values: np.ndarray, pad: float = 0.6) -> tuple[float, float]:
    """Y limits taken from the TRUE function (2nd-98th percentile, so a pole
    or a diverging polynomial can't stretch the axis), widened by `pad` of
    its spread. Fixed for every frame so the animation doesn't rescale."""
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return (-1.0, 1.0)
    lo, hi = float(np.percentile(finite, 2)), float(np.percentile(finite, 98))
    span = hi - lo if hi > lo else max(abs(hi), 1.0)
    return (lo - pad * span, hi + pad * span)


def _real_grid(f, xs: np.ndarray) -> np.ndarray:
    """Evaluates lambdified `f` on xs as a real float array of the same shape
    (a constant lambdifies to a bare scalar; undefined points become NaN)."""
    with np.errstate(all="ignore"):
        out = np.real(np.asarray(f(xs), dtype=complex))
    return np.where(np.isfinite(out), np.broadcast_to(out, xs.shape), np.nan).astype(float)


def _rms(diff: np.ndarray) -> float:
    finite = diff[np.isfinite(diff)]
    return float(np.sqrt(np.mean(finite ** 2))) if finite.size else float("nan")


# ------------------------------------------------------------------ Taylor / Laurent

def taylor_partial_sums(result: SeriesResult, var_name: str = "x", half_width: float = 3.0,
                         n_points: int = 600) -> PartialSums:
    """Cumulative partial sums of a Taylor/Laurent SeriesResult, one frame per
    distinct power present (so sin(x) animates x, x - x^3/6, ... rather than
    repeating a frame for every power whose coefficient is zero). The RMS
    error is measured over the INNER half of the plotted window, around the
    expansion point -- where a Taylor series is supposed to be accurate --
    so it falls as terms are added, and visibly grows instead if the series
    is being used beyond its radius of convergence."""
    if result.error or not result.expansion_available or result.truncated is None \
            or result.input_expr is None or result.point is None or result.kind != "taylor":
        return PartialSums(kind="taylor", error="There is no Taylor series to animate for this expression.")
    if half_width <= 0:
        return PartialSums(kind="taylor", error="The plot half-width must be positive.")

    x = sp.Symbol(var_name)
    h = sp.Dummy("h")
    center = float(result.point)
    shifted = sp.expand(result.truncated.subs(x, result.point + h))

    by_power: dict[sp.Expr, sp.Expr] = {}
    for term in sp.Add.make_args(shifted):
        coeff, power = term.as_coeff_exponent(h)
        if coeff.has(h):
            return PartialSums(kind="taylor",
                                error="This expansion has terms that aren't plain powers (e.g. a logarithm), "
                                      "so it can't be split into term-by-term partial sums.")
        by_power[power] = by_power.get(power, sp.Integer(0)) + coeff
    powers = sorted(by_power, key=lambda p: float(p))

    xs = np.linspace(center - half_width, center + half_width, n_points)
    target = _real_grid(sp.lambdify(x, result.input_expr, "numpy"), xs)

    inner = np.abs(xs - center) <= half_width / 2
    cumulative = sp.Integer(0)
    sums: list[np.ndarray] = []
    labels: list[str] = []
    errors: list[float] = []
    base = "x" if center == 0 else f"(x \u2212 {center:g})"
    for k, power in enumerate(powers, start=1):
        cumulative = cumulative + by_power[power] * h ** power
        partial = sp.lambdify(x, cumulative.subs(h, x - result.point), "numpy")
        values = _real_grid(partial, xs)
        sums.append(values)
        labels.append(f"through {base}^{power} \u2014 {k} term{'s' if k != 1 else ''}")
        errors.append(_rms((values - target)[inner]))

    return PartialSums(kind="taylor", xs=xs, target=target, sums=sums, labels=labels, errors=errors,
                        error_region=f"within \u00b1{half_width / 2:g} of the expansion point",
                        y_range=_fixed_y_range(target), center=center)


# ------------------------------------------------------------------ Fourier series

_MAX_HARMONICS = 60


def fourier_partial_sums(expr_str: str, var_name: str = "x", half_period: float = float(np.pi),
                          n_harmonics: int = 12, n_points: int = 800,
                          quadrature_points: int = 40001) -> PartialSums:
    """Fourier series of f on [-L, L] (L = `half_period`, so the period is 2L),
    extended periodically, with one frame per harmonic added:

        S_N(x) = a0/2 + sum_{n=1..N} a_n cos(n pi x / L) + b_n sin(n pi x / L)

    Coefficients come from trapezoid quadrature on a dense grid, not symbolic
    integration -- so a piecewise or discontinuous f (a square or sawtooth
    wave, the cases where the picture is most instructive, including the
    Gibbs overshoot at a jump) costs nothing and can't time out. The check
    reported in `monotone` is a real property of Fourier partial sums: each
    S_N is the best L2 approximation of f by trigonometric polynomials of
    degree N, so the RMS error over one period never increases as a harmonic
    is added; a violation would mean the coefficients are wrong."""
    if half_period <= 0:
        return PartialSums(kind="fourier", error="The half-period L must be positive.")
    if not 1 <= n_harmonics <= _MAX_HARMONICS:
        return PartialSums(kind="fourier", error=f"Harmonics must be between 1 and {_MAX_HARMONICS}.")
    try:
        expr, var = _parse(expr_str, var_name)
    except Exception as exc:  # noqa: BLE001
        return PartialSums(kind="fourier", error=f"Could not parse expression: {exc}")
    extra = expr.free_symbols - {var}
    if extra:
        return PartialSums(kind="fourier",
                            error=f"The function may only depend on {var_name}; found {sorted(map(str, extra))}.")

    try:
        f = sp.lambdify(var, expr, "numpy")
    except Exception as exc:  # noqa: BLE001
        return PartialSums(kind="fourier", error=f"Could not evaluate this expression numerically: {exc}")

    L = float(half_period)
    grid = np.linspace(-L, L, quadrature_points)
    fg = _real_grid(f, grid)
    if not np.all(np.isfinite(fg)):
        return PartialSums(kind="fourier",
                            error=f"The function must be finite everywhere on [-{L:g}, {L:g}] "
                                  "(it is undefined or infinite somewhere in that interval).")

    coefficients: list[tuple[float, float]] = []
    for k in range(n_harmonics + 1):
        a_k = float(trapezoid(fg * np.cos(k * np.pi * grid / L), grid) / L)
        b_k = float(trapezoid(fg * np.sin(k * np.pi * grid / L), grid) / L) if k > 0 else 0.0
        coefficients.append((a_k, b_k))

    def partial(at: np.ndarray, upto: int) -> np.ndarray:
        total = np.full_like(at, coefficients[0][0] / 2, dtype=float)
        for n in range(1, upto + 1):
            a_n, b_n = coefficients[n]
            total = total + a_n * np.cos(n * np.pi * at / L) + b_n * np.sin(n * np.pi * at / L)
        return total

    xs = np.linspace(-2 * L, 2 * L, n_points)            # two periods, so the periodic extension shows
    wrapped = ((xs + L) % (2 * L)) - L
    target = _real_grid(f, wrapped)

    sums, labels, errors = [], [], []
    for n in range(1, n_harmonics + 1):
        sums.append(partial(xs, n))
        labels.append(f"N = {n} harmonic{'s' if n != 1 else ''}")
        errors.append(_rms(partial(grid, n) - fg))        # over one full period, on the quadrature grid

    # The tolerance is relative to the function's own size, NOT to the first
    # error: when f is reproduced (almost) exactly -- a constant, or a single
    # sine -- every error is ~1e-16 of floating-point noise, and a tolerance
    # scaled to that would flag the noise as a "rising" error.
    scale = max(float(np.sqrt(np.mean(fg ** 2))), 1e-12)
    monotone = all(later <= earlier + 1e-6 * scale for earlier, later in zip(errors, errors[1:]))
    return PartialSums(kind="fourier", xs=xs, target=target, sums=sums, labels=labels, errors=errors,
                        error_region="over one full period", y_range=_fixed_y_range(target),
                        monotone=monotone, coefficients=coefficients)
