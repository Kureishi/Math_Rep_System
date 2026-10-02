"""
Uncertainty over time for an ODE solution: a fan chart.

modules.monte_carlo propagates uncertain inputs to a single number. For an
ODE the answer is a whole CURVE, and what matters is how the uncertainty
grows (or shrinks) as time goes on. This samples the uncertain parameters
and initial values, evaluates the symbolic closed-form solution for every
draw at once (vectorised -- no re-solving, no numerical integration per
sample), and reports the median and percentile bands at each time.

Optionally also returns a GUARANTEED envelope from interval arithmetic
(modules.interval_arithmetic): not "90% of draws fall in this band" but
"the solution cannot leave this band if every uncertain input stays within
mean +/- k standard deviations". The two answer different questions, and the
envelope is usually wider -- interval arithmetic treats every appearance of
a parameter in the formula as independent (the dependency problem), so it
over-covers whenever a parameter appears more than once. It is a valid
bound, not a tight one.
"""
import warnings
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np
import sympy as sp
from sympy.core.function import AppliedUndef

from modules.equation_engine import Equation, ProblemModel
from modules.interval_arithmetic import INTERVAL_FUNCTIONS, Interval
from modules.ode_utils import _funcs_used, _ics_for_group, initial_condition_symbol
from modules.timeout_utils import run_with_timeout

PERCENTILES = (5, 25, 50, 75, 95)
MAX_SAMPLES = 20000


@dataclass
class UncertainParameter:
    """Something in a solution that can be given an uncertainty: a known
    model parameter (kind "parameter") or the value of an initial
    condition (kind "initial condition")."""
    name: str            # what a person sees: "k" or "N(0)"
    symbol: sp.Symbol    # what the symbolic solution contains
    value: float         # its nominal (mean) value
    kind: str


@dataclass
class FanResult:
    applicable: bool
    reason: str = ""
    names: list[str] = field(default_factory=list)            # one entry per function
    t: np.ndarray = field(default_factory=lambda: np.empty(0))
    nominal: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))   # (n_funcs, n_t): every input at its mean
    p5: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    p25: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    median: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    p75: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    p95: np.ndarray = field(default_factory=lambda: np.empty((0, 0)))
    envelope_lo: np.ndarray | None = None                      # guaranteed interval bound, or None
    envelope_hi: np.ndarray | None = None
    envelope_sigmas: float | None = None
    envelope_note: str = ""                                    # why there is no envelope / what it assumes
    n_samples: int = 0
    seed: int = 0
    finite_fraction: float = 1.0                               # share of (sample, time) cells that evaluated
    uncertain: dict[str, float] = field(default_factory=dict)  # name -> standard deviation actually used


def _time_symbol(solutions: dict[str, sp.Eq]) -> sp.Symbol | None:
    for sol in solutions.values():
        args = sol.lhs.args
        if args:
            return args[0]
    return None


def candidate_parameters(model: ProblemModel, group: list[Equation],
                          symbolic_solutions: dict[str, sp.Eq]) -> tuple[list[UncertainParameter], list[str]]:
    """Every parameter and initial value the group's SYMBOLIC solutions
    (solve_ode(model, symbolic_initial_conditions=True)) depend on, with
    nominal values -- plus the names of any symbols with NO value (an
    unresolved integration constant, an undeclared parameter), which make
    a fan meaningless."""
    func_names: set[str] = set()
    for eq in group:
        if eq.sympy_eq is not None:
            func_names |= _funcs_used(eq.sympy_eq)
    present = [symbolic_solutions[n] for n in sorted(func_names) if n in symbolic_solutions]
    t = _time_symbol(symbolic_solutions)
    free = set().union(*(s.rhs.free_symbols for s in present)) - ({t} if t is not None else set())

    ics = _ics_for_group(model, func_names)
    ic_by_symbol = {initial_condition_symbol(lhs): (str(lhs), float(rhs)) for lhs, rhs in ics.items()}
    known = {v.symbol: float(v.known_value) for v in model.variables if v.known_value is not None}

    found: list[UncertainParameter] = []
    unresolved: list[str] = []
    for sym in sorted(free, key=str):
        if sym in ic_by_symbol:
            label, value = ic_by_symbol[sym]
            found.append(UncertainParameter(label, sym, value, "initial condition"))
        elif sym.name in known:
            found.append(UncertainParameter(sym.name, sym, known[sym.name], "parameter"))
        else:
            unresolved.append(sym.name)
    return found, unresolved


def _real_cells(values: np.ndarray) -> np.ndarray:
    """Complex -> real, with anything that is genuinely complex (an
    imaginary part beyond rounding noise) or non-finite turned into NaN."""
    values = np.asarray(values, dtype=complex)
    ok = np.abs(values.imag) <= 1e-9 * (1.0 + np.abs(values.real))
    out = np.where(ok, values.real, np.nan)
    return np.where(np.isfinite(out), out, np.nan)


@lru_cache(maxsize=128)
def real_form(expr: sp.Expr) -> sp.Expr:
    """`expr` with any imaginary unit removed. dsolve_system writes an
    oscillating solution as complex exponentials (exp(-t*(k - I*w))), which
    interval arithmetic can't evaluate; for real parameters the solution is
    real, so its real part is the same function written with sin and cos.
    Returns `expr` unchanged if it has no I, or if the real part can't be
    found in time."""
    if not expr.has(sp.I):
        return expr
    real_symbols = {s: sp.Symbol(s.name, real=True) for s in expr.free_symbols}
    try:
        real = run_with_timeout(lambda: sp.simplify(sp.re(expr.xreplace(real_symbols))),
                                 label="real form of solution")
    except Exception:  # noqa: BLE001  (includes ComputationTimeoutError)
        return expr
    back = {v: k for k, v in real_symbols.items()}
    result = real.xreplace(back)
    return expr if result.has(sp.I) or result.has(sp.re) or result.has(sp.im) else result


def _interval_envelope(expr: sp.Expr, t_sym: sp.Symbol, ts: np.ndarray, params: list[UncertainParameter],
                        sigmas: dict[str, float], k: float, nominal: dict[sp.Symbol, float]
                        ) -> tuple[np.ndarray, np.ndarray] | str:
    """Guaranteed (lo, hi) of `expr` at every time when each uncertain input
    lies in mean +/- k*std, or a string saying why that isn't possible."""
    uncertain = [p for p in params if sigmas.get(p.name, 0.0) > 0]
    fixed = {s: v for s, v in nominal.items() if s not in {p.symbol for p in uncertain}}
    try:
        f = sp.lambdify((t_sym, *[p.symbol for p in uncertain]), real_form(expr).subs(fixed),
                         modules=[INTERVAL_FUNCTIONS, "math"])
    except Exception as exc:  # noqa: BLE001
        return f"No interval form of this solution: {exc}"
    args = [Interval(p.value - k * sigmas[p.name], p.value + k * sigmas[p.name]) for p in uncertain]
    lo = np.full(len(ts), np.nan)
    hi = np.full(len(ts), np.nan)
    try:
        for i, tv in enumerate(ts):
            r = f(float(tv), *args)
            if isinstance(r, Interval):
                lo[i], hi[i] = r.lo, r.hi
            else:
                lo[i] = hi[i] = float(r)
    except Exception as exc:  # noqa: BLE001  (an unsupported function, sqrt of a negative range, ...)
        return f"Interval arithmetic can't evaluate this solution: {exc}"
    return lo, hi


def ode_uncertainty_fan(model: ProblemModel, group: list[Equation], symbolic_solutions: dict[str, sp.Eq],
                         sigmas: dict[str, float], t_range: tuple[float, float], n_points: int = 120,
                         n_samples: int = 800, seed: int = 12345,
                         envelope_sigmas: float | None = None) -> FanResult:
    """Fan chart for the functions of `group`. `sigmas` maps a parameter
    name (see candidate_parameters) to the standard deviation of a normal
    distribution around its nominal value (0 or absent = certain).
    `envelope_sigmas`, if given, also computes the guaranteed interval
    envelope for mean +/- that many standard deviations."""
    if not 1 <= n_samples <= MAX_SAMPLES:
        return FanResult(applicable=False, reason=f"The sample count must be between 1 and {MAX_SAMPLES}.")
    if not t_range[1] > t_range[0]:
        return FanResult(applicable=False, reason="The end time must be after the start time.")

    params, unresolved = candidate_parameters(model, group, symbolic_solutions)
    if unresolved:
        return FanResult(applicable=False,
                          reason=f"The solution still contains {', '.join(sorted(unresolved))} with no value "
                                 "(an unresolved constant or an undeclared parameter), so there is nothing "
                                 "definite to put uncertainty around.")
    group_funcs: set[str] = set()
    for eq in group:
        if eq.sympy_eq is not None:
            group_funcs |= _funcs_used(eq.sympy_eq)
    names = sorted(n for n in group_funcs if n in symbolic_solutions)
    if not names:
        return FanResult(applicable=False, reason="None of these equations has a closed-form solution.")
    t_sym = _time_symbol(symbolic_solutions)
    assert t_sym is not None

    used = {n: float(s) for n, s in sigmas.items() if s and s > 0}
    by_name = {p.name: p for p in params}
    stray = sorted(set(used) - set(by_name))
    if stray:
        return FanResult(applicable=False,
                          reason=f"{', '.join(stray)} isn't something this solution depends on.")
    if any(s < 0 or not np.isfinite(s) for s in sigmas.values()):
        return FanResult(applicable=False, reason="Standard deviations must be finite and not negative.")
    if not used:
        return FanResult(applicable=False,
                          reason="Give at least one parameter a standard deviation above zero.")

    rng = np.random.default_rng(seed)
    uncertain = [by_name[n] for n in sorted(used)]
    draws = {p.symbol: rng.normal(p.value, used[p.name], size=n_samples) for p in uncertain}
    nominal_values = {p.symbol: p.value for p in params}
    certain = {s: v for s, v in nominal_values.items() if s not in draws}

    ts = np.linspace(t_range[0], t_range[1], n_points)
    unc_syms = [p.symbol for p in uncertain]
    n_funcs = len(names)
    stats = {q: np.full((n_funcs, n_points), np.nan) for q in PERCENTILES}
    nominal = np.full((n_funcs, n_points), np.nan)
    env_lo = np.full((n_funcs, n_points), np.nan)
    env_hi = np.full((n_funcs, n_points), np.nan)
    envelope_note = ""
    have_envelope = envelope_sigmas is not None and envelope_sigmas > 0
    finite_cells = 0

    for i, name in enumerate(names):
        expr = symbolic_solutions[name].rhs.subs(certain)
        f = sp.lambdify((t_sym, *unc_syms), expr, "numpy")
        with np.errstate(all="ignore"):
            grid = f(ts[None, :], *[draws[s][:, None] for s in unc_syms])
        cells = _real_cells(np.broadcast_to(np.asarray(grid), (n_samples, n_points)))
        finite_cells += int(np.isfinite(cells).sum())
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)                  # an all-NaN time column
            quantiles = np.nanpercentile(cells, PERCENTILES, axis=0)
        for q, row in zip(PERCENTILES, quantiles):
            stats[q][i] = row
        nominal_f = sp.lambdify(t_sym, symbolic_solutions[name].rhs.subs(nominal_values), "numpy")
        with np.errstate(all="ignore"):
            nominal[i] = _real_cells(np.broadcast_to(np.asarray(nominal_f(ts)), ts.shape))
        if have_envelope and not envelope_note:
            assert envelope_sigmas is not None
            enclosure = _interval_envelope(symbolic_solutions[name].rhs, t_sym, ts, params, used,
                                            envelope_sigmas, nominal_values)
            if isinstance(enclosure, str):
                envelope_note = enclosure
            else:
                env_lo[i], env_hi[i] = enclosure

    has_envelope = have_envelope and not envelope_note
    if has_envelope:
        envelope_note = (f"Guaranteed bound for every input within \u00b1{envelope_sigmas:g} standard "
                         "deviations. Usually wider than the sampled bands: each appearance of a parameter in "
                         "the formula is treated as independent.")
    return FanResult(
        applicable=True, names=names, t=ts, nominal=nominal,
        p5=stats[5], p25=stats[25], median=stats[50], p75=stats[75], p95=stats[95],
        envelope_lo=env_lo if has_envelope else None, envelope_hi=env_hi if has_envelope else None,
        envelope_sigmas=envelope_sigmas if has_envelope else None, envelope_note=envelope_note,
        n_samples=n_samples, seed=seed, finite_fraction=finite_cells / (n_funcs * n_samples * n_points),
        uncertain=dict(used))
