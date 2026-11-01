"""
Limiting-case and large-N checks -- Streamlit-free.

A physical result should reduce to the older, simpler theory in the right limit: quantum mechanics to classical as hbar -> 0, relativity to
Newtonian as c -> infinity, a finite system to its thermodynamic limit as N -> infinity. This module makes that a CHECK instead of a
remark, using modules/series_asymptotics.py for the expansions.

`check_limit(expr, param, to, expected, correction)`:
  1. expands the expression in the limiting parameter (asymptotic_expansion for -> infinity, taylor_series about 0 for -> 0; a plain
     SymPy series is the fallback and is reported as such);
  2. checks the LEADING term equals the expected classical / Newtonian expression EXACTLY (symbolic difference simplifies to 0),
     and, if given, that the FIRST CORRECTION equals the expected one;
  3. confirms it NUMERICALLY with 40-digit arithmetic at a geometric sequence of parameter values approaching the limit: the relative
     difference |f/f_expected - 1| must shrink, and its measured power law must equal the one the series predicts (the power of the
     parameter in correction/leading).
`check_ratio_limit` handles limits with no power series (an essential singularity, like tunnelling as hbar -> 0): f / g must tend to 1.
`check_large_n` fits |f(N) - L| ~ a N^(-p) and compares p with the expected rate.
The presets cover the textbook cases: relativistic kinetic energy, momentum, velocity addition and time dilation (c -> infinity);
Planck -> Rayleigh-Jeans, the quantum oscillator's energy and heat capacity -> equipartition, WKB tunnelling (hbar -> 0); Stirling,
de Moivre-Laplace, the harmonic numbers and the 1D Ising free energy (N -> infinity).
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import mpmath
import numpy as np
import sympy as sp

from modules import series_asymptotics as sa
from modules.cc_common import CrossCuttingError, names_in
from modules.quantum_common import CheckList


@dataclass
class LimitSpec:
    key: str
    title: str
    expr: str
    param: str
    to: str                           # "0" or "oo"
    expected: str
    correction: str | None
    values: dict[str, float]          # numeric values of every symbol (the limiting one gives the starting point)
    description: str
    mode: str = "series"              # "series" | "ratio"


LIMITS: dict[str, LimitSpec] = {s.key: s for s in [
    LimitSpec("rel_ke", "Relativistic kinetic energy → ½mv²  (c → ∞)", "m*c^2*(1/sqrt(1 - v^2/c^2) - 1)", "c", "oo", "m*v^2/2", "3*m*v^4/(8*c^2)",
              {"m": 1.0, "v": 1.0, "c": 4.0}, "The Newtonian kinetic energy, with its first relativistic correction (3/8) m v⁴/c²."),
    LimitSpec("rel_p", "Relativistic momentum → mv  (c → ∞)", "m*v/sqrt(1 - v^2/c^2)", "c", "oo", "m*v", "m*v^3/(2*c^2)",
              {"m": 1.0, "v": 1.0, "c": 4.0}, "γ m v → m v."),
    LimitSpec("rel_add", "Velocity addition → u + v  (c → ∞)", "(u + v)/(1 + u*v/c^2)", "c", "oo", "u + v", "-u*v*(u + v)/c^2",
              {"u": 1.0, "v": 1.5, "c": 6.0}, "Einstein's rule becomes the Galilean one."),
    LimitSpec("time_dilation", "Time dilation → t  (c → ∞)", "t/sqrt(1 - v^2/c^2)", "c", "oo", "t", "t*v^2/(2*c^2)",
              {"t": 1.0, "v": 1.0, "c": 4.0}, "Absolute time."),
    LimitSpec("planck_rj", "Planck's law → Rayleigh–Jeans  (ħ → 0)", "hbar*w^3/(pi^2*c^3)/(exp(hbar*w/(k*T)) - 1)", "hbar", "0", "w^2*k*T/(pi^2*c^3)",
              "-hbar*w^3/(2*pi^2*c^3)", {"hbar": 1.0, "w": 1.0, "k": 1.0, "T": 1.0, "c": 1.0},
              "Spectral energy density per angular frequency: the classical (ultraviolet-catastrophe) law, with the first quantum correction."),
    LimitSpec("qho_energy", "Quantum oscillator thermal energy → kT  (ħ → 0)", "hbar*w*(1/2 + 1/(exp(hbar*w/(k*T)) - 1))", "hbar", "0", "k*T",
              "hbar^2*w^2/(12*k*T)", {"hbar": 1.0, "w": 1.0, "k": 1.0, "T": 1.0}, "Equipartition: kT per oscillator."),
    LimitSpec("qho_heat", "Quantum oscillator heat capacity → k  (ħ → 0)", "k*(hbar*w/(k*T))^2*exp(hbar*w/(k*T))/(exp(hbar*w/(k*T)) - 1)^2", "hbar", "0",
              "k", "-k*(hbar*w/(k*T))^2/12", {"hbar": 1.0, "w": 1.0, "k": 1.0, "T": 1.0}, "The Dulong–Petit value."),
    LimitSpec("tunnelling", "Barrier tunnelling → WKB form  (ħ → 0)",
              "1/(1 + V^2*sinh(a*sqrt(2*m*(V - W))/hbar)^2/(4*W*(V - W)))", "hbar", "0",
              "16*W*(V - W)/V^2*exp(-2*a*sqrt(2*m*(V - W))/hbar)", None, {"hbar": 1.0, "a": 1.0, "m": 1.0, "V": 2.0, "W": 1.0},
              "(W is the particle energy, V the barrier height.) The transmission coefficient vanishes faster than any power of ħ (an essential singularity: no power series), but the ratio to the WKB "
              "exponential tends to 1.", "ratio"),
]}


@dataclass
class LimitReport:
    spec_title: str
    series_text: str
    leading: sp.Expr | None
    expected: sp.Expr
    leading_matches: bool | None
    correction_matches: bool | None
    predicted_exponent: float | None
    observed_exponent: float | None
    table: list[tuple[float, float, float, float]]           # (parameter, f, expected, |f/expected - 1|)
    module_used: str
    checks: CheckList = field(default_factory=CheckList)


def _symbols_for(exprs: list[str]) -> dict[str, sp.Symbol]:
    out: dict[str, sp.Symbol] = {}
    for e in exprs:
        for n in sorted(names_in(e)):                       # variable names only (pi and function names are not symbols)
            out.setdefault(n, sp.Symbol(n))
    return out


def _numeric_sequence(f: sp.Expr, g: sp.Expr, param: sp.Symbol, to: str, values: dict[str, float], steps: int = 6, ratio_mode: bool = False
                      ) -> list[tuple[float, float, float, float]]:
    mpmath.mp.dps = 50
    syms = [s for s in f.free_symbols | g.free_symbols]
    ff, gg = sp.lambdify(syms, f, "mpmath"), sp.lambdify(syms, g, "mpmath")
    rows = []
    start = values[str(param)]
    for k in range(steps):
        p = start * (10 ** (-k) if to == "0" else 10 ** k)
        args = [mpmath.mpf(p if s == param else values[str(s)]) for s in syms]
        fv, gv = ff(*args), gg(*args)
        rel = abs(fv / gv - 1) if gv != 0 else abs(fv)
        rows.append((float(p), float(fv), float(gv), float(rel)))
    return rows


def _fit_exponent(rows: list[tuple[float, float, float, float]], to: str) -> float | None:
    pts = [(math.log(abs(p) if to == "0" else 1.0 / abs(p)), math.log(r)) for p, _, _, r in rows if r > 1e-30]
    if len(pts) < 3:
        return None
    x, y = np.array([a for a, _ in pts]), np.array([b for _, b in pts])
    return float(np.polyfit(x[-4:], y[-4:], 1)[0])


def check_limit(expr: str, param: str, to: str, expected: str, correction: str | None = None, values: dict[str, float] | None = None,
                order: int = 4, mode: str = "series", title: str = "limit") -> LimitReport:
    if to not in ("0", "oo"):
        raise CrossCuttingError("The limit must be '0' or 'oo' (infinity).")
    if mode not in ("series", "ratio"):
        raise CrossCuttingError("mode must be 'series' or 'ratio'.")
    syms = _symbols_for([expr, expected] + ([correction] if correction else []))
    if param not in syms:
        raise CrossCuttingError(f"The expression does not contain the parameter '{param}'.")
    vals = {k: 1.0 for k in syms}
    vals.update(values or {})
    fexpr = sp.sympify(expr.replace("^", "**"), locals=syms)
    gexpr = sp.sympify(expected.replace("^", "**"), locals=syms)
    ck = CheckList()
    leading = None
    lead_ok: bool | None = None
    corr_ok: bool | None = None
    pred: float | None = None
    series_text = ""
    used = "—"
    if mode == "series":
        res = sa.asymptotic_expansion(expr, param, order) if to == "oo" else sa.taylor_series(expr, param, 0, order)
        used = "series_asymptotics." + ("asymptotic_expansion" if to == "oo" else "taylor_series")
        if res.error or not res.expansion_available or res.truncated is None:
            ps = sp.Symbol(param)
            used = "sympy.series (fallback: the asymptotics module gave no expansion)"
            ser = sp.series(fexpr.subs(syms[param], ps), ps, sp.oo if to == "oo" else 0, order).removeO()
            truncated = sp.sympify(ser)
        else:
            truncated = res.truncated
        series_text = str(truncated)
        terms = sp.Add.make_args(sp.expand(truncated))
        p_sym = sp.Symbol(param)

        def growth(t: sp.Expr) -> float:
            e = t.as_powers_dict().get(p_sym, 0)
            return float(e) if to == "oo" else -float(e)
        groups: dict[float, sp.Expr] = {}
        for tm in terms:                                        # terms with the same power of the parameter belong together (u + v is ONE term)
            groups[growth(tm)] = groups.get(growth(tm), sp.Integer(0)) + tm
        ordered = [groups[k] for k in sorted(groups, reverse=True)]
        terms = tuple(ordered)
        leading = terms[0]
        exp_s = gexpr.subs({syms[k]: sp.Symbol(k) for k in syms})
        lead_ok = bool(sp.simplify(leading - exp_s) == 0)
        ck.add(f"the leading term of the expansion IS the expected result ({expected})", lead_ok, f"expansion: {series_text}  [{used}]")
        if len(terms) > 1:
            q = (sp.simplify(terms[1] / leading)).as_powers_dict().get(p_sym, None)
            pred = float(-q) if to == "oo" and q is not None else (float(q) if q is not None else None)
        if correction:
            corr = sp.sympify(correction.replace("^", "**"), locals={k: sp.Symbol(k) for k in syms})
            second = terms[1] if len(terms) > 1 else sp.Integer(0)
            corr_ok = bool(sp.simplify(second - corr) == 0)
            ck.add(f"the first correction is {correction}", corr_ok, f"found: {second}")
    table = _numeric_sequence(fexpr, gexpr, syms[param], to, vals)
    obs = _fit_exponent(table, to)
    shrinking = all(table[i][3] > table[i + 1][3] or table[i + 1][3] < 1e-30 for i in range(len(table) - 1))   # (below 1e-30: converged beyond the working precision)
    ck.add("the relative difference from the expected result shrinks at every step towards the limit", shrinking, "|f/f_expected − 1|: " + ", ".join(f"{r[3]:.2e}" for r in table))
    if pred is not None and obs is not None:
        ck.add(f"measured convergence rate = the series' prediction (power {pred:g} of the {'inverse ' if to == 'oo' else ''}parameter)",
               abs(obs - pred) < 0.15, f"measured {obs:.3f}")
    if mode == "ratio":
        ck.add("f / g → 1", table[-1][3] < 1e-3, f"|f/g − 1| = {table[-1][3]:.2e} at the last step")
    return LimitReport(title, series_text, leading, gexpr, lead_ok, corr_ok, pred, obs, table, used, ck)


def check_preset(key: str) -> LimitReport:
    if key not in LIMITS:
        raise CrossCuttingError(f"Unknown limit '{key}'. Available: {', '.join(LIMITS)}.")
    s = LIMITS[key]
    return check_limit(s.expr, s.param, s.to, s.expected, s.correction, s.values, 4, s.mode, s.title)


# ------------------------------------------------------------------------------------------ large N

@dataclass
class LargeNSpec:
    key: str
    title: str
    func: Callable[[int], Any]               # f(N), computed with mpmath
    limit: Callable[[int], Any] | float   # the large-N form (a function of N) or the limit value
    expected_exponent: float                 # error ~ N^-p
    description: str


def _binom_ratio(N: int) -> Any:
    z = 1.0
    p = 0.3
    q = 1 - p
    k = int(round(N * p + z * math.sqrt(N * p * q)))
    pmf = mpmath.binomial(N, k) * mpmath.mpf(p) ** k * mpmath.mpf(q) ** (N - k)
    x = (k - N * p) / mpmath.sqrt(N * p * q)
    gauss = mpmath.exp(-x ** 2 / 2) / mpmath.sqrt(2 * mpmath.pi * N * p * q)
    return pmf / gauss


def _ising_fN(N: int) -> Any:
    b = mpmath.mpf(1)
    J, h = mpmath.mpf(1), mpmath.mpf("0.3")
    root = mpmath.sqrt(mpmath.exp(2 * b * J) * mpmath.sinh(b * h) ** 2 + mpmath.exp(-2 * b * J))
    lp, lm = mpmath.exp(b * J) * mpmath.cosh(b * h) + root, mpmath.exp(b * J) * mpmath.cosh(b * h) - root
    return mpmath.log(lp ** N + lm ** N) / N - mpmath.log(lp)


LARGE_N: dict[str, LargeNSpec] = {s.key: s for s in [
    LargeNSpec("stirling", "Stirling: ln N! − (N ln N − N + ½ ln 2πN) → 0 like 1/(12N)",
               lambda N: mpmath.loggamma(N + 1) - (N * mpmath.log(N) - N + mpmath.log(2 * mpmath.pi * N) / 2), 0.0, 1.0,
               "The Stirling series: the remainder is 1/(12N) − 1/(360N³) + …"),
    LargeNSpec("de_moivre", "de Moivre–Laplace: binomial pmf / Gaussian → 1 like 1/√N",
               _binom_ratio, 1.0, 0.5, "At a fixed number of standard deviations from the mean the relative error falls as N^(−1/2)."),
    LargeNSpec("harmonic", "Harmonic number: H_N − ln N → γ like 1/(2N)",
               lambda N: mpmath.harmonic(N) - mpmath.log(N), float(mpmath.euler), 1.0, "Euler–Mascheroni constant γ = 0.5772…"),
    LargeNSpec("ising_1d", "1D Ising: (1/N) ln Z_N − ln λ₊ → 0 exponentially (the thermodynamic limit)",
               _ising_fN, 0.0, float("inf"), "A periodic chain's free energy per site converges to the transfer-matrix eigenvalue faster than any power of N."),
]}


@dataclass
class LargeNReport:
    title: str
    ns: list[int]
    errors: list[float]
    observed_exponent: float | None
    limit_value: float
    checks: CheckList = field(default_factory=CheckList)


def check_large_n(key: str, ns: list[int] | None = None) -> LargeNReport:
    if key not in LARGE_N:
        raise CrossCuttingError(f"Unknown large-N check '{key}'. Available: {', '.join(LARGE_N)}.")
    s = LARGE_N[key]
    mpmath.mp.dps = 50
    ns = ns or [50, 100, 200, 400, 800, 1600] if key != "ising_1d" else [5, 10, 20, 40, 80, 160]
    lim = float(s.limit(ns[-1])) if callable(s.limit) else float(s.limit)
    vals = [s.func(n) for n in ns]
    errs = [float(abs(v - lim)) for v in vals]
    ck = CheckList()
    ck.add("the error shrinks as N grows", all(errs[i] > errs[i + 1] for i in range(len(errs) - 1)), "errors: " + ", ".join(f"{e:.2e}" for e in errs))
    pts = [(math.log(n), math.log(e)) for n, e in zip(ns, errs) if e > 1e-40]
    obs = None
    if s.expected_exponent == float("inf"):
        steepening = len(errs) >= 3 and (errs[-2] / max(errs[-1], 1e-300) > errs[0] / errs[1] if errs[1] else True)
        ck.add("the error decays faster than any power of N (exponentially)", bool(steepening or errs[-1] < 1e-30), f"error ratios: {errs[0] / errs[1]:.3g} (first step), {errs[-2] / max(errs[-1], 1e-300):.3g} (last step)")
    elif len(pts) >= 3:
        x, y = np.array([a for a, _ in pts]), np.array([b for _, b in pts])
        obs = float(np.polyfit(x[-4:], y[-4:], 1)[0])
        ck.add(f"error ~ N^(−{s.expected_exponent:g})", abs(-obs - s.expected_exponent) < 0.1, f"fitted exponent {-obs:.3f}")
    return LargeNReport(s.title, ns, errs, obs, lim, ck)
