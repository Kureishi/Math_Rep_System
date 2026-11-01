"""
Symbolic-numeric cross-validation -- Streamlit-free.

Wherever a CLOSED FORM exists, the numerical version is run too and both are shown, with their agreement (the app does this for algebra
already; this extends it to the other things a student or researcher asks SymPy for). Each task has an independent numerical route:

    integral    SymPy `integrate`                    scipy.integrate.quad (with its own error estimate), infinite limits allowed
    derivative  SymPy `diff` at a point              mpmath's finite-difference derivative of the ORIGINAL expression, 40 digits
    roots       SymPy `solve` (real roots in range)  sign-change scan + Brent's method, and the residual |f(root)| of every symbolic root
    ode         SymPy `dsolve` with initial values   scipy.integrate.solve_ivp (DOP853, tight tolerance), compared over a time grid
    sum         SymPy `summation`                    direct partial sums (finite) or mpmath.nsum (infinite)
    limit       SymPy `limit`                        the function evaluated at 40-digit points approaching the limit
    eigenvalues exact eigenvalues of a matrix        numpy.linalg.eigvals (eigvalsh for a symmetric matrix)

OUTCOMES are stated plainly: "agree" (within the stated tolerance), "DISAGREE" (a bug in one of the two routes, or a branch/domain issue
worth knowing about), "numeric only" (SymPy returned no closed form, e.g. an unevaluated Integral: the number is still given, labelled
as such), "symbolic only" (the numerical route failed). A SymPy computation is bounded by a time limit and reported, not hung on.
"""
import warnings
from dataclasses import dataclass, field
from typing import Any

import mpmath
import numpy as np
import sympy as sp
from scipy.integrate import IntegrationWarning, quad, solve_ivp
from scipy.optimize import brentq

from modules.cc_common import CrossCuttingError, names_in, parse_with
from modules.quantum_common import CheckList
from modules.timeout_utils import ComputationTimeoutError, run_with_timeout

SYM_TIMEOUT = 20.0


@dataclass
class CrossValidation:
    task: str
    problem: str
    symbolic: str | None                  # the closed form as text/latex source
    symbolic_latex: str | None
    symbolic_value: Any | None            # float, list, or array
    numeric_value: Any | None
    numeric_method: str
    error: float | None
    tolerance: float
    status: str                           # "agree" | "DISAGREE" | "numeric only" | "symbolic only"
    detail: str = ""
    checks: CheckList = field(default_factory=CheckList)


def _symbols(*texts: str, exclude: set[str] | None = None, values: dict[str, float] | None = None) -> tuple[dict[str, sp.Symbol], dict[str, float]]:
    names: set[str] = set()
    for t in texts:
        names |= names_in(t)
    names -= exclude or set()
    syms = {n: sp.Symbol(n, real=True) for n in sorted(names)}
    return syms, {n: float((values or {}).get(n, 1.0)) for n in names}


def _bound(text: str) -> sp.Expr:
    t = text.strip().lower()
    if t in ("oo", "inf", "infinity", "+oo", "+inf"):
        return sp.oo
    if t in ("-oo", "-inf", "-infinity"):
        return -sp.oo
    return parse_with(text, {})


def _finish(cv: CrossValidation) -> CrossValidation:
    if cv.symbolic_value is None and cv.numeric_value is not None:
        cv.status = "numeric only"
    elif cv.numeric_value is None and cv.symbolic_value is not None:
        cv.status = "symbolic only"
    elif cv.error is not None:
        cv.status = "agree" if cv.error <= cv.tolerance else "DISAGREE"
    cv.checks.add(f"closed form and numerical value agree ({cv.task})", cv.status == "agree",
                  f"{cv.status}: error {cv.error:.2e} against tolerance {cv.tolerance:.1e}" if cv.error is not None else cv.status + ". " + cv.detail)
    return cv


def _real(x: object) -> float | None:
    try:
        v = complex(sp.N(x))
    except (TypeError, ValueError):
        return None
    return v.real if abs(v.imag) <= 1e-12 * (1 + abs(v.real)) else None


# ------------------------------------------------------------------------------------------ integral

def validate_integral(expr: str, var: str, lower: str, upper: str, values: dict[str, float] | None = None) -> CrossValidation:
    syms, vals = _symbols(expr, exclude={var}, values=values)
    x = sp.Symbol(var, real=True)
    f = parse_with(expr, {**syms, var: x})
    a, b = _bound(lower), _bound(upper)
    sub = {syms[k]: v for k, v in vals.items()}
    fn = sp.lambdify(x, f.subs(sub), "numpy")
    lo, hi = float(a) if a.is_finite else (-np.inf if a == -sp.oo else np.inf), float(b) if b.is_finite else (np.inf if b == sp.oo else -np.inf)
    cv = CrossValidation("integral", f"∫ {expr} d{var} from {lower} to {upper}" + (f"  with {vals}" if vals else ""), None, None, None, None, "scipy.integrate.quad", None, 1e-8, "")
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", IntegrationWarning)
            val, err = quad(lambda t: float(fn(t)), lo, hi, limit=400, epsabs=1e-13, epsrel=1e-12)
        cv.numeric_value = val
        cv.tolerance = max(1e-8 * abs(val), 50 * err, 1e-10)
        if caught:
            cv.detail += f" quad warned: {str(caught[0].message)[:90]}."
            cv.tolerance = max(cv.tolerance, 1e-5 * abs(val))
    except Exception as exc:  # noqa: BLE001
        cv.detail = f"the numerical integral failed ({exc})"
    try:
        res = run_with_timeout(lambda: sp.integrate(f, (x, a, b)), timeout=SYM_TIMEOUT, label="integrate")
    except ComputationTimeoutError:
        res = None
        cv.detail += " SymPy timed out."
    except Exception as exc:  # noqa: BLE001
        res = None
        cv.detail += f" SymPy failed ({exc})."
    if res is not None and not res.has(sp.Integral):
        cv.symbolic, cv.symbolic_latex = str(res), sp.latex(res)
        sv = _real(res.subs(sub))
        if sv is not None:
            cv.symbolic_value = sv
        else:
            cv.detail += " The closed form is not a real number for these parameters."
    elif res is not None:
        cv.detail += " SymPy returned no closed form (an unevaluated Integral)."
    if cv.symbolic_value is not None and cv.numeric_value is not None:
        cv.error = abs(float(cv.symbolic_value) - cv.numeric_value)
    return _finish(cv)


# ------------------------------------------------------------------------------------------ derivative

def validate_derivative(expr: str, var: str, point: float, order: int = 1, values: dict[str, float] | None = None) -> CrossValidation:
    if not 1 <= order <= 4:
        raise CrossCuttingError("order must be between 1 and 4.")
    syms, vals = _symbols(expr, exclude={var}, values=values)
    x = sp.Symbol(var, real=True)
    f = parse_with(expr, {**syms, var: x})
    sub = {syms[k]: v for k, v in vals.items()}
    cv = CrossValidation("derivative", f"d^{order}/d{var}^{order} of {expr} at {var} = {point:g}" + (f"  with {vals}" if vals else ""), None, None, None, None,
                         "mpmath.diff (40-digit finite differences of the original expression)", None, 1e-9, "")
    d = sp.diff(f, x, order)
    cv.symbolic, cv.symbolic_latex = str(d), sp.latex(d)
    sv = _real(d.subs(sub).subs(x, point))
    cv.symbolic_value = sv
    mpmath.mp.dps = 40
    g = sp.lambdify(x, f.subs(sub), "mpmath")
    try:
        cv.numeric_value = float(mpmath.diff(g, point, order))
    except Exception as exc:  # noqa: BLE001
        cv.detail = f"numerical differentiation failed ({exc})"
    if sv is not None and cv.numeric_value is not None:
        cv.error = abs(sv - cv.numeric_value)
        cv.tolerance = max(1e-9 * abs(sv), 1e-9)
    return _finish(cv)


# ------------------------------------------------------------------------------------------ roots

def validate_roots(expr: str, var: str, lo: float, hi: float, values: dict[str, float] | None = None) -> CrossValidation:
    syms, vals = _symbols(expr, exclude={var}, values=values)
    x = sp.Symbol(var, real=True)
    f = parse_with(expr, {**syms, var: x})
    sub = {syms[k]: v for k, v in vals.items()}
    fn = sp.lambdify(x, f.subs(sub), "numpy")
    cv = CrossValidation("roots", f"{expr} = 0 for {var} in [{lo:g}, {hi:g}]" + (f"  with {vals}" if vals else ""), None, None, None, None,
                         "sign-change scan (20000 points) + Brent's method", None, 1e-8, "")
    grid = np.linspace(lo, hi, 20001)
    with np.errstate(all="ignore"):
        ys = np.array([float(fn(t)) for t in grid])
    num = []
    for i in range(len(grid) - 1):
        if np.isfinite(ys[i]) and np.isfinite(ys[i + 1]) and ys[i] * ys[i + 1] < 0 and abs(ys[i]) < 1e12:
            r = brentq(lambda t: float(fn(t)), grid[i], grid[i + 1], xtol=1e-14)
            if abs(float(fn(r))) < 1e-6 * (1 + abs(ys[i]) + abs(ys[i + 1])):                  # not a pole
                num.append(r)
    cv.numeric_value = num
    try:
        sols = run_with_timeout(lambda: sp.solve(f.subs(sub), x), timeout=SYM_TIMEOUT, label="solve")
    except Exception as exc:  # noqa: BLE001
        sols = None
        cv.detail += f" SymPy failed ({exc})."
    if sols is not None:
        real = sorted({float(v) for v in (_real(s) for s in sols) if v is not None and lo <= v <= hi})
        cv.symbolic = ", ".join(str(s) for s in sols)
        cv.symbolic_latex = ", ".join(sp.latex(s) for s in sols)
        cv.symbolic_value = real
    if cv.symbolic_value is not None:
        sym_roots: list[float] = list(cv.symbolic_value)
        residuals = [abs(float(fn(r))) for r in sym_roots]
        unmatched_num = [r for r in num if not any(abs(r - s) < 1e-7 * (1 + abs(r)) for s in sym_roots)]
        unmatched_sym = [s for s, res in zip(sym_roots, residuals) if not any(abs(s - r) < 1e-7 * (1 + abs(s)) for r in num) and res > 1e-9]
        touching = [s for s in sym_roots if not any(abs(s - r) < 1e-7 * (1 + abs(s)) for r in num)]
        cv.error = 0.0 if not unmatched_num and not unmatched_sym else 1.0
        cv.tolerance = 0.5
        if touching and not unmatched_sym:
            cv.detail += f" {len(touching)} symbolic root(s) with no sign change (a double root touching zero): residual ≈ 0, so they are genuine."
        if unmatched_num:
            cv.detail += f" Numerical roots SymPy did not list: {unmatched_num}."
        if unmatched_sym:
            cv.detail += f" Symbolic roots with a non-zero residual: {unmatched_sym}."
    return _finish(cv)


# ------------------------------------------------------------------------------------------ ODE

def validate_ode(equation: str, dependent: str, indep: str, initial: dict[str, float], t_end: float, values: dict[str, float] | None = None) -> CrossValidation:
    """The equation in the app's subscript notation (x_tt + 3*x_t + 2*x = 0), initial values keyed x, x_t, ... at 0, compared on [0, t_end]."""
    if equation.count("=") != 1:
        raise CrossCuttingError("Write the equation with exactly one '='.")
    lhs_t, rhs_t = (s.strip() for s in equation.split("="))
    names = names_in(lhs_t) | names_in(rhs_t)
    deriv = sorted(n for n in names if n.startswith(dependent + "_") and set(n[len(dependent) + 1:]) == {indep})
    params = sorted(names - set(deriv) - {dependent, indep})
    vals = {p: float((values or {}).get(p, 1.0)) for p in params}
    t = sp.Symbol(indep, real=True)
    xf = sp.Function(dependent)
    local: dict[str, sp.Symbol] = {p: sp.Symbol(p, real=True) for p in params}
    local[indep] = t
    chain_names = [dependent] + [f"{dependent}_{indep * k}" for k in range(1, max([len(n) - len(dependent) - 1 for n in deriv] + [0]) + 1)]
    placeholder = {n: sp.Symbol(n, real=True) for n in chain_names}
    expr = parse_with(lhs_t, {**local, **placeholder}) - parse_with(rhs_t, {**local, **placeholder})
    order = max([len(n) - len(dependent) - 1 for n in deriv] + [0])
    if order < 1:
        raise CrossCuttingError("The equation must contain a derivative of the dependent variable (x_t, x_tt, ...).")
    chain = [dependent] + [f"{dependent}_{indep * k}" for k in range(1, order + 1)]
    missing = [n for n in chain[:-1] if n not in initial]
    if missing:
        raise CrossCuttingError("Give initial values for: " + ", ".join(missing))
    sub = {local[p]: vals[p] for p in params}
    cv = CrossValidation("ode", f"{equation} with {initial}" + (f", {vals}" if vals else ""), None, None, None, None, "scipy solve_ivp (DOP853, rtol 1e-12)", None, 1e-6, "")
    top = placeholder[chain[-1]]
    sol_top = sp.solve(expr.subs(sub), top)
    if len(sol_top) != 1:
        raise CrossCuttingError("The equation cannot be solved uniquely for its highest derivative.")
    f_num = sp.lambdify([t] + [placeholder[n] for n in chain[:-1]], sol_top[0], "numpy")
    ts = np.linspace(0, t_end, 101)
    so = solve_ivp(lambda tt, y: list(y[1:]) + [float(f_num(tt, *y))], (0, t_end), [initial[n] for n in chain[:-1]], t_eval=ts, rtol=1e-12, atol=1e-14, method="DOP853")
    cv.numeric_value = so.y[0] if so.success else None
    # symbolic
    ode_sym = sp.Eq(sum(0 for _ in []) + _to_derivatives(expr, dependent, indep, xf, t, placeholder, chain).subs(sub), 0)
    ics = {xf(0) if k == 0 else sp.Derivative(xf(t), (t, k)).subs(t, 0): initial[chain[k]] for k in range(order)}
    try:
        sol = run_with_timeout(lambda: sp.dsolve(ode_sym, xf(t), ics=ics), timeout=SYM_TIMEOUT, label="dsolve")
        rhs = sol.rhs if isinstance(sol, sp.Eq) else sol
        cv.symbolic, cv.symbolic_latex = str(rhs), sp.latex(sol)
        g = sp.lambdify(t, rhs, "numpy")
        cv.symbolic_value = np.broadcast_to(np.asarray(g(ts), dtype=complex).real, ts.shape).copy()
    except Exception as exc:  # noqa: BLE001
        cv.detail += f" SymPy dsolve gave no closed form ({type(exc).__name__}: {str(exc)[:80]})."
    if cv.symbolic_value is not None and cv.numeric_value is not None:
        cv.error = float(np.max(np.abs(np.asarray(cv.symbolic_value) - np.asarray(cv.numeric_value))))
        cv.tolerance = 1e-6 * max(1.0, float(np.max(np.abs(cv.numeric_value))))
    return _finish(cv)


def _to_derivatives(expr: sp.Expr, dep: str, indep: str, xf, t: sp.Symbol, placeholder: dict[str, sp.Symbol], chain: list[str]) -> sp.Expr:
    rep: dict[sp.Symbol, sp.Expr] = {placeholder[chain[0]]: xf(t)}
    for k in range(1, len(chain)):
        rep[placeholder[chain[k]]] = sp.Derivative(xf(t), (t, k))
    return expr.xreplace({p: rep[p] for p in rep if p in expr.free_symbols})


# ------------------------------------------------------------------------------------------ sums, limits, eigenvalues

def validate_sum(expr: str, var: str, lower: str, upper: str, values: dict[str, float] | None = None) -> CrossValidation:
    syms, vals = _symbols(expr, exclude={var}, values=values)
    k = sp.Symbol(var, integer=True)
    f = parse_with(expr, {**syms, var: k})
    a, b = _bound(lower), _bound(upper)
    sub = {syms[n]: v for n, v in vals.items()}
    cv = CrossValidation("sum", f"Σ {expr}, {var} = {lower} … {upper}" + (f"  with {vals}" if vals else ""), None, None, None, None,
                         "mpmath.nsum" if b == sp.oo else "direct partial sum (mpmath, 40 digits)", None, 1e-9, "")
    mpmath.mp.dps = 40
    g = sp.lambdify(k, f.subs(sub), "mpmath")
    try:
        cv.numeric_value = float(mpmath.nsum(g, [int(a), mpmath.inf])) if b == sp.oo else float(mpmath.fsum(g(i) for i in range(int(a), int(b) + 1)))
    except Exception as exc:  # noqa: BLE001
        cv.detail = f"the numerical sum failed ({exc})"
    try:
        res = run_with_timeout(lambda: sp.summation(f, (k, a, b)), timeout=SYM_TIMEOUT, label="summation")
        if not res.has(sp.Sum):
            cv.symbolic, cv.symbolic_latex = str(res), sp.latex(res)
            cv.symbolic_value = _real(res.subs(sub))
        else:
            cv.detail += " SymPy returned no closed form (an unevaluated Sum)."
    except Exception as exc:  # noqa: BLE001
        cv.detail += f" SymPy failed ({exc})."
    if cv.symbolic_value is not None and cv.numeric_value is not None:
        cv.error = abs(float(cv.symbolic_value) - cv.numeric_value)
        cv.tolerance = max(1e-9 * abs(cv.numeric_value), 1e-10)
    return _finish(cv)


def validate_limit(expr: str, var: str, point: str, direction: str = "+", values: dict[str, float] | None = None) -> CrossValidation:
    syms, vals = _symbols(expr, exclude={var}, values=values)
    x = sp.Symbol(var, real=True)
    f = parse_with(expr, {**syms, var: x})
    p = _bound(point)
    sub = {syms[n]: v for n, v in vals.items()}
    cv = CrossValidation("limit", f"lim {expr} as {var} → {point}{direction if p.is_finite else ''}" + (f"  with {vals}" if vals else ""), None, None, None, None,
                         "40-digit evaluation at 10^-k from the point (or 10^k for infinity)", None, 1e-8, "")
    mpmath.mp.dps = 50
    g = sp.lambdify(x, f.subs(sub), "mpmath")
    seq = []
    for kk in (12, 16, 20, 24):
        if p == sp.oo:
            pt = mpmath.mpf(10) ** kk
        elif p == -sp.oo:
            pt = -mpmath.mpf(10) ** kk
        else:
            pt = mpmath.mpf(float(p)) + (1 if direction == "+" else -1) * mpmath.mpf(10) ** (-kk)
        try:
            seq.append(g(pt))
        except Exception:  # noqa: BLE001
            seq = []
            break
    if seq and all(abs(s - seq[-1]) < 1e-6 * (1 + abs(seq[-1])) for s in seq[-2:]) and mpmath.isfinite(seq[-1]):
        cv.numeric_value = float(mpmath.re(seq[-1]))
    else:
        cv.detail += " The numerical sequence does not settle (divergent or oscillating)."
    try:
        res = run_with_timeout(lambda: sp.limit(f.subs(sub), x, p, direction if p.is_finite else "+"), timeout=SYM_TIMEOUT, label="limit")
        cv.symbolic, cv.symbolic_latex = str(res), sp.latex(res)
        cv.symbolic_value = _real(res) if res.is_finite else None
        if res.is_infinite:
            cv.detail += f" SymPy: the limit is {res} (diverges)."
    except Exception as exc:  # noqa: BLE001
        cv.detail += f" SymPy failed ({exc})."
    if cv.symbolic_value is not None and cv.numeric_value is not None:
        cv.error = abs(float(cv.symbolic_value) - cv.numeric_value)
        cv.tolerance = 1e-8 * (1 + abs(cv.numeric_value))
    return _finish(cv)


def validate_eigenvalues(rows: list[list[str]]) -> CrossValidation:
    n = len(rows)
    if n < 2 or n > 6 or any(len(r) != n for r in rows):
        raise CrossCuttingError("Give a square matrix between 2×2 and 6×6.")
    M = sp.Matrix(n, n, lambda i, j: parse_with(str(rows[i][j]), {}))
    cv = CrossValidation("eigenvalues", f"eigenvalues of {M.tolist()}", None, None, None, None, "numpy.linalg.eigvals" + (" (eigvalsh, symmetric)" if M == M.T else ""), None, 1e-8, "")
    A = np.array(M.evalf().tolist(), dtype=complex)
    ev = np.linalg.eigvalsh(A.real) if M == M.T and np.allclose(A.imag, 0) else np.linalg.eigvals(A)
    cv.numeric_value = sorted(ev.tolist(), key=lambda z: (round(complex(z).real, 8), round(complex(z).imag, 8)))
    try:
        eig = run_with_timeout(lambda: M.eigenvals(), timeout=SYM_TIMEOUT, label="eigenvals")
        flat = [v for val, mult in eig.items() for v in [val] * mult]
        cv.symbolic = ", ".join(str(sp.simplify(v)) for v in flat)
        cv.symbolic_latex = ", ".join(sp.latex(sp.simplify(v)) for v in flat)
        vals_c = sorted((complex(sp.N(v)) for v in flat), key=lambda z: (round(z.real, 8), round(z.imag, 8)))     # rounding: 3e-17 must not reorder +-i
        cv.symbolic_value = vals_c
        if len(vals_c) == n:
            cv.error = float(max(abs(a - b) for a, b in zip(vals_c, cv.numeric_value)))
            cv.tolerance = 1e-8 * (1 + max(abs(v) for v in vals_c))
        else:
            cv.detail += " SymPy found fewer eigenvalues than the matrix size (roots it cannot express exactly)."
    except Exception as exc:  # noqa: BLE001
        cv.detail += f" SymPy failed ({exc})."
    return _finish(cv)


# ------------------------------------------------------------------------------------------ the demonstration set

def demonstration_suite() -> list[CrossValidation]:
    """One of each task, including one where SymPy has no closed form (numeric only) -- used by the UI's 'run all' and the tests."""
    return [
        validate_integral("exp(-x^2)*cos(a*x)", "x", "-oo", "oo", {"a": 1.5}),
        validate_integral("exp(-x)*sin(x)/x", "x", "0", "oo"),
        validate_integral("exp(sin(x))", "x", "0", "1"),
        validate_derivative("x^x*sin(x)", "x", 1.3, 2),
        validate_roots("x^3 - 6*x^2 + 11*x - 6", "x", -1, 5),
        validate_roots("cos(x) - x", "x", -2, 2),
        validate_ode("x_tt + 3*x_t + 2*x = 0", "x", "t", {"x": 1.0, "x_t": 0.0}, 5.0),
        validate_sum("1/k^2", "k", "1", "oo"),
        validate_sum("k*r^k", "k", "0", "40", {"r": 0.9}),
        validate_limit("sin(x)/x", "x", "0"),
        validate_limit("(1 + 1/x)^x", "x", "oo"),
        validate_eigenvalues([["2", "1", "0"], ["1", "2", "1"], ["0", "1", "2"]]),
    ]
