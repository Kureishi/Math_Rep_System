"""
Curvature invariants and field equations for any metric (modules/gr_metrics.GRMetric) -- Streamlit-free.

    Christoffel symbols -> Riemann -> Ricci tensor / scalar -> Einstein tensor G_ab -> Weyl tensor C_abcd
    invariants:  R,  R_ab R^ab,  Kretschmann K = R_abcd R^abcd,  Weyl squared C_abcd C^abcd
    field equations:  G_ab + Lambda g_ab = 8 pi T_ab  (vacuum when T = 0), electromagnetic and effective-fluid T
    singularities:  where do the invariants blow up, as opposed to where only the COORDINATES misbehave?

WHY A NEW ENGINE (rather than modules/tensor_calculus.py, which stays as the general-dimension front end):
that module calls sp.simplify on every component, which is fine for a sphere and hopeless for Kerr. This one
simplifies with `simplify_expr`: sin/cos of a coordinate (and sinh/cosh) are replaced by two fresh symbols,
the expression is cancelled to a single rational function, and the identity s^2 + c^2 = 1 is applied as a
polynomial remainder. That is a CANONICAL form for the rational-trigonometric expressions GR metrics are made
of, so "is this exactly zero?" has a definite answer instead of "simplify() did not manage it". It brings a
full Kerr curvature tensor to about ten seconds. The two engines are cross-checked against each other in the
tests (Schwarzschild, sphere) -- different code, same answer.

VERIFICATION, in the spirit of the rest of the app. Every curvature computation also runs identities that hold
for ANY metric, so a wrong Christoffel symbol or a dropped term shows up as a failed check rather than a
plausible-looking wrong tensor: metric compatibility (nabla g = 0), the Riemann symmetries (antisymmetry, pair
symmetry, first Bianchi), symmetry of Ricci, the CONTRACTED SECOND BIANCHI IDENTITY nabla_a G^a_b = 0, the
tracelessness of Weyl, and the decomposition K = C^2 + (4/(n-2)) R_ab R^ab - 2 R^2/((n-1)(n-2)) between four
quantities computed by separate contractions. Zero tests are exact (canonical form) where possible; where the
canonical form does not close, a residual is evaluated at sample points and the result says it was numerical.

Sign convention: MTW (see gr_metrics): the 2-sphere has positive Ricci scalar.
"""
import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.gr_metrics import GR_TIMEOUT, GRInputError, GRMetric
from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList
from modules.timeout_utils import run_with_timeout

Partial = Callable[[sp.Expr, int], sp.Expr]

# ------------------------------------------------------------------------------------------ simplification

_TRIG = ((sp.sin, sp.cos, "t"), (sp.sinh, sp.cosh, "h"))


def _pair_symbols(arg: sp.Symbol, kind: str) -> tuple[sp.Symbol, sp.Symbol]:
    return sp.Symbol(f"_S{kind}_{arg.name}", real=True), sp.Symbol(f"_C{kind}_{arg.name}", real=True)


def simplify_expr(e: sp.Expr) -> sp.Expr:
    """Canonical simplification for rational-trigonometric expressions (see the module docstring). Falls back
    to sp.simplify when the expression contains functions of something other than a bare coordinate."""
    e = sp.sympify(e)
    if e == 0:
        return sp.Integer(0)
    e = e.replace(sp.tan, lambda x: sp.sin(x) / sp.cos(x)).replace(sp.cot, lambda x: sp.cos(x) / sp.sin(x))
    e = e.replace(sp.tanh, lambda x: sp.sinh(x) / sp.cosh(x))
    back: dict[sp.Symbol, sp.Expr] = {}
    fwd: dict[sp.Basic, sp.Symbol] = {}
    reductions: list[tuple[sp.Symbol, sp.Expr]] = []
    for sfun, cfun, kind in _TRIG:
        atoms = list(e.atoms(sfun, cfun))
        if any(not a.args[0].is_Symbol for a in atoms):
            e = sp.expand_trig(e) if kind == "t" else e.rewrite(sp.exp)
            atoms = list(e.atoms(sfun, cfun))
            if any(not a.args[0].is_Symbol for a in atoms):
                return sp.simplify(e)
        for a in atoms:
            x = a.args[0]
            s, c = _pair_symbols(x, kind)
            fwd[a] = s if a.func == sfun else c
            back[s], back[c] = sfun(x), cfun(x)      # both, even if only one occurs: reduction can introduce the other
        for x in sorted({a.args[0] for a in atoms}, key=str):
            s, c = _pair_symbols(x, kind)
            # trig: s^2 = 1 - c^2 (eliminate s^2);  hyperbolic: c^2 = 1 + s^2 (eliminate c^2)
            reductions.append((s, 1 - c ** 2) if kind == "t" else (c, 1 + s ** 2))
    if fwd:
        e = e.xreplace(fwd)
    e = sp.cancel(sp.together(e))
    if reductions:
        num, den = sp.fraction(e)
        for sym, repl in reductions:
            num, den = _reduce(num, sym, repl), _reduce(den, sym, repl)
        e = sp.cancel(num / den)
    if len(e.free_symbols) > 0 and sp.count_ops(e) < 400:
        try:
            e = sp.factor(e)
        except TypeError:       # sorting factors with symbolic exponents (t**(2p)) can be undecidable: factoring is cosmetic
            pass
    for sym, repl in reductions:                  # (c - 1)(c + 1) -> -s^2, which is what a human would write
        if repl.has(sym) is False and repl.is_Add and (1 - repl).is_Pow:
            e = _prettify(e, sym, next(iter(repl.free_symbols)))
    return e.xreplace(back) if back else e


def _prettify(e: sp.Expr, s: sp.Symbol, c: sp.Symbol) -> sp.Expr:
    if not e.has(c):
        return e
    num, den = sp.fraction(e)
    out = []
    for part in (num, den):
        if part.is_number:
            out.append(part)
            continue
        try:
            coeff, facs = sp.factor_list(part)
        except (sp.PolynomialError, TypeError):       # cosmetic step only: leave the expression as it is
            return e
        d = {f: k for f, k in facs}
        k = min(d.get(c - 1, 0), d.get(c + 1, 0))
        if k:
            d[c - 1] -= k
            d[c + 1] -= k
            coeff = coeff * (-1) ** k
            d[s] = d.get(s, 0) + 2 * k
        out.append(coeff * sp.Mul(*[f ** m for f, m in d.items() if m]))
    return out[0] / out[1]


def _reduce(p: sp.Expr, sym: sp.Symbol, repl: sp.Expr) -> sp.Expr:
    """p modulo sym^2 - repl, as a polynomial in sym."""
    p = sp.expand(p)
    if not p.has(sym):
        return p
    return sp.rem(sp.Poly(p, sym), sp.Poly(sym ** 2 - repl, sym)).as_expr()


# ------------------------------------------------------------------------------------------ numerics

def numeric_subs(metric: GRMetric, params: dict[str, float] | None = None) -> dict[sp.Symbol, float]:
    """Every coordinate and parameter of `metric` mapped to a number: the sample point and the defaults."""
    subs: dict[sp.Symbol, float] = dict(zip(metric.coords, metric.sample_point()))
    subs.update(metric.param_subs(params))
    return subs


def _jittered(subs: dict[sp.Symbol, float], coords: tuple[sp.Symbol, ...], k: int, seed: int = 20260) -> dict[sp.Symbol, float]:
    rng = random.Random(seed + k)
    out = dict(subs)
    for c in coords:
        out[c] = subs[c] * (1 + 0.15 * (rng.random() - 0.5)) if k else subs[c]
    return out


def evaluate(expr: sp.Expr, subs: dict[sp.Symbol, float]) -> float:
    """expr at a point, as a float (nan if not finite or not real)."""
    try:
        val = complex(sp.sympify(expr).xreplace({k: sp.Float(v) for k, v in subs.items()}).evalf(30))
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")
    if abs(val.imag) > 1e-9 * (1 + abs(val.real)):
        return float("nan")
    return val.real


def is_zero(expr: sp.Expr, subs_points: list[dict[sp.Symbol, float]] | None = None) -> tuple[bool, bool]:
    """(is zero, decided only numerically). Exact first; if the canonical form is not 0, sample points decide
    (an expression with a tiny but genuinely nonzero residual is still reported as nonzero: the tolerance is
    1e-9 relative to the size of the terms)."""
    if expr == 0:
        return True, False
    try:
        if simplify_expr(expr) == 0:
            return True, False
    except Exception:  # noqa: BLE001
        pass
    if not subs_points:
        return False, False
    vals = [evaluate(expr, s) for s in subs_points]
    finite = [v for v in vals if math.isfinite(v)]
    if not finite:
        return False, False
    scale = max([1.0] + [abs(evaluate(t, s)) for s in subs_points[:1] for t in sp.Add.make_args(expr)
                         if math.isfinite(evaluate(t, s))])
    return all(abs(v) < 1e-9 * scale for v in finite), True


# ------------------------------------------------------------------------------------------ the result type

@dataclass
class Curvature:
    metric: GRMetric | None
    n: int
    g: sp.Matrix
    ginv: sp.Matrix
    christoffel: list                 # [k][i][j] = Gamma^k_ij
    riemann: list                     # [r][s][m][n] = R^r_{smn}
    ricci: sp.Matrix                  # R_ab
    ricci_scalar: sp.Expr
    einstein: sp.Matrix               # G_ab
    weyl_mixed: list | None           # [a][b][c][d] = C^{ab}_{cd}  (None for n < 3)
    invariants: dict[str, sp.Expr]
    checks: CheckList
    seconds: dict[str, float] = field(default_factory=dict)
    numeric_decisions: int = 0        # how many zero tests were settled by sample points, not exact algebra
    sample_subs: dict[sp.Symbol, float] = field(default_factory=dict)

    @property
    def is_flat(self) -> bool:
        return all(self.riemann[a][b][c][d] == 0 for a in range(self.n) for b in range(self.n)
                   for c in range(self.n) for d in range(self.n))

    @property
    def is_ricci_flat(self) -> bool:
        return all(self.ricci[i, j] == 0 for i in range(self.n) for j in range(self.n))

    def einstein_mixed(self) -> sp.Matrix:
        return (self.ginv * self.einstein).applyfunc(simplify_expr)


def invert_metric(g: sp.Matrix) -> sp.Matrix:
    """g^-1, simplified. A diagonal metric is inverted entry by entry; otherwise SymPy's Gauss elimination is tried
    and, if it cannot decide a pivot (it raises TypeError on symbolic powers such as t**(2p), and only for some hash
    seeds), the adjugate / determinant formula, which needs no pivoting, is used."""
    n = g.shape[0]
    if all(g[i, j] == 0 for i in range(n) for j in range(n) if i != j):
        if any(g[i, i] == 0 for i in range(n)):
            raise GRInputError("a diagonal entry of the metric is identically zero, so it is degenerate")
        return sp.diag(*[1 / g[i, i] for i in range(n)]).applyfunc(simplify_expr)
    try:
        return g.inv().applyfunc(simplify_expr)
    except TypeError:
        return (g.adjugate() / g.det(method="berkowitz")).applyfunc(simplify_expr)


def _default_partial(coords: tuple[sp.Symbol, ...]) -> Partial:
    return lambda expr, i: sp.diff(expr, coords[i])


def compute_curvature(metric: GRMetric, *, progress: ProgressFn | None = None, timeout: float = GR_TIMEOUT,
                      params: dict[str, float] | None = None) -> Curvature:
    """Full curvature of `metric`. Raises GRInputError for a degenerate metric and ComputationTimeoutError if a
    stage exceeds `timeout` (each stage separately). `progress` is called between stages, never inside one."""
    subs = numeric_subs(metric, params)
    return curvature_from_components(metric.g, metric.dim, _default_partial(metric.coords), metric.coords,
                                     subs, metric=metric, progress=progress, timeout=timeout)


def curvature_from_components(g: sp.Matrix, n: int, partial: Partial, chart: tuple[sp.Symbol, ...],
                              sample_subs: dict[sp.Symbol, float], *, metric: GRMetric | None = None,
                              progress: ProgressFn | None = None, timeout: float = GR_TIMEOUT) -> Curvature:
    """The engine proper. `partial(expr, i)` is the derivative of expr in the i-th direction of the basis the
    components `g` refer to: an ordinary d/dx^i for a normal chart, a chain-rule combination for components
    expressed in another chart's variables (gr_transforms uses that to find the invariants of a metric whose
    coordinate change cannot be inverted in closed form)."""
    import time
    seconds: dict[str, float] = {}
    stages = ["inverse", "Christoffel symbols", "Riemann tensor", "Ricci and Einstein tensors", "Weyl and invariants",
              "identity checks"]

    def stage(index: int, fn: Callable, *args):
        report(progress, stages[index], index / len(stages))
        t0 = time.time()
        out = run_with_timeout(fn, *args, timeout=timeout, label=stages[index])
        seconds[stages[index]] = time.time() - t0
        return out

    try:
        ginv = stage(0, invert_metric, g)
    except Exception as exc:  # noqa: BLE001
        if type(exc).__name__.startswith("Computation"):
            raise
        raise GRInputError(f"The metric could not be inverted (is it degenerate?): {exc}") from None
    gamma = stage(1, _christoffel, g, ginv, n, partial)
    riem = stage(2, _riemann, gamma, n, partial)
    ricci, rscalar, einstein = stage(3, _ricci_einstein, riem, ginv, g, n)
    weyl, invariants = stage(4, _weyl_and_invariants, riem, ricci, rscalar, ginv, n)
    curv = Curvature(metric, n, g, ginv, gamma, riem, ricci, rscalar, einstein, weyl, invariants, CheckList(),
                     seconds, 0, sample_subs)
    stage(5, _run_checks, curv, partial, chart)
    report(progress, "done", 1.0)
    return curv


# ------------------------------------------------------------------------------------------ the tensors

def _christoffel(g: sp.Matrix, ginv: sp.Matrix, n: int, partial: Partial) -> list:
    dg = [[[partial(g[a, b], c) for c in range(n)] for b in range(n)] for a in range(n)]    # dg[a][b][c] = d_c g_ab
    gam = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                s = sp.Integer(0)
                for l in range(n):
                    if ginv[k, l] != 0:
                        s += ginv[k, l] * (dg[l][j][i] + dg[l][i][j] - dg[i][j][l])
                gam[k][i][j] = gam[k][j][i] = simplify_expr(s / 2)
    return gam


def _riemann(gam: list, n: int, partial: Partial) -> list:
    """R^r_{smn} = d_m G^r_{ns} - d_n G^r_{ms} + G^r_{ml} G^l_{ns} - G^r_{nl} G^l_{ms}"""
    riem = [[[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for r in range(n):
        for s in range(n):
            for m in range(n):
                for k in range(m + 1, n):
                    term = partial(gam[r][k][s], m) - partial(gam[r][m][s], k)
                    for l in range(n):
                        term += gam[r][m][l] * gam[l][k][s] - gam[r][k][l] * gam[l][m][s]
                    v = simplify_expr(term)
                    riem[r][s][m][k] = v
                    riem[r][s][k][m] = -v
    return riem


def _ricci_einstein(riem: list, ginv: sp.Matrix, g: sp.Matrix, n: int):
    ricci = sp.Matrix(n, n, lambda s, k: simplify_expr(sum(riem[r][s][r][k] for r in range(n))))
    rscalar = simplify_expr(sum(ginv[i, j] * ricci[i, j] for i in range(n) for j in range(n)))
    einstein = (ricci - rscalar * g / 2).applyfunc(simplify_expr)
    return ricci, rscalar, einstein


def _weyl_and_invariants(riem: list, ricci: sp.Matrix, rscalar: sp.Expr, ginv: sp.Matrix, n: int):
    # R^{ab}_{cd} = g^{bl} R^a_{l c d}
    S = [[[[simplify_expr(sum(ginv[b, l] * riem[a][l][c][d] for l in range(n) if ginv[b, l] != 0))
            for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
    kretsch = simplify_expr(sum(S[a][b][c][d] * S[c][d][a][b] for a in range(n) for b in range(n)
                                for c in range(n) for d in range(n)))
    ric_mixed = (ginv * ricci).applyfunc(simplify_expr)                    # R^a_b
    ric2 = simplify_expr(sum(ric_mixed[a, b] * ric_mixed[b, a] for a in range(n) for b in range(n)))
    invariants = {"Ricci scalar R": rscalar, "Ricci squared R_ab R^ab": ric2, "Kretschmann R_abcd R^abcd": kretsch}
    weyl = None
    if n >= 3:
        d = lambda a, c: 1 if a == c else 0  # noqa: E731
        weyl = [[[[simplify_expr(
            S[a][b][c][e]
            - sp.Rational(1, n - 2) * (d(a, c) * ric_mixed[b, e] - d(a, e) * ric_mixed[b, c]
                                       - d(b, c) * ric_mixed[a, e] + d(b, e) * ric_mixed[a, c])
            + rscalar * sp.Rational(1, (n - 1) * (n - 2)) * (d(a, c) * d(b, e) - d(a, e) * d(b, c)))
            for e in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]
        invariants["Weyl squared C_abcd C^abcd"] = simplify_expr(sum(
            weyl[a][b][c][e] * weyl[c][e][a][b] for a in range(n) for b in range(n) for c in range(n)
            for e in range(n)))
    return weyl, invariants


# ------------------------------------------------------------------------------------------ identity checks

def _points(curv: Curvature, chart: tuple[sp.Symbol, ...], k: int = 3) -> list[dict[sp.Symbol, float]]:
    return [_jittered(curv.sample_subs, chart, i) for i in range(k)]


def _all_zero(curv: Curvature, exprs: list[sp.Expr], pts: list[dict[sp.Symbol, float]]) -> tuple[bool, str]:
    numeric = 0
    for e in exprs:
        ok, was_numeric = is_zero(e, pts)
        if not ok:
            return False, "a component is not zero"
        numeric += was_numeric
    curv.numeric_decisions += numeric
    return True, ("exact" if not numeric else f"exact, except {numeric} of {len(exprs)} settled at sample points")


def _run_checks(curv: Curvature, partial: Partial, chart: tuple[sp.Symbol, ...]) -> None:
    n, g, gam, riem = curv.n, curv.g, curv.christoffel, curv.riemann
    pts = _points(curv, chart)
    ck = curv.checks

    # signature at a regular sample point
    try:
        mat = np.array([[evaluate(g[i, j], curv.sample_subs) for j in range(n)] for i in range(n)])
        ev = np.linalg.eigvalsh(mat)
        neg = int((ev < 0).sum())
        ck.add("Lorentzian signature at the sample point", neg == 1 and bool((ev != 0).all()),
               f"{neg} negative and {n - neg} positive eigenvalues of g (expected 1 and {n - 1})")
    except (ValueError, np.linalg.LinAlgError):
        ck.add("Lorentzian signature at the sample point", False, "the metric could not be evaluated at the sample point")

    exprs = [partial(g[i, j], k) - sum(gam[l][k][i] * g[l, j] + gam[l][k][j] * g[i, l] for l in range(n))
             for k in range(n) for i in range(n) for j in range(i, n)]
    ok, how = _all_zero(curv, exprs, pts)
    ck.add("Metric compatibility  ∇g = 0", ok, how)

    low = lambda a, b, c, d: sum(g[a, l] * riem[l][b][c][d] for l in range(n) if g[a, l] != 0)  # noqa: E731
    pair, anti, bianchi = [], [], []
    for a in range(n):
        for b in range(a + 1, n):
            for c in range(n):
                for d in range(c + 1, n):
                    pair.append(low(a, b, c, d) - low(c, d, a, b))
        for b in range(n):
            for c in range(n):
                for d in range(c + 1, n):
                    anti.append(low(a, b, c, d) + low(b, a, c, d))
    for a in range(n):
        for b in range(n):
            for c in range(b + 1, n):
                for d in range(c + 1, n):
                    bianchi.append(low(a, b, c, d) + low(a, c, d, b) + low(a, d, b, c))
    ok, how = _all_zero(curv, pair + anti, pts)
    ck.add("Riemann symmetries  R_abcd = -R_bacd = R_cdab", ok, how)
    ok, how = _all_zero(curv, bianchi, pts)
    ck.add("First Bianchi identity  R_a[bcd] = 0", ok, how)

    ok, how = _all_zero(curv, [curv.ricci[i, j] - curv.ricci[j, i] for i in range(n) for j in range(i + 1, n)], pts)
    ck.add("Ricci tensor is symmetric", ok, how)

    gm = (curv.ginv * curv.einstein).applyfunc(simplify_expr)                       # G^a_b
    div = [sum(partial(gm[a, b], a) for a in range(n))
           + sum(gam[a][a][c] * gm[c, b] for a in range(n) for c in range(n))
           - sum(gam[c][a][b] * gm[a, c] for a in range(n) for c in range(n)) for b in range(n)]
    ok, how = _all_zero(curv, div, pts)
    ck.add("Contracted Bianchi identity  ∇_a G^a_b = 0", ok, how)

    if curv.weyl_mixed is not None:
        w = curv.weyl_mixed
        ok, how = _all_zero(curv, [sum(w[a][b][a][d] for a in range(n)) for b in range(n) for d in range(n)], pts)
        ck.add("Weyl tensor is traceless", ok, how)
        inv = curv.invariants
        rhs = (inv["Weyl squared C_abcd C^abcd"] + sp.Rational(4, n - 2) * inv["Ricci squared R_ab R^ab"]
               - sp.Rational(2, (n - 1) * (n - 2)) * inv["Ricci scalar R"] ** 2)
        ok, how = _all_zero(curv, [inv["Kretschmann R_abcd R^abcd"] - rhs], pts)
        ck.add("Decomposition  K = C² + 4/(n−2)·Ric² − 2R²/((n−1)(n−2))", ok,
               how + " (four quantities from separate contractions)")


# ------------------------------------------------------------------------------------------ field equations

@dataclass
class FieldEquationResult:
    residual: sp.Matrix               # G_ab + Lambda g_ab - 8 pi T_ab
    satisfied: bool
    nonzero: list[tuple[int, int, sp.Expr]]
    numeric: bool
    lam: sp.Expr
    T: sp.Matrix


def parse_tensor_rows(metric: GRMetric, rows: list[list[str]]) -> sp.Matrix:
    n = metric.dim
    if len(rows) != n or any(len(r) != n for r in rows):
        raise GRInputError(f"The tensor must be a {n} x {n} grid.")
    return sp.Matrix(n, n, lambda i, j: metric.parse(str(rows[i][j])))


def verify_field_equations(curv: Curvature, T: sp.Matrix | None = None, cosmological_constant: sp.Expr | float = 0,
                           coupling: sp.Expr | float = 8 * sp.pi) -> FieldEquationResult:
    """Residual of G_ab + Lambda g_ab = coupling * T_ab (vacuum when T is None). `satisfied` is exact when the
    canonical form of every component is 0, numerical (see FieldEquationResult.numeric) otherwise."""
    n = curv.n
    T = T if T is not None else sp.zeros(n, n)
    lam = sp.sympify(cosmological_constant)
    res = (curv.einstein + lam * curv.g - sp.sympify(coupling) * T).applyfunc(simplify_expr)
    pts = [_jittered(curv.sample_subs, tuple(curv.sample_subs), k) for k in range(3)]
    nonzero: list[tuple[int, int, sp.Expr]] = []
    numeric = False
    for i in range(n):
        for j in range(n):
            ok, was_num = is_zero(res[i, j], pts)
            numeric |= was_num
            if not ok:
                nonzero.append((i, j, res[i, j]))
    return FieldEquationResult(res, not nonzero, nonzero, numeric, lam, T)


def effective_stress_energy(curv: Curvature, cosmological_constant: sp.Expr | float = 0) -> sp.Matrix:
    """The T_ab that would source this geometry: (G_ab + Lambda g_ab) / 8 pi."""
    return ((curv.einstein + sp.sympify(cosmological_constant) * curv.g) / (8 * sp.pi)).applyfunc(simplify_expr)


def electromagnetic_stress_energy(curv: Curvature, F_rows: list[list[str]]) -> sp.Matrix:
    """T_ab = (F_ac F_b^c - g_ab F_cd F^cd / 4) / 4 pi (Gaussian units) for an antisymmetric field strength F_ab
    given as a grid of expressions in the metric's coordinates and parameters."""
    if curv.metric is None:
        raise GRInputError("An electromagnetic source needs a metric with named coordinates.")
    F = parse_tensor_rows(curv.metric, F_rows)
    n = curv.n
    for i in range(n):
        for j in range(i, n):
            if sp.simplify(F[i, j] + F[j, i]) != 0:
                raise GRInputError("The field strength F_ab must be antisymmetric.")
    Fup = curv.ginv * F * curv.ginv                                          # F^{ab}
    f2 = sum(F[a, b] * Fup[a, b] for a in range(n) for b in range(n))
    T = sp.Matrix(n, n, lambda a, b: (sum(F[a, c] * F[b, d] * curv.ginv[c, d] for c in range(n) for d in range(n))
                                       - curv.g[a, b] * f2 / 4) / (4 * sp.pi))
    return T.applyfunc(simplify_expr)


@dataclass
class FluidDecomposition:
    density: sp.Expr
    pressures: list[sp.Expr]          # one per spatial direction (mixed components T^i_i)
    isotropic: bool
    note: str


def perfect_fluid_decomposition(curv: Curvature, cosmological_constant: sp.Expr | float = 0) -> FluidDecomposition:
    """Reads G_ab + Lambda g_ab = 8 pi T_ab as an effective fluid for a metric that is DIAGONAL with the time
    coordinate first: rho = -T^t_t and p_i = T^i_i. For a flat FRW metric this returns the Friedmann equations
    (rho = 3 H^2 / 8 pi, p = -(2 H' + 3 H^2)/8 pi). Refuses a non-diagonal metric rather than guess a frame."""
    n = curv.n
    if any(curv.g[i, j] != 0 for i in range(n) for j in range(n) if i != j):
        raise GRInputError("The fluid decomposition needs a diagonal metric (a comoving or static frame).")
    T = effective_stress_energy(curv, cosmological_constant)
    mixed = (curv.ginv * T).applyfunc(simplify_expr)
    rho = simplify_expr(-mixed[0, 0])
    pressures = [simplify_expr(mixed[i, i]) for i in range(1, n)]
    iso = all(simplify_expr(p - pressures[0]) == 0 for p in pressures[1:])
    return FluidDecomposition(rho, pressures, iso,
                              "isotropic: a perfect fluid" if iso else "anisotropic: not a perfect fluid in this frame")


# ------------------------------------------------------------------------------------------ singularities

@dataclass
class SingularityLocus:
    description: str
    conditions: dict[sp.Symbol, sp.Expr]
    kind: str                         # "curvature singularity" | "coordinate artefact" | "undetermined"
    evidence: str


def _factors(expr: sp.Expr, coords: tuple[sp.Symbol, ...]) -> list[sp.Expr]:
    out = []
    for fac, _ in sp.factor_list(expr)[1]:
        if fac.free_symbols & set(coords):
            out.append(fac)
    return out


def _nonneg_terms(factor: sp.Expr) -> list[sp.Expr] | None:
    """The terms of a sum that are each manifestly >= 0 (even powers times positive things); None if not so."""
    terms = []
    for t in sp.Add.make_args(sp.expand(factor)):
        coeff, rest = t.as_coeff_Mul()
        if coeff < 0:
            return None
        for f in sp.Mul.make_args(rest):
            base, ex = f.as_base_exp()
            if not (ex.is_Integer and ex % 2 == 0) and not f.is_positive:
                return None
        terms.append(t)
    return terms if len(terms) > 1 else None


def _real_loci(factor: sp.Expr, coords: tuple[sp.Symbol, ...]) -> list[dict[sp.Symbol, sp.Expr]]:
    """Real points where `factor` vanishes, as {coordinate: value}. Solved in plain real symbols (a positive
    radial coordinate would make SymPy say r = 0 has no solution), then restricted to each coordinate's range:
    >= 0 for a positive coordinate, [0, pi] for theta."""
    real = {c: sp.Symbol(f"_{c.name}", real=True) for c in coords}
    back = {v: k for k, v in real.items()}
    fac = factor.xreplace(real)
    present = [real[c] for c in coords if c in factor.free_symbols]

    def in_range(sym: sp.Symbol, val: sp.Expr) -> bool:
        c = back[sym]
        if val.free_symbols or val.is_real is False:
            return False
        v = float(sp.N(val))
        if c.is_positive:
            return v >= 0
        if c.name == "theta":
            return -1e-12 <= v <= math.pi + 1e-12
        return True

    terms = _nonneg_terms(fac)
    if terms:
        try:
            sols = sp.solve(terms, present, dict=True)
        except Exception:  # noqa: BLE001
            sols = []
        good = [s for s in sols if s and all(in_range(k, v) for k, v in s.items())]
        if good:
            return [{back[k]: v for k, v in s.items()} for s in good]
    out: list[dict[sp.Symbol, sp.Expr]] = []
    for c in present:
        try:
            roots = sp.solve(fac, c)
        except Exception:  # noqa: BLE001
            continue
        out += [{back[c]: rt} for rt in roots if in_range(c, rt)]
    return out


def analyze_singularities(curv: Curvature, params: dict[str, float] | None = None) -> list[SingularityLocus]:
    """Where do things go wrong, and is it the geometry or only the coordinates?

    Candidate loci are the zeros of every factor that sits in a denominator of the metric, its inverse, the
    Christoffel symbols or any curvature invariant (and in det g). Each is then classified by approaching it
    numerically: if some invariant grows without bound the point is a CURVATURE singularity (no change of
    coordinates can remove it); if every invariant stays finite while the metric components misbehave it is a
    COORDINATE artefact (Schwarzschild's r = 2M, the polar axis, ...). Parameters take the given values."""
    if curv.metric is None:
        raise GRInputError("Singularity analysis needs a metric with named coordinates.")
    metric = curv.metric
    coords = metric.coords
    psubs = {sp.Symbol(k, positive=True): sp.nsimplify(v, rational=True) for k, v in
             {**metric.params, **(params or {})}.items() if k in metric.params}
    n = curv.n

    metric_exprs = [curv.g[i, j] for i in range(n) for j in range(i, n)] + [curv.ginv[i, j] for i in range(n)
                                                                              for j in range(i, n)]
    metric_exprs += [curv.christoffel[k][i][j] for k in range(n) for i in range(n) for j in range(i, n)]
    sources: dict[str, tuple[sp.Expr, bool]] = {}
    for e in metric_exprs:
        e = sp.together(e.xreplace(psubs))
        for part in (sp.fraction(e)[1],):
            for f in _factors(part, coords):
                sources.setdefault(str(f), (f, False))
    det = sp.factor(sp.together(curv.g.det().xreplace(psubs)))
    for f in _factors(sp.fraction(det)[0], coords) + _factors(sp.fraction(det)[1], coords):
        sources.setdefault(str(f), (f, False))
    for inv in curv.invariants.values():
        for f in _factors(sp.fraction(sp.together(inv.xreplace(psubs)))[1], coords):
            sources[str(f)] = (f, True)

    invs = {k: sp.together(v.xreplace(psubs)) for k, v in curv.invariants.items()}
    base = dict(zip(coords, metric.sample_point()))
    found: list[SingularityLocus] = []
    seen: set[str] = set()
    for key in sorted(sources):
        factor, _from_inv = sources[key]
        for cond in _real_loci(factor, coords):
            desc = " and ".join(f"{k} = {sp.simplify(v)}" for k, v in sorted(cond.items(), key=lambda kv: str(kv[0])))
            if desc in seen:
                continue
            seen.add(desc)
            diverges, text = _approach(invs, cond, base, coords)
            if diverges is None:
                kind = "undetermined"
            else:
                kind = "curvature singularity" if diverges else "coordinate artefact"
            found.append(SingularityLocus(desc, cond, kind, text))
    order = {"curvature singularity": 0, "coordinate artefact": 1, "undetermined": 2}
    found.sort(key=lambda s: (order[s.kind], s.description))
    return found


def _approach(invs: dict[str, sp.Expr], cond: dict[sp.Symbol, sp.Expr], base: dict[sp.Symbol, float],
              coords: tuple[sp.Symbol, ...]) -> tuple[bool | None, str]:
    """Evaluate each invariant approaching the locus; (diverges?, human-readable evidence)."""
    import mpmath
    mpmath.mp.dps = 40
    eps_list = [mpmath.mpf(10) ** -k for k in (2, 3, 4, 5, 6)]
    lines = []
    diverges = False
    measured = False
    for name, expr in invs.items():
        if expr == 0 or not expr.free_symbols & set(coords):
            continue
        fn = sp.lambdify(coords, expr, modules="mpmath")
        vals = []
        for eps in eps_list:
            point = [mpmath.mpf(base[c]) for c in coords]
            for i, c in enumerate(coords):
                if c in cond:
                    point[i] = mpmath.mpf(sp.N(cond[c], 40)) + eps
            try:
                vals.append(abs(fn(*point)))
            except (ZeroDivisionError, ValueError, TypeError):
                vals = []
                break
        if len(vals) < len(eps_list):
            continue
        measured = True
        grows = vals[-1] > 1e3 * (vals[0] + mpmath.mpf("1e-12")) and vals[-1] > 1e4
        diverges |= bool(grows)
        lines.append(f"{name}: {mpmath.nstr(vals[0], 3)} at distance 1e-2 → {mpmath.nstr(vals[-1], 3)} at 1e-6"
                     + (" (diverges)" if grows else " (finite)"))
    if not measured:
        return False, "all invariants are constant or zero here: nothing diverges"
    return diverges, "; ".join(lines)
