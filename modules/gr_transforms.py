"""
Coordinate transformations of a metric, with the curvature invariants as the check -- Streamlit-free.

A change of coordinates x -> x' must change the COMPONENTS of every tensor and nothing else, so the scalar
invariants (Ricci scalar, R_ab R^ab, Kretschmann, Weyl squared) must come out identical. That is the test this
module applies to every transformation it performs, and it is a sharp one: a slip in a Jacobian, a missed
factor, or a transformation that is not invertible all change an invariant or fail to give a Lorentzian metric.

TWO WAYS TO SPECIFY A TRANSFORMATION
  EXPLICIT  "old coordinates as functions of the new" (x = f(x')):  g'_ab(x') = J^m_a J^n_b g_mn(f(x')),
            J = dx/dx'. Everything is then a function of the new coordinates and the result is an ordinary
            GRMetric you can feed to any other tool (geodesics, curvature, ...). Schwarzschild -> Eddington-
            Finkelstein, isotropic coordinates, Minkowski -> Rindler / spherical, de Sitter -> conformal time.
  IMPLICIT  "new coordinates as functions of the old" (x' = F(x)): the usual way to write Kruskal-Szekeres,
            whose inverse needs the Lambert W function. g'_ab = (J^-1)^m_a (J^-1)^n_b g_mn with J = dF/dx.
            The components are returned as functions of the OLD coordinates -- the same tensor, written in the
            new basis at the point labelled by the old coordinates -- and the curvature is computed through the
            CHAIN RULE (d'_a = (J^-1)^m_a d_m), so no inversion is ever needed. SymPy is first asked for the
            inverse (and the transformation becomes explicit if it finds a unique one).
In both modes the Ricci tensor is ALSO transformed by the tensor law (R'_ab = J^T R J) and compared with the Ricci
tensor the engine computes directly from g': two different routes to the same object.

SCOPE. Transformations are local and symbolic: domains (r > 2M for the Kruskal map, eta < 0 for conformal time)
are the caller's responsibility and are stated in each preset's note. Piecewise or multi-valued maps are not handled.
"""
from dataclasses import dataclass, field

import sympy as sp

from modules.gr_curvature import (invert_metric, Curvature, Partial, compute_curvature, curvature_from_components, evaluate, is_zero,
                                  numeric_subs, simplify_expr)
from modules.gr_metrics import (GR_TIMEOUT, GRInputError, GRMetric, coordinate_symbol, parameter_symbol, parse_in_symbols)
from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList
from modules.timeout_utils import run_with_timeout


@dataclass
class InvariantComparison:
    name: str
    old: sp.Expr
    new: sp.Expr
    equal: bool
    numeric: bool


@dataclass
class TransformResult:
    old: GRMetric
    new_names: list[str]
    mode: str                                   # "explicit" | "implicit"
    components: sp.Matrix                       # g'_ab (in new coordinates if explicit, old if implicit)
    new_metric: GRMetric | None                 # the transformed metric as a first-class metric (explicit only)
    jacobian: sp.Matrix
    curvature_old: Curvature
    curvature_new: Curvature
    comparisons: list[InvariantComparison] = field(default_factory=list)
    checks: CheckList = field(default_factory=CheckList)
    inverted: bool = False                      # an implicit transformation that SymPy could invert

    @property
    def invariants_preserved(self) -> bool:
        return bool(self.comparisons) and all(c.equal for c in self.comparisons)


def _parse_exprs(exprs: list[str], symbols: dict[str, sp.Symbol], n: int, what: str) -> list[sp.Expr]:
    if len(exprs) != n:
        raise GRInputError(f"Give {n} expressions for {what} (one per coordinate).")
    return [parse_in_symbols(str(e), symbols) for e in exprs]


def _regular_sample(g: sp.Matrix, coords: tuple[sp.Symbol, ...], params: dict[sp.Symbol, float],
                    candidates: list[list[float]]) -> list[float]:
    """The first candidate point where g is finite, invertible and Lorentzian; else the first candidate."""
    import numpy as np
    for cand in candidates:
        subs = {**dict(zip(coords, cand)), **params}
        try:
            mat = np.array([[evaluate(g[i, j], subs) for j in range(g.shape[1])] for i in range(g.shape[0])])
            if np.isfinite(mat).all() and (np.linalg.eigvalsh(mat) < 0).sum() == 1 and abs(np.linalg.det(mat)) > 1e-12:
                return cand
        except (ValueError, np.linalg.LinAlgError):
            continue
    return candidates[0]


def transform_metric(old: GRMetric, new_names: list[str], *, old_in_terms_of_new: list[str] | None = None,
                     new_in_terms_of_old: list[str] | None = None, old_curvature: Curvature | None = None,
                     params: dict[str, float] | None = None, try_invert: bool = True,
                     progress: ProgressFn | None = None, timeout: float = GR_TIMEOUT) -> TransformResult:
    """Transform `old` to coordinates called `new_names`; give exactly one of the two specifications (module
    docstring). The result carries the new components, both curvature computations and the comparison."""
    n = old.dim
    new_names = [s.strip() for s in new_names]
    if len(new_names) != n or len(set(new_names)) != n or not all(s.isidentifier() for s in new_names):
        raise GRInputError(f"Give {n} distinct names for the new coordinates.")
    if (old_in_terms_of_new is None) == (new_in_terms_of_old is None):
        raise GRInputError("Specify either the old coordinates in terms of the new, or the new in terms of the old.")
    pvals = old.param_subs(params)
    report(progress, "curvature of the original metric", 0.0)
    cold = old_curvature or compute_curvature(old, progress=None, timeout=timeout, params=params)
    mode = "explicit" if old_in_terms_of_new is not None else "implicit"
    inverted = False

    if mode == "implicit" and try_invert:
        inv = _try_invert(old, new_names, new_in_terms_of_old or [])
        if inv is not None:
            old_in_terms_of_new, new_in_terms_of_old, mode, inverted = inv, None, "explicit", True

    if mode == "explicit":
        result = _explicit(old, new_names, old_in_terms_of_new or [], cold, pvals, progress, timeout)
    else:
        result = _implicit(old, new_names, new_in_terms_of_old or [], cold, pvals, progress, timeout)
    result.inverted = inverted
    _compare(result, pvals)
    return result


def _try_invert(old: GRMetric, new_names: list[str], exprs: list[str]) -> list[str] | None:
    """old coordinates in terms of new, if SymPy finds a unique closed form quickly (as strings), else None."""
    n = old.dim
    new_syms = [sp.Symbol(f"_n{i}", real=True) for i in range(n)]
    try:
        F = [old.parse(e) for e in exprs]
        eqs = [F[i] - new_syms[i] for i in range(n)]
        sols = run_with_timeout(lambda: sp.solve(eqs, list(old.coords), dict=True), timeout=4.0, label="inverting the map")
    except Exception:  # noqa: BLE001
        return None
    if len(sols) != 1 or set(sols[0]) != set(old.coords):
        return None
    back = {new_syms[i]: sp.Symbol(new_names[i]) for i in range(n)}
    return [str(sols[0][c].xreplace(back)).replace("**", "^") for c in old.coords]


def _explicit(old: GRMetric, new_names: list[str], exprs: list[str], cold: Curvature, pvals, progress, timeout) -> TransformResult:
    n = old.dim
    new_coords = tuple(coordinate_symbol(nm) for nm in new_names)
    # an old coordinate keeps its own symbol if the new one has the same name (r -> r), so the substitution is consistent
    new_coords = tuple(next((c for c in old.coords if str(c) == nm), nc) for nm, nc in zip(new_names, new_coords))
    symbols: dict[str, sp.Symbol] = {str(c): c for c in new_coords}
    symbols.update(old.param_symbols)
    f = _parse_exprs(exprs, symbols, n, "the old coordinates")
    J = sp.Matrix(f).jacobian(list(new_coords))                          # dx/dx'
    det = simplify_expr(J.det())
    if det == 0:
        raise GRInputError("That transformation is not invertible (its Jacobian determinant is zero).")
    sub = dict(zip(old.coords, f))
    g_old = old.g.xreplace(sub)
    gnew = (J.T * g_old * J).applyfunc(simplify_expr)
    candidates = [[float(old.sample.get(nm, d)) for nm, d in zip(new_names, (0.5, 3.1, 1.1, 0.9, 1.3))],
                  [0.7, 3.1, 1.1, 0.9, 1.3][:n], [0.4, 2.2, 0.8, 0.6, 1.1][:n], [0.3, 5.0, 1.3, 0.5, 0.9][:n]]
    sample = _regular_sample(gnew, new_coords, pvals, candidates)
    new_metric = GRMetric("transformed", f"{old.name} in ({', '.join(new_names)})", new_coords, gnew, dict(old.params),
                          f"{old.name} after a change of coordinates.", embedding="none", family="",
                          sample=dict(zip(new_names, sample)))
    report(progress, "curvature of the transformed metric", 0.5)
    cnew = compute_curvature(new_metric, progress=None, timeout=timeout)
    res = TransformResult(old, new_names, "explicit", gnew, new_metric, J, cold, cnew)
    res.checks.add("Jacobian determinant is nonzero", True, f"det dx/dx' = {det}")
    res._map = (sub, new_coords, J)                                       # type: ignore[attr-defined]
    return res


def _implicit(old: GRMetric, new_names: list[str], exprs: list[str], cold: Curvature, pvals, progress, timeout) -> TransformResult:
    n = old.dim
    F = _parse_exprs(exprs, old.symbols_by_name(), n, "the new coordinates")
    J = sp.Matrix(F).jacobian(list(old.coords))                           # dx'/dx
    det = simplify_expr(J.det())
    if det == 0:
        raise GRInputError("That transformation is not invertible (its Jacobian determinant is zero).")
    Jinv = invert_metric(J)
    gnew = (Jinv.T * old.g * Jinv).applyfunc(simplify_expr)
    partial: Partial = lambda e, i: sum(Jinv[m, i] * sp.diff(e, old.coords[m]) for m in range(n))  # noqa: E731
    subs = numeric_subs(old)
    subs.update(pvals)
    report(progress, "curvature through the chain rule", 0.5)
    cnew = curvature_from_components(gnew, n, partial, old.coords, subs, metric=None, progress=None, timeout=timeout)
    res = TransformResult(old, new_names, "implicit", gnew, None, J, cold, cnew)
    res.checks.add("Jacobian determinant is nonzero", True, f"det dx'/dx = {det}")
    res._map = (None, old.coords, Jinv)                                   # type: ignore[attr-defined]
    return res


def _sample_points(res: TransformResult, pvals) -> list[dict[sp.Symbol, float]]:
    base = res.curvature_new.sample_subs
    out = []
    for k in range(3):
        pt = dict(base)
        for s in list(pt):
            if s in pvals:
                continue
            pt[s] = base[s] * (1 + 0.07 * k)
        out.append(pt)
    return out


def _compare(res: TransformResult, pvals) -> None:
    sub, coords, J = res._map                                              # type: ignore[attr-defined]
    pts = _sample_points(res, pvals)
    old_inv, new_inv = res.curvature_old.invariants, res.curvature_new.invariants
    for name, old_expr in old_inv.items():
        if name not in new_inv:
            continue
        o = old_expr.xreplace(sub) if sub is not None else old_expr
        diff = simplify_expr(o - new_inv[name])
        ok, numeric = is_zero(diff, pts)
        res.comparisons.append(InvariantComparison(name, o, new_inv[name], ok, numeric))
        res.checks.add(f"invariant unchanged: {name}", ok,
                       "identical after the change of coordinates" + (" (settled at sample points)" if numeric and ok else "")
                       if ok else f"differs: {diff}")
    # tensor law for Ricci
    n = res.old.dim
    ric_old = res.curvature_old.ricci.xreplace(sub) if sub is not None else res.curvature_old.ricci
    law = (J.T * ric_old * J) if sub is not None else (J.inv().T * ric_old * J.inv())
    direct = res.curvature_new.ricci
    ok_all, numeric_any = True, False
    for i in range(n):
        for j in range(i, n):
            ok, numeric = is_zero(simplify_expr(law[i, j] - direct[i, j]), pts)
            ok_all &= ok
            numeric_any |= numeric
    res.checks.add("Ricci tensor transforms as a tensor (R' = Jᵀ R J) and matches the engine's direct R'", ok_all,
                   "two independent routes agree" + (" (partly at sample points)" if numeric_any else ""))
    res.checks.add("curvature identities hold for the transformed metric", res.curvature_new.checks.passed,
                   res.curvature_new.checks.summary())


# ------------------------------------------------------------------------------------------ presets

@dataclass(frozen=True)
class TransformPreset:
    key: str
    label: str
    source: str                                 # a gr_metrics preset key
    new_names: tuple[str, ...]
    exprs: tuple[str, ...]
    direction: str                              # "old_in_terms_of_new" | "new_in_terms_of_old"
    note: str
    try_invert: bool = True                     # False where SymPy is known to fail (it would only burn its time limit)


TRANSFORM_PRESETS: dict[str, TransformPreset] = {p.key: p for p in [
    TransformPreset("schw_ingoing_ef", "Schwarzschild → ingoing Eddington–Finkelstein", "schwarzschild",
                    ("v", "r", "theta", "phi"), ("v - r - 2*M*log(r - 2*M)", "r", "theta", "phi"), "old_in_terms_of_new",
                    "v = t + r*, with r* = r + 2M ln(r/2M − 1). Valid for r > 2M; the new metric is regular across the future horizon."),
    TransformPreset("schw_outgoing_ef", "Schwarzschild → outgoing Eddington–Finkelstein", "schwarzschild",
                    ("u", "r", "theta", "phi"), ("u + r + 2*M*log(r - 2*M)", "r", "theta", "phi"), "old_in_terms_of_new",
                    "u = t − r*. Regular across the past horizon."),
    TransformPreset("schw_isotropic", "Schwarzschild → isotropic coordinates", "schwarzschild",
                    ("t", "rho", "theta", "phi"), ("t", "rho*(1 + M/(2*rho))^2", "theta", "phi"), "old_in_terms_of_new",
                    "r = ρ (1 + M/2ρ)²: the spatial metric becomes conformally flat. Covers the exterior, ρ > M/2."),
    TransformPreset("schw_kruskal", "Schwarzschild → Kruskal–Szekeres", "schwarzschild",
                    ("T", "X", "theta", "phi"),
                    ("sqrt(r/(2*M) - 1)*exp(r/(4*M))*sinh(t/(4*M))", "sqrt(r/(2*M) - 1)*exp(r/(4*M))*cosh(t/(4*M))",
                     "theta", "phi"), "new_in_terms_of_old",
                    "T and X from (t, r) in region I (r > 2M). The inverse needs the Lambert W function, so this runs in "
                    "implicit mode (curvature through the chain rule). The (T, X) block comes out conformally flat.",
                    try_invert=False),
    TransformPreset("mink_rindler", "Minkowski → Rindler (accelerated observers)", "minkowski",
                    ("eta", "xi", "y", "z"), ("xi*sinh(eta)", "xi*cosh(eta)", "y", "z"), "old_in_terms_of_new",
                    "t = ξ sinh η, x = ξ cosh η (ξ > 0). Flat space seen by uniformly accelerated observers: curvature stays exactly 0."),
    TransformPreset("mink_spherical", "Minkowski → spherical coordinates", "minkowski",
                    ("t", "r", "theta", "phi"),
                    ("t", "r*sin(theta)*cos(phi)", "r*sin(theta)*sin(phi)", "r*cos(theta)"), "old_in_terms_of_new",
                    "The textbook polar-coordinate metric of flat space."),
    TransformPreset("desitter_conformal", "de Sitter (flat slicing) → conformal time", "de_sitter",
                    ("eta", "r", "theta", "phi"), ("-log(-H*eta)/H", "r", "theta", "phi"), "old_in_terms_of_new",
                    "η = −e^(−Ht)/H, so η < 0. The metric becomes (1/(Hη)²)(−dη² + dr² + r² dΩ²): conformally flat."),
]}


def apply_preset(key: str, params: dict[str, float] | None = None, progress: ProgressFn | None = None,
                 timeout: float = GR_TIMEOUT) -> TransformResult:
    from modules.gr_metrics import preset as metric_preset
    if key not in TRANSFORM_PRESETS:
        raise GRInputError(f"Unknown transformation '{key}'.")
    p = TRANSFORM_PRESETS[key]
    kwargs = {p.direction: list(p.exprs)}
    return transform_metric(metric_preset(p.source), list(p.new_names), params=params, progress=progress, timeout=timeout,
                            try_invert=p.try_invert, **kwargs)  # type: ignore[arg-type]
