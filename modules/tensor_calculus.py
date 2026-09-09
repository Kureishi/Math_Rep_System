"""
Classical (index-based) tensor calculus on a Riemannian manifold given
by a metric tensor g_ij(coords) in some coordinate system: Christoffel
symbols, the Riemann curvature tensor, Ricci tensor and scalar
curvature, covariant derivatives, and index raising/lowering. Standalone
symbolic tool, same category as dimensional_analysis.py / transforms.py:
the person supplies a metric directly, there's no "problem" for an LLM
to extract.

Deliberately scoped to METRIC-based tensor calculus rather than pulling
in sympy.diffgeom's Manifold/Patch/CoordSystem machinery: for the actual
use case here (given a metric, compute curvature; given a vector field,
compute its covariant derivative) the raw index formulas operating
directly on a sympy Matrix are more transparent, easier to verify term
by term, and easier to unit-test than round-tripping through
diffgeom's more general but heavier object model. This does mean
higher-rank general tensor fields beyond vectors (rank-2 tensor fields,
general contractions) aren't supported here -- that would be the
natural next extension if needed.

Verification-first, and unusually strong here because most of what
this module computes has an EXACT known answer to check against, not
just a numerical approximation:
- Metric compatibility (the covariant derivative of the metric itself
  must be EXACTLY zero, symbolically, for ANY metric -- this is a
  mathematical identity, not something that depends on which metric
  was given) is checked on every call, and would itself indicate a bug
  in the Christoffel-symbol computation if it ever failed.
- Flat space (Euclidean, in any coordinate system) must have EXACTLY
  zero Riemann tensor -- checked directly for the identity/Cartesian
  metric case and available as a general sanity check via
  is_flat_metric().
- The sign convention for the Riemann tensor was pinned down empirically
  against the sphere's known Gaussian curvature (K = 1/radius**2, so
  Ricci scalar = 2/radius**2 in 2D) during development -- there are
  multiple equally "standard" sign conventions in different textbooks
  (MTW vs. Wald vs. Weinberg differ), and getting this wrong silently
  produces a curvature of the right magnitude but the WRONG SIGN, which
  is exactly the kind of error that looks plausible without a known-
  answer check. See test_tensor_calculus.py's sphere/flat-space tests.
"""
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class TensorResult:
    coords: list[sp.Symbol]
    metric: sp.Matrix | None = None
    metric_inverse: sp.Matrix | None = None
    christoffel: list | None = None          # Christoffel[k][i][j] = Gamma^k_ij
    riemann: list | None = None              # Riemann[l][i][j][k] = R^l_ijk
    ricci_tensor: sp.Matrix | None = None
    ricci_scalar: sp.Expr | None = None
    metric_compatible: bool = False          # nabla_k g_ij == 0 exactly, for every k,i,j
    is_flat: bool | None = None              # None if not checked / inconclusive
    error: str | None = None


def _parse_metric(metric_rows: list[list[str]], coord_names: list[str]) -> tuple[sp.Matrix, list[sp.Symbol]]:
    coords = [sp.Symbol(c) for c in coord_names]
    local_dict = dict(zip(coord_names, coords))
    n = len(metric_rows)
    entries = []
    for row in metric_rows:
        entries.append([parse_expr(cell, local_dict=local_dict, transformations=_TRANSFORMS) for cell in row])
    return sp.Matrix(entries), coords


def _is_exactly_zero(expr: sp.Expr) -> bool:
    """Checks whether expr is identically zero, trying progressively
    harder simplification strategies before giving up -- plain
    sp.simplify() alone was found to miss some trig identities that
    genuinely ARE zero (e.g. a metric-compatibility residual on the
    sphere metric that numerically evaluates to ~1e-16 but sp.simplify()
    leaves as an unreduced trig expression). Rewriting through the
    exponential form first canonicalizes these cases reliably."""
    if expr == 0:
        return True
    try:
        if sp.simplify(expr) == 0:
            return True
    except Exception:  # noqa: BLE001
        pass
    try:
        return sp.simplify(expr.rewrite(sp.exp)) == 0
    except Exception:  # noqa: BLE001
        return False


def _christoffel_symbols(g: sp.Matrix, ginv: sp.Matrix, coords: list[sp.Symbol]) -> list:
    """Gamma^k_ij = (1/2) g^kl (d_i g_lj + d_j g_li - d_l g_ij), the
    Christoffel symbols of the second kind (the ones that appear in the
    geodesic equation and the covariant derivative)."""
    n = len(coords)
    Gamma = [[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)]
    for k in range(n):
        for i in range(n):
            for j in range(i, n):  # symmetric in i, j -- compute once, mirror
                s = sp.Integer(0)
                for l in range(n):
                    s += ginv[k, l] * (sp.diff(g[l, j], coords[i]) + sp.diff(g[l, i], coords[j])
                                        - sp.diff(g[i, j], coords[l]))
                val = sp.simplify(s / 2)
                Gamma[k][i][j] = val
                Gamma[k][j][i] = val
    return Gamma


def _riemann_tensor(Gamma: list, coords: list[sp.Symbol]) -> list:
    """R^l_ijk = d_j Gamma^l_ik - d_i Gamma^l_jk + Gamma^l_jm Gamma^m_ik
    - Gamma^l_im Gamma^m_jk. Sign convention pinned against the sphere's
    known positive Gaussian curvature -- see module docstring."""
    n = len(coords)
    Riem = [[[[sp.Integer(0)] * n for _ in range(n)] for _ in range(n)] for _ in range(n)]
    for l in range(n):
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    term = sp.diff(Gamma[l][i][k], coords[j]) - sp.diff(Gamma[l][j][k], coords[i])
                    for m in range(n):
                        term += Gamma[l][j][m] * Gamma[m][i][k] - Gamma[l][i][m] * Gamma[m][j][k]
                    Riem[l][i][j][k] = sp.simplify(term)
    return Riem


def _ricci_tensor(Riem: list, n: int) -> sp.Matrix:
    """R_jk = R^i_jik (contract the upper index with the second lower index)."""
    Ricci = sp.zeros(n, n)
    for j in range(n):
        for k in range(n):
            Ricci[j, k] = sp.simplify(sum(Riem[i][j][i][k] for i in range(n)))
    return Ricci


def _check_metric_compatibility(g: sp.Matrix, Gamma: list, coords: list[sp.Symbol]) -> bool:
    """nabla_k g_ij = d_k g_ij - Gamma^l_ki g_lj - Gamma^l_kj g_il must be
    EXACTLY zero for every i, j, k -- this is a mathematical identity
    that holds for ANY metric when Gamma was derived correctly FROM that
    metric (it's literally how the Christoffel symbols are defined), so
    this check is really a self-consistency check on the Christoffel
    computation above, not a property of the specific metric given."""
    n = len(coords)
    for k in range(n):
        for i in range(n):
            for j in range(n):
                nabla = sp.diff(g[i, j], coords[k])
                for l in range(n):
                    nabla -= Gamma[l][k][i] * g[l, j] + Gamma[l][k][j] * g[i, l]
                if not _is_exactly_zero(nabla):
                    return False
    return True


def analyze_metric(metric_rows: list[list[str]], coord_names: list[str]) -> TensorResult:
    """Given a metric tensor as a list of rows of expression strings
    (e.g. [["R**2", "0"], ["0", "R**2*sin(theta)**2"]] for a sphere of
    radius R in (theta, phi) coordinates) computes the Christoffel
    symbols, Riemann tensor, Ricci tensor, and Ricci scalar, checking
    metric compatibility along the way."""
    try:
        g, coords = _parse_metric(metric_rows, coord_names)
    except Exception as exc:  # noqa: BLE001
        return TensorResult(coords=[], error=f"Could not parse metric: {exc}")

    if g.shape[0] != g.shape[1] or g.shape[0] != len(coords):
        return TensorResult(coords=coords, error="Metric must be a square matrix matching the "
                                                    "number of coordinates given.")

    try:
        ginv = run_with_timeout(g.inv, label="metric_inverse")
    except ComputationTimeoutError as exc:
        return TensorResult(coords=coords, metric=g, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TensorResult(coords=coords, metric=g,
                             error=f"Metric could not be inverted (is it degenerate?): {exc}")

    try:
        Gamma = run_with_timeout(_christoffel_symbols, g, ginv, coords, label="christoffel")
        Riem = run_with_timeout(_riemann_tensor, Gamma, coords, label="riemann")
    except ComputationTimeoutError as exc:
        return TensorResult(coords=coords, metric=g, metric_inverse=ginv, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TensorResult(coords=coords, metric=g, metric_inverse=ginv,
                             error=f"Could not compute curvature: {exc}")

    n = len(coords)
    Ricci = _ricci_tensor(Riem, n)
    Rscalar = sp.simplify(sum(ginv[i, j] * Ricci[i, j] for i in range(n) for j in range(n)))
    compatible = _check_metric_compatibility(g, Gamma, coords)
    is_flat = all(_is_exactly_zero(Riem[l][i][j][k]) for l in range(n) for i in range(n)
                  for j in range(n) for k in range(n))

    return TensorResult(coords=coords, metric=g, metric_inverse=ginv, christoffel=Gamma,
                         riemann=Riem, ricci_tensor=Ricci, ricci_scalar=Rscalar,
                         metric_compatible=compatible, is_flat=is_flat)


def nonzero_christoffel_symbols(result: TensorResult) -> list[tuple[int, int, int, sp.Expr]]:
    """Returns (k, i, j, value) for every Gamma^k_ij that isn't
    identically zero, deduplicated for the i<->j symmetry (Gamma^k_ij ==
    Gamma^k_ji always), for a compact display instead of an n**3 grid
    mostly full of zeros."""
    if result.christoffel is None:
        return []
    n = len(result.coords)
    out = []
    for k in range(n):
        for i in range(n):
            for j in range(i, n):
                val = result.christoffel[k][i][j]
                if val != 0:
                    out.append((k, i, j, val))
    return out


def covariant_derivative_of_vector(result: TensorResult, vector_components: list[str]) -> sp.Matrix | None:
    """nabla_j V^i = d_j V^i + Gamma^i_jk V^k, for a contravariant vector
    field V^i given as a list of expression strings (one per
    coordinate). Returns an n x n matrix indexed [j][i] (derivative
    direction, then component), or None if the metric analysis that
    produced `result` failed."""
    if result.christoffel is None:
        return None
    coords = result.coords
    n = len(coords)
    local_dict = {str(c): c for c in coords}
    V = [parse_expr(v, local_dict=local_dict, transformations=_TRANSFORMS) for v in vector_components]

    nabla = sp.zeros(n, n)
    for j in range(n):
        for i in range(n):
            val = sp.diff(V[i], coords[j])
            for k in range(n):
                val += result.christoffel[i][j][k] * V[k]
            nabla[j, i] = sp.simplify(val)
    return nabla


def lower_index(result: TensorResult, vector_components: list[str]) -> sp.Matrix | None:
    """V_i = g_ij V^j -- converts a contravariant vector to covariant form."""
    if result.metric is None:
        return None
    coords = result.coords
    local_dict = {str(c): c for c in coords}
    V = sp.Matrix([parse_expr(v, local_dict=local_dict, transformations=_TRANSFORMS)
                   for v in vector_components])
    return sp.simplify(result.metric * V)


def raise_index(result: TensorResult, covector_components: list[str]) -> sp.Matrix | None:
    """V^i = g^ij V_j -- converts a covariant vector (covector) to contravariant form."""
    if result.metric_inverse is None:
        return None
    coords = result.coords
    local_dict = {str(c): c for c in coords}
    V = sp.Matrix([parse_expr(v, local_dict=local_dict, transformations=_TRANSFORMS)
                   for v in covector_components])
    return sp.simplify(result.metric_inverse * V)
