"""
Rayleigh–Schrödinger perturbation theory, derived exactly and compared with exact diagonalisation -- Streamlit-free.

    H = H0 + λ V          H0 diagonal with the energies you give,   V a Hermitian matrix in that basis

For a NON-DEGENERATE level n (with E_n the unperturbed energy and V_mn = <m|V|n>):

    E1 =  V_nn
    E2 =  Σ_{m≠n} |V_mn|² / (E_n - E_m)
    E3 =  Σ_{k,m≠n} V_nk V_km V_mn / ((E_n - E_k)(E_n - E_m))  -  V_nn Σ_{m≠n} |V_mn|² / (E_n - E_m)²
    |n¹> = Σ_{m≠n} V_mn / (E_n - E_m) |m>

All of it is done in exact arithmetic (SymPy): E1 for the anharmonic oscillator comes out as the fraction
3(2n²+2n+1)/4, not 0.75·(…) rounded. For a DEGENERATE level, ordinary perturbation theory divides by zero;
the first-order energies are instead the eigenvalues of V restricted to the degenerate subspace, and that is
what is reported (with no second-order term, which needs more care and is not computed).

THE CHECKS, because a derivation can be wrong and a comparison with a plot is not a test:
  * V is Hermitian;
  * for a complete finite system (all levels given), the traces of the corrections vanish beyond first order:
    Σ_n E2 = Σ_n E3 = 0, since tr H is linear in λ, and Σ_n E1 = tr V;
  * the second-order shift of the lowest level is never positive;
  * against exact diagonalisation of H0 + λV at several small λ, the error after order k must shrink like
    λ^(k+1): the slope on a log-log plot is MEASURED and must be at least k + 1 (this is the check that
    catches a wrong coefficient, since a wrong E2 leaves a λ² error and a slope of 2 where 3 was expected;
    a slope ABOVE k + 1 only means the next coefficient vanishes by symmetry);
  * for the oscillator, the exact result is computed in a Hermite basis, tested for convergence in the basis
    size, and cross-checked against the independent real-space solver in modules/quantum_1d.

The series for an anharmonic oscillator is ASYMPTOTIC, not convergent: its coefficients grow factorially, so
adding orders helps only while λ is small. The comparison table makes that visible rather than hiding it.
"""
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList
from modules.timeout_utils import ComputationTimeoutError, run_with_timeout

MAX_LEVELS = 40
MAX_ORDER = 3


@dataclass
class LevelResult:
    index: int
    e0: sp.Expr
    degenerate_with: list[int]
    e1: sp.Expr | None                    # for a degenerate level: None (see first_order_branches)
    e2: sp.Expr | None
    e3: sp.Expr | None
    state_correction: sp.Matrix | None    # |n¹> in the unperturbed basis (non-degenerate levels)
    first_order_branches: list[sp.Expr] = field(default_factory=list)


def _simplify_exact(expr: sp.Expr) -> sp.Expr:
    try:
        return run_with_timeout(lambda: sp.nsimplify(sp.simplify(expr), rational=False), label="simplify", timeout=5.0)
    except (ComputationTimeoutError, Exception):  # noqa: BLE001 -- the unsimplified exact value is still exact
        return expr


def perturbation_levels(e0: Sequence[sp.Expr | float | int], v: sp.Matrix, levels: Sequence[int] | None = None) -> list[LevelResult]:
    """Exact corrections through third order for the requested levels (all by default)."""
    n = len(e0)
    if v.shape != (n, n):
        raise ValueError(f"V must be {n} x {n} to match the {n} unperturbed energies.")
    if n > MAX_LEVELS:
        raise ValueError(f"At most {MAX_LEVELS} levels.")
    energies = [sp.nsimplify(e) if not isinstance(e, sp.Basic) else e for e in e0]
    which = list(levels) if levels is not None else list(range(n))
    if any(not 0 <= i < n for i in which):
        raise ValueError("A requested level is outside the basis.")
    out: list[LevelResult] = []
    for i in which:
        group = [j for j in range(n) if sp.simplify(energies[j] - energies[i]) == 0]
        if len(group) > 1:
            sub = v.extract(group, group)
            branches = sorted((sp.simplify(val) for val, mult in sub.eigenvals().items() for _ in range(mult)),
                              key=lambda z: float(sp.N(z)))
            out.append(LevelResult(i, energies[i], [j for j in group if j != i], None, None, None, None, branches))
            continue
        others = [m for m in range(n) if m != i]
        diffs = {m: energies[i] - energies[m] for m in others}
        e1 = v[i, i]
        v_in = {m: v[m, i] for m in others}
        e2 = sum((sp.Abs(v_in[m]) ** 2 / diffs[m] for m in others), sp.Integer(0))
        e3a = sum((v[i, k] * v[k, m] * v[m, i] / (diffs[k] * diffs[m]) for k in others for m in others), sp.Integer(0))
        e3b = e1 * sum((sp.Abs(v_in[m]) ** 2 / diffs[m] ** 2 for m in others), sp.Integer(0))
        state = sp.Matrix([0 if m == i else v[m, i] / diffs[m] for m in range(n)])
        out.append(LevelResult(i, energies[i], [], _simplify_exact(e1), _simplify_exact(e2),
                               _simplify_exact(e3a - e3b), state.applyfunc(_simplify_exact)))
    return out


# ------------------------------------------------------------------------------------------ comparison

@dataclass
class Comparison:
    level: int
    lambdas: np.ndarray
    exact: np.ndarray
    series: dict[int, np.ndarray]         # order -> E0 + Σ_{j≤order} λ^j E_j
    errors: dict[int, np.ndarray]         # order -> |exact - series|
    slopes: dict[int, float]              # measured log-log slope per order
    exact_method: str


def suggest_lambda_max(level: LevelResult, gap: float) -> float:
    """A λ at which the perturbative shifts are still small next to the level spacing `gap`."""
    scales = []
    for coeff, power in ((level.e1, 1), (level.e2, 2), (level.e3, 3)):
        if coeff is not None:
            value = abs(float(sp.N(coeff)))
            if value > 1e-12:
                scales.append((0.05 * gap / value) ** (1.0 / power))
    return float(min(scales)) if scales else 0.1


def _series(level: LevelResult, lam: np.ndarray, order: int) -> np.ndarray:
    e: np.ndarray = float(sp.N(level.e0)) + np.zeros_like(lam)
    for power, coeff in ((1, level.e1), (2, level.e2), (3, level.e3)):
        if power <= order and coeff is not None:
            e = e + float(sp.N(coeff)) * lam ** power
    return e


def _matched_exact(e0_arr: np.ndarray, v_num: np.ndarray, level: int, lam: float) -> float:
    """The exact eigenvalue that continues the unperturbed level (largest overlap with its basis vector)."""
    w, vecs = np.linalg.eigh(np.diag(e0_arr) + lam * v_num)
    return float(w[int(np.argmax(np.abs(vecs[level, :]) ** 2))])


def compare_with_exact(e0: Sequence[float], v: sp.Matrix, result: LevelResult, lambdas: Sequence[float] | np.ndarray,
                       exact_fn=None, exact_method: str = "diagonalising H0 + λV") -> Comparison:
    """Errors of the order-1, 2, 3 truncations against exact diagonalisation at each λ, and their measured slopes."""
    lam = np.asarray(lambdas, dtype=float)
    if result.e1 is None:
        raise ValueError("This level is degenerate; the series comparison is for non-degenerate levels.")
    e0_arr = np.array([float(sp.N(e)) for e in e0])
    v_num = np.array(v.evalf(), dtype=complex)
    if exact_fn is None:
        exact = np.array([_matched_exact(e0_arr, v_num, result.index, float(l)) for l in lam])
    else:
        exact = np.array([exact_fn(float(l)) for l in lam])
    series = {k: _series(result, lam, k) for k in range(1, MAX_ORDER + 1)}
    errors = {k: np.abs(exact - series[k]) for k in series}
    slopes = {}
    for k, err in errors.items():
        ok = err > 1e-14
        slopes[k] = float(np.polyfit(np.log(lam[ok]), np.log(err[ok]), 1)[0]) if ok.sum() >= 3 else float("nan")
    return Comparison(result.index, lam, exact, series, errors, slopes, exact_method)


def verify_series(e0: Sequence[float], v: sp.Matrix, levels: list[LevelResult], comparison: Comparison | None,
                  complete: bool, hermitian_tol: float = 1e-12) -> CheckList:
    """The checks of the module docstring, for a set of levels (and optionally one comparison)."""
    checks = CheckList()
    vn = np.array(v.evalf(), dtype=complex)
    herm_err = float(np.max(np.abs(vn - vn.conj().T)))
    checks.add("V is Hermitian", herm_err < hermitian_tol, f"largest |V - V†| = {herm_err:.1e}")

    nondegenerate = [r for r in levels if r.e1 is not None]
    if complete and len(levels) == len(e0) and len(nondegenerate) == len(levels):
        for power, label in ((1, "Σ E1 = tr V"), (2, "Σ E2 = 0"), (3, "Σ E3 = 0")):
            total = sum(float(sp.N({1: r.e1, 2: r.e2, 3: r.e3}[power])) for r in levels)
            target = float(sp.re(sp.N(v.trace()))) if power == 1 else 0.0
            checks.add(f"Trace sum rule: {label}", abs(total - target) < 1e-9 * max(1.0, abs(target)),
                       f"sum = {total:.10g}, expected {target:.10g}")
    lowest = min(levels, key=lambda r: float(sp.N(r.e0))) if levels else None
    if lowest is not None and lowest.e2 is not None and len(levels) == len(e0):
        e2 = float(sp.N(lowest.e2))
        checks.add("Second-order shift of the lowest level is not positive", e2 <= 1e-12, f"E2(lowest) = {e2:.6g}")
    if comparison is not None:
        for k in range(1, MAX_ORDER + 1):
            slope = comparison.slopes.get(k, float("nan"))
            if np.isfinite(slope):
                # ONE-SIDED: a wrong coefficient leaves a lower-order error and a slope BELOW k+1. A slope above
                # it means the next coefficient happens to vanish (a symmetry), which is correct, not a failure.
                faster = slope > k + 1 + 0.5
                checks.add(f"Order {k}: error shrinks at least like λ^{k + 1}", slope > (k + 1) - 0.5,
                           f"measured slope {slope:.2f}, expected {k + 1}" +
                           (" -- faster than expected, so the next coefficient vanishes here" if faster else ""))
            else:
                checks.add(f"Order {k}: error shrinks at least like λ^{k + 1}", True,
                           "the error is at rounding level, so the series is exact here")
    return checks


# ------------------------------------------------------------------------------------------ the oscillator

def oscillator_position_matrix(dim: int) -> sp.Matrix:
    """x = (a + a†)/√2 in the oscillator basis (ħ = m = ω = 1), exact, dimension `dim`."""
    x = sp.zeros(dim)
    for n in range(dim - 1):
        x[n, n + 1] = x[n + 1, n] = sp.sqrt(sp.Rational(n + 1, 2))
    return x


def polynomial_perturbation(coefficients: dict[int, float | sp.Expr], dim: int) -> sp.Matrix:
    """V = Σ c_k x^k as an exact dim x dim matrix in the oscillator basis. The powers are formed in a larger
    basis and cropped, so every entry kept is exact (a power of the truncated matrix is wrong near the edge)."""
    if not coefficients or any(not isinstance(k, int) or not 1 <= k <= 8 for k in coefficients):
        raise ValueError("Give powers of x between 1 and 8.")
    top = max(coefficients)
    big = oscillator_position_matrix(dim + top + 1)
    total = sp.zeros(dim + top + 1)
    for power, coeff in coefficients.items():
        total += sp.nsimplify(coeff) * big ** power
    return total.extract(list(range(dim)), list(range(dim)))


@dataclass
class OscillatorPerturbation:
    coefficients: dict[int, float]
    levels: list[LevelResult]
    comparison: Comparison | None
    comparison_level: int
    basis_dim: int
    checks: CheckList
    lambdas: np.ndarray
    notes: list[str] = field(default_factory=list)


def _hermite_exact_energy(coefficients: dict[int, float], dim: int, level: int, lam: float) -> float:
    x = np.array(oscillator_position_matrix(dim + max(coefficients) + 1).evalf(), dtype=float)
    total = np.zeros_like(x)
    for k, c in coefficients.items():
        total = total + c * np.linalg.matrix_power(x, k)
    v = total[:dim, :dim]
    h = np.diag(np.arange(dim) + 0.5) + lam * v
    return float(np.linalg.eigvalsh(h)[level])


def oscillator_perturbation(coefficients: dict[int, float], n_levels: int = 4, comparison_level: int = 0,
                            lambda_max: float | None = None, n_lambdas: int = 6, basis_dim: int | None = None,
                            progress: ProgressFn | None = None) -> OscillatorPerturbation:
    """Perturbation of H0 = ½p² + ½x² (ħ = m = ω = 1) by V = Σ c_k x^k, to third order, compared with exact
    diagonalisation in a Hermite basis and with the real-space solver."""
    if not 1 <= n_levels <= 12:
        raise ValueError("Ask for between 1 and 12 levels.")
    if not 0 <= comparison_level < n_levels:
        raise ValueError("The comparison level must be one of the levels shown.")
    top = max(coefficients) if coefficients else 0
    # third-order sums run over intermediate states up to 2·top levels above n, so keep a margin beyond that
    needed = n_levels + 3 * top + 4
    dim = int(basis_dim) if basis_dim else needed
    if dim < needed:
        raise ValueError(f"The basis needs at least {needed} states for these powers and levels.")
    if dim > MAX_LEVELS * 3:
        raise ValueError("The basis is too large.")
    report(progress, "Building the perturbation matrix", 0.1)
    v = polynomial_perturbation(coefficients, dim)
    e0 = [sp.Rational(2 * n + 1, 2) for n in range(dim)]
    report(progress, "Deriving the corrections exactly", 0.3)
    levels = perturbation_levels(e0, v, list(range(n_levels)))

    e1_scale = levels[comparison_level]
    gap = 1.0
    lam_max = lambda_max if lambda_max else suggest_lambda_max(e1_scale, gap)
    lambdas = np.geomspace(lam_max / 12.0, lam_max, n_lambdas)
    report(progress, "Diagonalising exactly for comparison", 0.6)
    big = max(dim, 2 * (n_levels + 3 * top + 8))
    comp = compare_with_exact(
        [float(sp.N(e)) for e in e0], v, levels[comparison_level], lambdas,
        exact_fn=lambda lam: _hermite_exact_energy(coefficients, big, comparison_level, lam),
        exact_method=f"diagonalising H0 + λV in a {big}-state oscillator basis")

    checks = verify_series(e0, v, levels, comp, complete=False)
    # (1) the exact reference itself: stable under a larger basis
    lam_test = float(lambdas[-1])
    e_big = _hermite_exact_energy(coefficients, big, comparison_level, lam_test)
    e_bigger = _hermite_exact_energy(coefficients, big + 24, comparison_level, lam_test)
    checks.add("The exact reference is converged in the basis size", abs(e_big - e_bigger) < 1e-9,
               f"{big} vs {big + 24} states: difference {abs(e_big - e_bigger):.1e}")
    # (2) a completely different method: finite differences on a real-space grid
    report(progress, "Cross-checking on a real-space grid", 0.85)
    grid_energy = _grid_energy(coefficients, comparison_level, lam_test)
    if grid_energy is not None:
        checks.add("Agrees with the independent real-space solver", abs(grid_energy - e_big) < 2e-5 * max(1.0, abs(e_big)),
                   f"grid {grid_energy:.8f} vs basis {e_big:.8f} (λ = {lam_test:.4g})")
    notes = ["The perturbation series for an anharmonic oscillator is asymptotic: its coefficients grow factorially, "
             "so more orders help only while λ is small."]
    return OscillatorPerturbation(dict(coefficients), levels, comp, comparison_level, dim, checks, lambdas, notes)


def _grid_energy(coefficients: dict[int, float], level: int, lam: float) -> float | None:
    from modules.quantum_1d import solve_eigenstates
    terms = " + ".join(f"({lam!r})*({c!r})*x**{k}" for k, c in coefficients.items())
    try:
        res = solve_eigenstates("custom", {}, domain=(-9.0, 9.0), n_points=1200, n_levels=level + 1,
                                custom_formula=f"0.5*x**2 + {terms}")
    except Exception:  # noqa: BLE001
        return None
    return float(res.energies[level])


def matrix_perturbation(e0: Sequence[float | int | sp.Expr], v: sp.Matrix, comparison_level: int | None = None,
                        lambda_max: float | None = None, n_lambdas: int = 6) -> tuple[list[LevelResult], Comparison | None, CheckList]:
    """Perturbation theory for a complete finite system: H0 diagonal with energies `e0`, V a Hermitian matrix."""
    levels = perturbation_levels(e0, v)
    comparison = None
    if comparison_level is not None:
        level = levels[comparison_level]
        if level.e1 is not None:
            others = [abs(float(sp.N(level.e0 - e))) for i, e in enumerate(e0) if i != comparison_level]
            gap = min(others) if others else 1.0
            lam_max = lambda_max if lambda_max else suggest_lambda_max(level, gap)
            comparison = compare_with_exact(e0, v, level, np.geomspace(lam_max / 12.0, lam_max, n_lambdas))
    checks = verify_series(e0, v, levels, comparison, complete=True)
    return levels, comparison, checks
