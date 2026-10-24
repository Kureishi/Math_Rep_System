"""
Partition-function engine -- Streamlit-free.

Given an energy spectrum {E_n, g_n} (levels with degeneracies), at temperature T (k_B given, default 1):

    Z = sum g_n exp(-E_n / kT)        F = -kT ln Z        U = <E> = -d ln Z / d beta
    S = (U - F) / T = -k sum_states p ln p                C = dU/dT = (<E^2> - <E>^2) / (k T^2)

Computed in a numerically stable way (energies measured from the ground state, so nothing overflows at low T),
from the MOMENTS of the Boltzmann distribution rather than by differentiating a numerical ln Z: U and C come out
exact to rounding. An INFINITE spectrum (the harmonic oscillator, the rigid rotor, anything typed with no upper
n) is truncated where the terms at the highest requested temperature have fallen below 1e-15 of the sum, and if
they never do (E_n growing too slowly) the engine says the partition function diverges instead of returning a
number.

BUILT-IN CHECKS (`verify_thermo`), each independent of the way the numbers were produced:
  * the Gibbs entropy -k sum p ln p equals (U - F)/T;
  * U equals -d ln Z/d beta and C equals dU/dT by central FINITE DIFFERENCES of ln Z (the moment formulas and
    the derivative of the sum are different routes to the same quantities);
  * the third law: S -> k ln g_0 as T -> 0, and for a finite spectrum S -> k ln (total degeneracy) as T -> infinity;
  * C >= 0 and S >= 0 everywhere;
  * where a CLOSED FORM exists (two-level system, harmonic oscillator) the numerical sum is compared with it;
  * the rigid rotor is compared with its high-temperature (Mulholland) series.
The Schottky anomaly of the two-level system is also located numerically and compared with the root of its
transcendental equation  e^x = (x + 2)/(x - 2),  x = Delta/kT  (x ~ 2.3994).

IDEAL GAS: exact ln Z_N = N ln(V/lambda^3) - ln N! (the 1/N! is what makes the entropy extensive: the Gibbs paradox is
shown by checking extensivity with and without it), compared with the Sackur-Tetrode (Stirling) form, and the
single-particle translational sum over the levels of a box is compared with V/lambda^3.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import gammaln

from modules.gr_metrics import parse_in_symbols
from modules.quantum_common import CheckList
from modules.sm_common import StatMechError

N_CAP = 400_000
TOL_TAIL = 1e-15


@dataclass
class ClosedForm:
    """Closed-form Z, F, U, S/k, C/k as functions of kT (energy units), for comparison with the numerical sum."""
    latex: dict[str, str]
    funcs: dict[str, Callable[[np.ndarray], np.ndarray]]


@dataclass
class Spectrum:
    name: str
    energies: np.ndarray
    degeneracies: np.ndarray
    generator: Callable[[int], tuple[np.ndarray, np.ndarray]] | None = None     # n_levels -> (E, g), for infinite spectra
    closed_form: ClosedForm | None = None
    gap: float | None = None                                                    # lowest excitation energy, if known

    @property
    def infinite(self) -> bool:
        return self.generator is not None


@dataclass
class Thermo:
    kT: np.ndarray
    kB: float
    lnZ: np.ndarray
    F: np.ndarray
    U: np.ndarray
    S: np.ndarray                      # (U - F)/T
    S_gibbs: np.ndarray                # -k sum p ln p
    C: np.ndarray
    n_levels: int
    populations: np.ndarray | None = None     # (n_T, n_levels_shown) probabilities of the lowest levels
    energies_shown: np.ndarray | None = None
    ground: float = 0.0
    g0: float = 1.0


# ------------------------------------------------------------------------------------------ builders

def _sym_form(Z: sp.Expr, kT: sp.Symbol) -> ClosedForm:
    beta = sp.Symbol("beta", positive=True)
    lnZ = sp.log(Z)
    F = -kT * lnZ
    U = sp.simplify((-sp.diff(lnZ.subs(kT, 1 / beta), beta)).subs(beta, 1 / kT))
    S = sp.simplify((U - F) / kT)
    C = sp.simplify(sp.diff(U, kT))
    exprs = {"Z": Z, "F": F, "U": U, "S": S, "C": C}
    funcs = {k: sp.lambdify(kT, v, modules="numpy") for k, v in exprs.items()}
    return ClosedForm({k: sp.latex(v) for k, v in exprs.items()}, funcs)


def two_level(gap: float, g_ground: int = 1, g_excited: int = 1) -> Spectrum:
    if gap <= 0 or g_ground < 1 or g_excited < 1:
        raise StatMechError("The gap must be positive and the degeneracies at least 1.")
    kT = sp.Symbol("kT", positive=True)
    Z = g_ground + g_excited * sp.exp(-sp.Float(gap) / kT)
    return Spectrum(f"two-level system (gap {gap:g})", np.array([0.0, gap]), np.array([g_ground, g_excited], dtype=float),
                    closed_form=_sym_form(Z, kT), gap=gap)


def harmonic_oscillator(hw: float, zero_point: bool = True) -> Spectrum:
    if hw <= 0:
        raise StatMechError("The oscillator quantum must be positive.")
    shift = 0.5 * hw if zero_point else 0.0
    kT = sp.Symbol("kT", positive=True)
    x = sp.Float(hw) / kT
    Z = 1 / (2 * sp.sinh(x / 2)) if zero_point else 1 / (1 - sp.exp(-x))

    def gen(n: int) -> tuple[np.ndarray, np.ndarray]:
        return hw * np.arange(n) + shift, np.ones(n)
    e, g = gen(64)
    return Spectrum(f"harmonic oscillator (ħω = {hw:g})", e, g, generator=gen, closed_form=_sym_form(Z, kT), gap=hw)


def rigid_rotor(B: float) -> Spectrum:
    if B <= 0:
        raise StatMechError("The rotational constant must be positive.")

    def gen(n: int) -> tuple[np.ndarray, np.ndarray]:
        j = np.arange(n, dtype=float)
        return B * j * (j + 1), 2 * j + 1
    e, g = gen(64)
    return Spectrum(f"rigid rotor (B = {B:g})", e, g, generator=gen, gap=2 * B)


def explicit_levels(energies: list[float], degeneracies: list[float] | None = None, name: str = "custom levels") -> Spectrum:
    e = np.asarray(energies, dtype=float)
    g = np.ones_like(e) if degeneracies is None else np.asarray(degeneracies, dtype=float)
    if e.ndim != 1 or len(e) < 1 or len(e) != len(g):
        raise StatMechError("Give one degeneracy per energy level.")
    if not np.all(np.isfinite(e)) or np.any(g <= 0):
        raise StatMechError("Energies must be finite and degeneracies positive.")
    order = np.argsort(e)
    e, g = e[order], g[order]
    gaps = np.diff(np.unique(e))
    return Spectrum(name, e, g, gap=float(gaps[0]) if len(gaps) else None)


def from_expression(energy: str, degeneracy: str = "1", n_min: int = 0, n_max: int | None = None,
                    params: dict[str, float] | None = None) -> Spectrum:
    """E_n and g_n typed as expressions in the integer n (and parameters); n runs from n_min to n_max, or to infinity
    when n_max is None. Names other than n are parameters, with the values given (default 1)."""
    n_sym = sp.Symbol("n", integer=True)
    from modules.gr_metrics import _names_in
    names = sorted((_names_in(energy) | _names_in(degeneracy)) - {"n"})
    syms: dict[str, sp.Symbol] = {"n": n_sym}
    syms.update({p: sp.Symbol(p, positive=True) for p in names})
    vals = {p: float((params or {}).get(p, 1.0)) for p in names}
    try:
        e_expr, g_expr = parse_in_symbols(energy, syms), parse_in_symbols(degeneracy, syms)
    except ValueError as exc:
        raise StatMechError(str(exc)) from None
    e_expr, g_expr = e_expr.subs({syms[p]: v for p, v in vals.items()}), g_expr.subs({syms[p]: v for p, v in vals.items()})
    fe, fg = sp.lambdify(n_sym, e_expr, modules="numpy"), sp.lambdify(n_sym, g_expr, modules="numpy")

    def levels(count: int) -> tuple[np.ndarray, np.ndarray]:
        n = np.arange(n_min, n_min + count, dtype=float)
        with np.errstate(all="ignore"):
            e = np.broadcast_to(np.asarray(fe(n), dtype=float), n.shape).copy()
            g = np.broadcast_to(np.asarray(fg(n), dtype=float), n.shape).copy()
        if not (np.all(np.isfinite(e)) and np.all(np.isfinite(g))):
            raise StatMechError("E_n or g_n is not finite for some n in range (a division by zero or an overflow?).")
        if np.any(g <= 0):
            raise StatMechError("Degeneracies must be positive.")
        return e, g

    if n_max is not None:
        if n_max < n_min:
            raise StatMechError("n_max must be at least n_min.")
        if n_max - n_min + 1 > N_CAP:
            raise StatMechError(f"At most {N_CAP} levels.")
        e, g = levels(n_max - n_min + 1)
        return Spectrum(f"E_n = {energy}, g_n = {degeneracy}", e, g)
    e, g = levels(64)
    return Spectrum(f"E_n = {energy}, g_n = {degeneracy}", e, g, generator=levels)


def levels_for(spec: Spectrum, kT_max: float, tol: float = TOL_TAIL) -> tuple[np.ndarray, np.ndarray]:
    """The levels needed so that the neglected tail of Z at kT_max is below `tol` (relative). A finite spectrum is
    returned unchanged. Raises if an infinite spectrum's partition function does not converge."""
    if spec.generator is None:
        return spec.energies, spec.degeneracies
    n = 256
    while n <= N_CAP:
        e, g = spec.generator(n)
        e0 = float(e.min())
        w = g * np.exp(-(e - e0) / kT_max)
        total = float(w.sum())
        tail = float(w[int(0.9 * n):].sum())
        if total > 0 and tail <= tol * total and w[-1] < w[:-1].max():
            return e, g
        n *= 2
    raise StatMechError("The partition function does not converge: E_n grows too slowly for the sum over n to settle "
                        f"even with {N_CAP} levels at kT = {kT_max:g}.")


# ------------------------------------------------------------------------------------------ thermodynamics

def thermodynamics(spec: Spectrum, kT: np.ndarray | list[float], kB: float = 1.0, show_levels: int = 6) -> Thermo:
    """All the thermodynamic functions on a grid of kT (energy units)."""
    kT_arr = np.atleast_1d(np.asarray(kT, dtype=float))
    if np.any(kT_arr <= 0) or not np.all(np.isfinite(kT_arr)):
        raise StatMechError("Temperatures must be positive and finite.")
    if kB <= 0:
        raise StatMechError("k_B must be positive.")
    e, g = levels_for(spec, float(kT_arr.max()))
    e0 = float(e.min())
    eps = e - e0
    g0 = float(g[eps == 0].sum())
    n_t = len(kT_arr)
    lnZ, U, C, Sg = (np.empty(n_t) for _ in range(4))
    show = min(show_levels, len(e))
    order = np.argsort(e)[:show]
    pops = np.empty((n_t, show))
    for i, t in enumerate(kT_arr):
        w = g * np.exp(-eps / t)
        z = w.sum()
        p = w / z
        mean = float((p * eps).sum())
        var = float((p * eps ** 2).sum()) - mean ** 2
        lnZ[i] = -e0 / t + math.log(z)
        U[i] = e0 + mean
        C[i] = kB * max(var, 0.0) / t ** 2
        q = p / g                                   # probability of ONE state of each level
        ok = q > 0                                  # p ln p -> 0 as p -> 0 (and avoids log of an underflowed 0)
        Sg[i] = -kB * float((p[ok] * np.log(q[ok])).sum())
        pops[i] = p[order]
    F = -kT_arr * lnZ
    T = kT_arr / kB
    S = (U - F) / T
    return Thermo(kT_arr, kB, lnZ, F, U, S, Sg, C, len(e), pops, e[order], e0, g0)


def verify_thermo(spec: Spectrum, th: Thermo) -> CheckList:
    ck = CheckList()
    kB = th.kB
    scale_S = max(1.0, float(np.max(np.abs(th.S))))
    d = float(np.max(np.abs(th.S - th.S_gibbs)))
    ck.add("Gibbs entropy −k Σp ln p = (U − F)/T", d < 1e-9 * scale_S, f"max difference {d:.1e}")
    # finite differences of ln Z
    e, g = levels_for(spec, float(th.kT.max()) * 1.05)
    e0 = float(e.min())

    def lnz(t: float) -> float:
        return -e0 / t + math.log(float((g * np.exp(-(e - e0) / t)).sum()))
    ub, cb = 0.0, 0.0
    for t, u, c in zip(th.kT, th.U, th.C):
        h = 1e-4 * t
        beta_p, beta_m = 1 / (t * (1 - 1e-4)), 1 / (t * (1 + 1e-4))
        ub = max(ub, abs(-(lnz(1 / beta_p) - lnz(1 / beta_m)) / (beta_p - beta_m) - u) / (abs(u) + abs(e0) + t))
        u_p = _u_at(e, g, e0, t + h)
        u_m = _u_at(e, g, e0, t - h)
        cb = max(cb, abs(kB * (u_p - u_m) / (2 * h) - c) / (c + 1e-3))
    ck.add("U = −∂lnZ/∂β  (finite difference of ln Z)", ub < 1e-6, f"max relative difference {ub:.1e}")
    ck.add("C = dU/dT  (finite difference of U)", cb < 1e-4, f"max relative difference {cb:.1e}")
    ck.add("C ≥ 0 and S ≥ 0 everywhere", bool(np.all(th.C >= -1e-12) and np.all(th.S >= -1e-9)),
           f"min C = {th.C.min():.2e}, min S = {th.S.min():.2e}")
    if spec.gap is not None and float(th.kT.min()) <= spec.gap / 12:
        j = int(np.argmin(th.kT))
        want = kB * math.log(th.g0)
        err = abs(th.S[j] - want)
        ck.add("third law: S → k ln g₀ as T → 0", err < 2e-3 * max(1.0, abs(want)) + 20 * math.exp(-spec.gap / th.kT[j]),
               f"S = {th.S[j]:.6g} at kT = {th.kT[j]:g}, k ln g₀ = {want:.6g}")
    if not spec.infinite and len(spec.energies) > 1 and float(th.kT.max()) >= 200 * float(np.ptp(spec.energies)):
        j = int(np.argmax(th.kT))
        want = kB * math.log(float(spec.degeneracies.sum()))
        ck.add("S → k ln (total degeneracy) as T → ∞", abs(th.S[j] - want) < 1e-3 * want,
               f"S = {th.S[j]:.6g}, k ln Σg = {want:.6g}")
    if spec.closed_form is not None:
        cf = spec.closed_form
        worst = 0.0
        for key, arr, factor in (("F", th.F, 1.0), ("U", th.U, 1.0), ("S", th.S, 1 / kB), ("C", th.C, 1 / kB)):
            ref = np.asarray(cf.funcs[key](th.kT), dtype=float)
            worst = max(worst, float(np.max(np.abs(arr * factor - ref) / (np.abs(ref) + 1e-3))))
        ck.add("numerical sum = closed form (F, U, S, C)", worst < 1e-8, f"max relative difference {worst:.1e}")
    if spec.name.startswith("rigid rotor"):
        B = spec.energies[1] / 2
        hi = th.kT >= 40 * B
        if hi.any():
            x = B / th.kT[hi]
            ser = np.log(1 / x * (1 + x / 3 + x ** 2 / 15 + 4 * x ** 3 / 315))
            err = float(np.max(np.abs(th.lnZ[hi] - ser)))
            ck.add("rotor: ln Z = ln[(kT/B)(1 + B/3kT + (B/kT)²/15 + 4(B/kT)³/315)] at high T", err < 1e-7, f"max difference {err:.1e}")
    return ck


def _u_at(e: np.ndarray, g: np.ndarray, e0: float, t: float) -> float:
    w = g * np.exp(-(e - e0) / t)
    return e0 + float((w * (e - e0)).sum() / w.sum())


def schottky_peak(gap: float) -> tuple[float, float]:
    """(x_numeric, x_root): the position x = Delta/kT of the heat-capacity maximum of the two-level system, found
    (a) by maximising C numerically and (b) as the root of e^x = (x + 2)/(x - 2). They must agree."""
    x = np.linspace(0.5, 8, 200001)
    c = x ** 2 * np.exp(x) / (1 + np.exp(x)) ** 2
    x_num = float(x[int(np.argmax(c))])
    x_root = brentq(lambda y: math.exp(y) - (y + 2) / (y - 2), 2.1, 5.0)
    return x_num, x_root


def tail_report(spec: Spectrum, kT_max: float) -> str:
    e, _ = levels_for(spec, kT_max)
    return f"{len(e)} levels" + (" (truncated where the terms drop below 1e-15 of the sum)" if spec.infinite else "")


# ------------------------------------------------------------------------------------------ ideal gas

@dataclass
class GasThermo:
    N: float
    V: float
    kT: np.ndarray
    lnZ: np.ndarray
    F: np.ndarray
    U: np.ndarray
    S: np.ndarray
    S_sackur_tetrode: np.ndarray
    S_without_gibbs_factor: np.ndarray
    C: np.ndarray
    checks: CheckList = field(default_factory=CheckList)


def thermal_wavelength(kT: np.ndarray | float, m: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    return hbar * np.sqrt(2 * np.pi / (m * np.atleast_1d(np.asarray(kT, dtype=float))))


def _gas_lnZ(N: int, V: float, kT: float, m: float, hbar: float, gibbs: bool = True) -> float:
    lam = float(thermal_wavelength(kT, m, hbar)[0])
    return N * math.log(V / lam ** 3) - (float(gammaln(N + 1)) if gibbs else 0.0)


def box_translational_Z1(L: float, kT: float, m: float = 1.0, hbar: float = 1.0, n_max: int = 20000) -> float:
    """Single-particle translational partition function of a particle in a cubic box of side L, summed over the
    quantum levels E = hbar^2 pi^2 (nx^2 + ny^2 + nz^2) / (2 m L^2): the cube of the 1D sum."""
    n = np.arange(1, n_max + 1, dtype=float)
    one_d = float(np.exp(-hbar ** 2 * np.pi ** 2 * n ** 2 / (2 * m * L ** 2 * kT)).sum())
    return one_d ** 3


def ideal_gas(N: int, V: float, kT: np.ndarray | list[float], m: float = 1.0, hbar: float = 1.0) -> GasThermo:
    """Monatomic ideal gas from ln Z_N = N ln(V/lambda^3) - ln N!, with its checks (module docstring)."""
    if N < 1 or V <= 0 or m <= 0 or hbar <= 0:
        raise StatMechError("Need N >= 1, V > 0, m > 0 and hbar > 0.")
    t = np.atleast_1d(np.asarray(kT, dtype=float))
    if np.any(t <= 0):
        raise StatMechError("Temperatures must be positive.")
    lam = thermal_wavelength(t, m, hbar)
    lnZ = N * np.log(V / lam ** 3) - float(gammaln(N + 1))
    F = -t * lnZ
    U = 1.5 * N * t
    S = (U - F) / t
    S_st = N * (np.log(V / (N * lam ** 3)) + 2.5)
    S_no = N * (np.log(V / lam ** 3) + 1.5)
    out = GasThermo(float(N), V, t, lnZ, F, U, S, S_st, S_no, np.full_like(t, 1.5 * N))
    ck = out.checks
    # Stirling: S_exact - S_ST = -(1/2) ln(2 pi N) - 1/(12 N) + ...
    resid = S - S_st + 0.5 * math.log(2 * math.pi * N)
    ck.add("exact S differs from Sackur–Tetrode by exactly the Stirling remainder −½ln(2πN)", float(np.max(np.abs(resid))) < 1.0 / (10 * N) + 1e-9,
           f"S − S_ST + ½ln(2πN) = {float(np.max(np.abs(resid))):.1e} (≤ 1/12N = {1 / (12 * N):.1e})")
    mid = float(np.median(t))
    dh = 1e-4
    bp, bm = 1 / (mid * (1 - dh)), 1 / (mid * (1 + dh))
    u_fd = -(_gas_lnZ(N, V, 1 / bp, m, hbar) - _gas_lnZ(N, V, 1 / bm, m, hbar)) / (bp - bm)
    ck.add("U = −∂lnZ/∂β = (3/2) N kT  (equipartition)", abs(u_fd - 1.5 * N * mid) < 1e-6 * 1.5 * N * mid, f"finite difference {u_fd:.8g} vs {1.5 * N * mid:.8g}")
    S2 = (1.5 * 2 * N * mid - (-mid * _gas_lnZ(2 * N, 2 * V, mid, m, hbar))) / mid
    S2_no = (1.5 * 2 * N * mid - (-mid * _gas_lnZ(2 * N, 2 * V, mid, m, hbar, gibbs=False))) / mid
    S1 = float(S[int(np.argmin(np.abs(t - mid)))])
    S1_no = (1.5 * N * mid - (-mid * _gas_lnZ(N, V, mid, m, hbar, gibbs=False))) / mid
    ext_with, ext_without = abs(S2 - 2 * S1) / abs(S1), abs(S2_no - 2 * S1_no) / abs(S1_no)
    ck.add("Gibbs paradox: S(2N, 2V) = 2 S(N, V) only WITH the 1/N! factor", ext_with < ext_without and ext_with < max(0.5 * math.log(math.pi * N) / abs(S1), 1e-9) + 1e-6,
           f"deviation from extensivity {ext_with:.1e} with 1/N!, {ext_without:.1e} without")
    L = V ** (1 / 3)
    lam_mid = float(thermal_wavelength(mid, m, hbar)[0])
    if L / lam_mid >= 20:
        zb = box_translational_Z1(L, mid, m, hbar)
        zc = V / lam_mid ** 3
        tol = 3 * lam_mid / L
        ck.add("single-particle sum over the levels of a box → V/λ³ when λ ≪ L", abs(zb - zc) / zc < tol,
               f"sum = {zb:.6g}, V/λ³ = {zc:.6g}, relative difference {abs(zb - zc) / zc:.1e} (expected ≈ 3λ/2L = {1.5 * lam_mid / L:.1e})")
    return out
