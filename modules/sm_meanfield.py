"""
Mean-field theory against exact solutions for the Ising model -- Streamlit-free.   H = -J sum s_i s_j - h sum s_i, k_B = 1.

THREE APPROXIMATIONS AND TWO EXACT SOLUTIONS, all for the same model:
  * MEAN FIELD (Bragg-Williams / Curie-Weiss): each spin sees the average of its z neighbours, so
    m = tanh((z J m + h)/T),  T_c = z J,  u = -(z J/2) m^2 - h m;  also the Landau free energy f(m).
  * BETHE-PEIERLS (cavity): exact on a tree, it keeps the correlation between a spin and ONE neighbour:
    tanh(J/T_c) = 1/(z-1). For z = 4: T_c = 2.885 J, between mean field (4 J) and the truth (2.269 J).
  * 1D EXACT (transfer matrix): no transition at any T > 0; m(h) = sinh(h/T)/sqrt(sinh^2(h/T) + e^(-4J/T)).
    The Bethe approximation with z = 2 is EXACT here (a chain is a tree), which the tests use as a check of both;
    mean field with z = 2 predicts T_c = 2 J, which is simply false.
  * 2D EXACT (Onsager): T_c = 2J/ln(1 + sqrt 2), the energy per site with the complete elliptic integral K, the
    spontaneous magnetisation (1 - sinh^-4(2J/T))^(1/8), and the free energy as a one-dimensional quadrature. The
    energy is obtained twice -- from the closed form and as d(beta f)/d(beta) of the free energy -- and they must agree.

WHERE MEAN FIELD FAILS (computed here, not asserted): T_c (4J vs 2.269J in 2D; 2J vs 0 in 1D); the order-parameter
exponent (fitted from the mean-field solution: beta = 1/2, against the exact 1/8); the specific heat (a finite JUMP of
3/2 against a logarithmic divergence, c ~ -(2/pi)(ln(1+sqrt 2))^2 ln|t| = -0.4945 ln|t|); and the energy above T_c,
where mean field says u = 0 (no short-range order at all) while the exact u(T_c) = -sqrt(2) J. Mean field becomes
correct above the upper critical dimension d = 4 (Ginzburg criterion), which the table of T_c against dimension
shows converging; the d = 3 value 4.5115 J is a high-precision numerical literature value and is labelled so.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ellipk

from modules.quantum_common import CheckList
from modules.sm_common import StatMechError

TC_3D_LITERATURE = 4.5115


# ------------------------------------------------------------------------------------------ mean field

def mf_tc(z: int = 4, J: float = 1.0) -> float:
    return z * J


def mf_magnetisation(T: float, h: float = 0.0, z: int = 4, J: float = 1.0) -> float:
    """The stable root of m = tanh((z J m + h)/T) (positive for h >= 0)."""
    if T <= 0:
        raise StatMechError("T must be positive.")
    if h == 0 and T >= z * J:
        return 0.0
    f = lambda m: m - math.tanh((z * J * m + h) / T)  # noqa: E731
    return float(brentq(f, 1e-14 if h == 0 else 0.0, 1.0, xtol=1e-15))


def mf_energy(T: float, h: float = 0.0, z: int = 4, J: float = 1.0) -> float:
    m = mf_magnetisation(T, h, z, J)
    return -0.5 * z * J * m * m - h * m


def mf_landau_free_energy(m: np.ndarray, T: float, h: float = 0.0, z: int = 4, J: float = 1.0) -> np.ndarray:
    """f(m) = -(zJ/2) m^2 - T [ln 2 - ((1+m)ln(1+m) + (1-m)ln(1-m))/2] - h m: its minimum is the self-consistent m."""
    m = np.asarray(m, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = np.log(2) - 0.5 * (np.where(1 + m > 0, (1 + m) * np.log(1 + m), 0) + np.where(1 - m > 0, (1 - m) * np.log(1 - m), 0))
    return -0.5 * z * J * m ** 2 - T * ent - h * m


def mf_susceptibility(T: float, z: int = 4, J: float = 1.0) -> float:
    """dm/dh at h = 0: (1 - m^2)/(T - zJ(1 - m^2)), which is 1/(T - T_c) above T_c."""
    m = mf_magnetisation(T, 0.0, z, J)
    return (1 - m * m) / (T - z * J * (1 - m * m))


# ------------------------------------------------------------------------------------------ Bethe

def bethe_tc(z: int = 4, J: float = 1.0) -> float:
    if z < 2:
        raise StatMechError("The coordination number must be at least 2.")
    if z == 2:
        return 0.0
    return J / math.atanh(1.0 / (z - 1))


def bethe_magnetisation(T: float, h: float = 0.0, z: int = 4, J: float = 1.0) -> float:
    """Magnetisation from the Bethe-Peierls cavity equations: the field a neighbour sends,
    phi(x) = atanh(tanh(J/T) tanh(x/T)) T, satisfies x = h + (z-1) phi(x); then m = tanh((h + z phi(x))/T)."""
    if T <= 0:
        raise StatMechError("T must be positive.")
    t = math.tanh(J / T)
    phi = lambda x: T * math.atanh(t * math.tanh(x / T))  # noqa: E731
    x = max(abs(h), 1.0) * 5.0 + 5 * z * J                                   # start in the ordered branch
    for _ in range(20000):
        nx = h + (z - 1) * phi(x)
        if abs(nx - x) < 1e-14:
            x = nx
            break
        x = nx
    return float(math.tanh((h + z * phi(x)) / T))


# ------------------------------------------------------------------------------------------ 1D exact

def ising_1d_magnetisation(T: float, h: float, J: float = 1.0) -> float:
    s = math.sinh(h / T)
    return float(s / math.sqrt(s * s + math.exp(-4 * J / T)))


def ising_1d_free_energy(T: float, h: float = 0.0, J: float = 1.0) -> float:
    """f per site = -T ln(lambda_+), lambda_+ = e^(J/T) cosh(h/T) + sqrt(e^(2J/T) sinh^2(h/T) + e^(-2J/T))."""
    b = 1.0 / T
    lam = math.exp(b * J) * math.cosh(b * h) + math.sqrt(math.exp(2 * b * J) * math.sinh(b * h) ** 2 + math.exp(-2 * b * J))
    return -T * math.log(lam)


def ising_1d_finite_lnZ(N: int, T: float, h: float = 0.0, J: float = 1.0) -> float:
    """Exact ln Z of a periodic chain of N spins: ln(lambda_+^N + lambda_-^N)."""
    b = 1.0 / T
    root = math.sqrt(math.exp(2 * b * J) * math.sinh(b * h) ** 2 + math.exp(-2 * b * J))
    lp, lm = math.exp(b * J) * math.cosh(b * h) + root, math.exp(b * J) * math.cosh(b * h) - root
    return N * math.log(lp) + math.log1p((lm / lp) ** N)


def ising_1d_brute_force_lnZ(N: int, T: float, h: float = 0.0, J: float = 1.0) -> float:
    if N > 20:
        raise StatMechError("Brute force is limited to N <= 20.")
    bits = ((np.arange(2 ** N)[:, None] >> np.arange(N)) & 1) * 2 - 1
    E = -J * (bits * np.roll(bits, -1, axis=1)).sum(axis=1) - h * bits.sum(axis=1)
    return float(-E.min() / T + math.log(np.exp(-(E - E.min()) / T).sum()))


def ising_1d_energy(T: float, J: float = 1.0) -> float:
    return -J * math.tanh(J / T)


def ising_1d_specific_heat(T: float, J: float = 1.0) -> float:
    return (J / T) ** 2 / math.cosh(J / T) ** 2


# ------------------------------------------------------------------------------------------ 2D exact (Onsager)

def onsager_tc(J: float = 1.0) -> float:
    return 2 * J / math.log(1 + math.sqrt(2))


def onsager_energy(T: float, J: float = 1.0) -> float:
    """u = -J coth(2K) [1 + (2/pi)(2 tanh^2(2K) - 1) K(kappa)],  K = J/T,  kappa = 2 sinh(2K)/cosh^2(2K)."""
    if T <= 0:
        raise StatMechError("T must be positive.")
    k2 = 2 * J / T
    th, kappa = math.tanh(k2), 2 * math.sinh(k2) / math.cosh(k2) ** 2
    if abs(kappa - 1) < 1e-13:
        return -math.sqrt(2) * J                                      # the limit: (2 tanh^2 - 1) -> 0 as K(kappa) -> infinity
    return -J / th * (1 + 2 / math.pi * (2 * th * th - 1) * float(ellipk(kappa ** 2)))


def onsager_magnetisation(T: float, J: float = 1.0) -> float:
    if T >= onsager_tc(J):
        return 0.0
    return float((1 - math.sinh(2 * J / T) ** -4) ** 0.125)


def onsager_free_energy(T: float, J: float = 1.0) -> float:
    """f per site = -T [ln 2 + (1/8 pi^2) integral integral ln(cosh^2 2K - sinh 2K (cos t1 + cos t2)) dt1 dt2], with the
    inner integral done exactly: int_0^{2 pi} ln(A - B cos t) dt = 2 pi ln((A + sqrt(A^2 - B^2))/2)."""
    k2 = 2 * J / T
    ch2, sh = math.cosh(k2) ** 2, math.sinh(k2)

    def inner(t1: float) -> float:
        a = ch2 - sh * math.cos(t1)
        return 2 * math.pi * math.log((a + math.sqrt(max(a * a - sh * sh, 0.0))) / 2)
    val, _ = quad(inner, 0, 2 * math.pi, limit=400, epsabs=1e-13, epsrel=1e-13, points=[math.pi] if abs(1 - sh) < 0.2 else None)
    return -T * (math.log(2) + val / (8 * math.pi ** 2))


def onsager_energy_from_free_energy(T: float, J: float = 1.0) -> float:
    """u = d(beta f)/d(beta), by a central difference of the free-energy quadrature."""
    d = 1e-5 / T
    b = 1.0 / T
    g = lambda beta: beta * onsager_free_energy(1 / beta, J)  # noqa: E731
    return (g(b + d) - g(b - d)) / (2 * d)


def onsager_specific_heat(T: float, J: float = 1.0) -> float:
    d = 1e-5 * T
    return (onsager_energy(T + d, J) - onsager_energy(T - d, J)) / (2 * d)


# ------------------------------------------------------------------------------------------ comparison

@dataclass
class ComparisonCurves:
    T: np.ndarray
    m_mf: np.ndarray
    m_bethe: np.ndarray
    m_exact_2d: np.ndarray
    u_mf: np.ndarray
    u_exact_2d: np.ndarray
    u_exact_1d: np.ndarray
    c_exact_2d: np.ndarray
    c_mf: np.ndarray


def compare_curves(T: np.ndarray, z: int = 4, J: float = 1.0) -> ComparisonCurves:
    """Mean field, Bethe and the exact 2D (z = 4) and 1D solutions on a temperature grid. The 'exact 2D' curves are for the
    square lattice whatever z is (z only changes the approximations)."""
    T = np.asarray(T, dtype=float)
    mf = np.array([mf_magnetisation(t, 0, z, J) for t in T])
    be = np.array([bethe_magnetisation(t, 0, z, J) for t in T])
    ex = np.array([onsager_magnetisation(t, J) for t in T])
    u_mf = -0.5 * z * J * mf ** 2
    u2 = np.array([onsager_energy(t, J) for t in T])
    u1 = np.array([ising_1d_energy(t, J) for t in T])
    c2 = np.array([onsager_specific_heat(t, J) for t in T])
    dT = 1e-6
    c_mf = np.array([(mf_energy(t + dT, 0, z, J) - mf_energy(max(t - dT, 1e-9), 0, z, J)) / (t + dT - max(t - dT, 1e-9)) for t in T])
    return ComparisonCurves(T, mf, be, ex, u_mf, u2, u1, c2, c_mf)


def fit_mf_exponent(z: int = 4, J: float = 1.0) -> float:
    """beta from the mean-field solution itself: slope of ln m against ln(1 - T/T_c) for small reduced temperature."""
    tc = mf_tc(z, J)
    t = np.geomspace(1e-6, 1e-3, 12)
    m = np.array([mf_magnetisation(tc * (1 - x), 0, z, J) for x in t])
    return float(np.polyfit(np.log(t), np.log(m), 1)[0])


def mf_specific_heat_jump(z: int = 4, J: float = 1.0) -> float:
    tc = mf_tc(z, J)
    e = 1e-7
    lo = (mf_energy(tc * (1 - e), 0, z, J) - mf_energy(tc * (1 - 2 * e), 0, z, J)) / (tc * e)
    return float(lo)


def onsager_log_coefficient() -> float:
    """The exact coefficient of the logarithmic divergence c ~ -A ln|t|: A = (2/pi) (ln(1 + sqrt 2))^2 = 0.4945."""
    return 2 / math.pi * math.log(1 + math.sqrt(2)) ** 2


def fit_onsager_log_coefficient() -> float:
    """-d c / d ln|t| measured on the exact specific heat just above T_c (should be A = 0.4945)."""
    tc = onsager_tc()
    t = np.array([1e-3, 1e-4])
    c = np.array([onsager_specific_heat(tc * (1 + x)) for x in t])
    return float(-(c[1] - c[0]) / (math.log(t[1]) - math.log(t[0])))


@dataclass
class FailureSummary:
    rows: list[tuple[str, str, str, str]]
    checks: CheckList = field(default_factory=CheckList)


def failure_summary(z: int = 4, J: float = 1.0) -> FailureSummary:
    """The quantitative list of where mean field fails (module docstring), with the cross-checks that validate the exact side."""
    tc_ex = onsager_tc(J)
    beta_fit = fit_mf_exponent(z, J)
    jump = mf_specific_heat_jump(z, J)
    a_meas, a_exact = fit_onsager_log_coefficient(), onsager_log_coefficient()
    rows = [
        ("T_c (square lattice, z = 4)", f"mean field {mf_tc(z, J):.4f} J", f"Bethe {bethe_tc(z, J):.4f} J", f"exact {tc_ex:.4f} J"),
        ("T_c (chain, z = 2)", f"mean field {mf_tc(2, J):.4f} J", f"Bethe {bethe_tc(2, J):.4f} J", "exact 0 (no transition)"),
        ("T_c (cubic lattice, z = 6)", f"mean field {mf_tc(6, J):.4f} J", f"Bethe {bethe_tc(6, J):.4f} J", f"numerical literature {TC_3D_LITERATURE} J"),
        ("order-parameter exponent β", f"{beta_fit:.4f} (fitted)", "—", "1/8 (exact, 2D)"),
        ("specific heat at T_c", f"jump of {jump:.4f}", "—", f"logarithmic divergence, −{a_exact:.4f} ln|t|"),
        ("energy u(T_c)", "0 above T_c: no short-range order", "—", f"{onsager_energy(tc_ex * (1 + 1e-9), J):.4f} (= −√2 J)"),
    ]
    out = FailureSummary(rows)
    ck = out.checks
    ck.add("mean-field β fitted from its own solution = 1/2", abs(beta_fit - 0.5) < 5e-3, f"fitted {beta_fit:.5f}")
    ck.add("mean-field specific-heat jump = 3/2", abs(jump - 1.5) < 2e-3 * 1.5 or z != 4 and abs(jump - 1.5) < 0.05, f"measured {jump:.5f}")
    ck.add("exact 2D specific heat diverges like −0.4945 ln|t|", abs(a_meas - a_exact) < 0.03, f"fitted {a_meas:.4f}, exact {a_exact:.4f}")
    f_u, c_u = onsager_energy_from_free_energy(0.9 * tc_ex, J), onsager_energy(0.9 * tc_ex, J)
    ck.add("Onsager energy: closed form = d(βf)/dβ of the free-energy integral", abs(f_u - c_u) < 1e-8, f"{c_u:.10f} vs {f_u:.10f}")
    ck.add("Onsager energy is continuous at T_c: u(T_c) = −√2 J", abs(onsager_energy(tc_ex * (1 - 1e-6), J) + math.sqrt(2) * J) < 2e-4
           and abs(onsager_energy(tc_ex * (1 + 1e-6), J) + math.sqrt(2) * J) < 2e-4, "both sides of T_c")
    t_b = 3.0
    ck.add("Bethe with z = 2 reproduces the exact 1D magnetisation", abs(bethe_magnetisation(t_b, 0.4, 2, J) - ising_1d_magnetisation(t_b, 0.4, J)) < 1e-9,
           f"{bethe_magnetisation(t_b, 0.4, 2, J):.10f} vs {ising_1d_magnetisation(t_b, 0.4, J):.10f}")
    ck.add("1D transfer matrix = brute-force sum over all 2^12 states", abs(ising_1d_finite_lnZ(12, 1.3, 0.2, J) - ising_1d_brute_force_lnZ(12, 1.3, 0.2, J)) < 1e-10,
           "ln Z of a periodic 12-spin chain")
    m = np.linspace(-0.99, 0.99, 19801)
    f = mf_landau_free_energy(m, 2.5, 0.0, z, J)
    ck.add("Landau free energy is minimised at the self-consistent m", abs(abs(m[int(np.argmin(f))]) - mf_magnetisation(2.5, 0, z, J)) < 2e-4,
           f"argmin |m| = {abs(m[int(np.argmin(f))]):.5f}, root {mf_magnetisation(2.5, 0, z, J):.5f}")
    return out
