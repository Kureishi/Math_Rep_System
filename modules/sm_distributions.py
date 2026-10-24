"""
Distribution tools: Maxwell-Boltzmann, Fermi-Dirac, Bose-Einstein -- Streamlit-free.   Units hbar = m = k_B = 1.

    occupation     f_MB = e^(-(e - mu)/T)     f_FD = 1/(e^((e - mu)/T) + 1)     f_BE = 1/(e^((e - mu)/T) - 1)
    free ideal gas in 3D, g spin states, thermal wavelength lambda = sqrt(2 pi / (m T)), fugacity z = e^(mu/T):
        n = (g / lambda^3) F_3/2(z)     P = (g T / lambda^3) F_5/2(z)     u = (3/2) P
        F_s = z (MB),  -Li_s(-z) (FD),  Li_s(z) (BE)        (Li = polylogarithm)

CHEMICAL POTENTIAL SOLVER: given n and T, find mu with n(mu, T) = n (Brent's method on a bracket that is grown until
it straddles the root). For bosons n(mu <= 0) is bounded by g zeta(3/2)/lambda^3: above that density the gas cannot hold
the particles in excited states, mu sticks at 0 and the rest is the BOSE-EINSTEIN CONDENSATE,
N_0/N = 1 - (T/T_c)^(3/2) with T_c = (2 pi / m) (n / (g zeta(3/2)))^(2/3).

THE TOOLS CHECK THEMSELVES against independent routes (`verify_gas_tools`):
  * the polylogarithm form of n and P against direct numerical integration of the density of states;
  * the CLASSICAL LIMIT, both ways: as the fugacity -> 0 the Fermi and Bose occupations approach Maxwell-Boltzmann with
    relative deviation -+ e^(-(e - mu)/T) to first order (so the ratio (f - f_MB)/(f_MB e^-y) -> -+1), and the equation of
    state P/(nT) = 1 +- n lambda^3 / (2^(5/2) g) (the second virial coefficient, +: fermions push harder than classical,
    -: bosons less) is recovered with slope +-2^(-5/2) = 0.1768;
  * the DEGENERATE FERMI limit: mu -> E_F = (1/2)(6 pi^2 n / g)^(2/3) and the Sommerfeld expansion
    mu = E_F [1 - (pi^2/12)(T/E_F)^2 + ...], P -> (2/5) n E_F;
  * the BOSE gas: mu -> 0^- as T -> T_c from above, a condensate below, and P = g T zeta(5/2)/lambda^3 there;
  * the three statistics give the SAME mu in the dilute limit, and every solved mu reproduces the requested n.
"""
import math
import warnings
from dataclasses import dataclass, field

import mpmath
import numpy as np
from scipy.integrate import IntegrationWarning
from scipy.integrate import quad as _quad
from scipy.optimize import brentq
from scipy.special import zeta
from scipy.stats import kstest, maxwell

from modules.quantum_common import CheckList
from modules.sm_common import StatMechError, resolve_seed

KINDS = ("MB", "FD", "BE")


def quad(*args, **kwargs):
    """scipy's quad without its roundoff warnings: for a nearly step-shaped (degenerate Fermi) integrand the requested 1e-11 cannot always
    be met, and the accuracy is verified by independent checks (verify_gas_tools) instead of by warnings."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", IntegrationWarning)
        return _quad(*args, **kwargs)


ZETA_32 = float(zeta(1.5))
ZETA_52 = float(zeta(2.5))


# ------------------------------------------------------------------------------------------ distributions

def occupation(eps: np.ndarray | float, mu: float, T: float, kind: str) -> np.ndarray:
    """Mean occupation of a single-particle level of energy eps."""
    if T <= 0:
        raise StatMechError("T must be positive.")
    if kind not in KINDS:
        raise StatMechError(f"kind must be one of {', '.join(KINDS)}.")
    y = (np.asarray(eps, dtype=float) - mu) / T
    if kind == "MB":
        return np.exp(-y)
    if kind == "FD":
        return 0.5 * (1 - np.tanh(y / 2))                    # = 1/(e^y + 1), without overflow
    if np.any(y <= 0):
        raise StatMechError("Bose-Einstein occupation needs every level above the chemical potential (eps > mu).")
    with np.errstate(over="ignore"):
        return 1.0 / np.expm1(y)


def mb_speed_pdf(v: np.ndarray, T: float, m: float = 1.0) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    return 4 * np.pi * v ** 2 * (m / (2 * np.pi * T)) ** 1.5 * np.exp(-m * v ** 2 / (2 * T))


def mb_moments(T: float, m: float = 1.0) -> dict[str, float]:
    return {"most probable speed": math.sqrt(2 * T / m), "mean speed": math.sqrt(8 * T / (math.pi * m)), "rms speed": math.sqrt(3 * T / m)}


def mb_sample_check(T: float, n: int = 200_000, seed: int | None = None, m: float = 1.0) -> tuple[CheckList, int]:
    """Draw Gaussian velocity components (variance T/m) and test that the speeds follow the Maxwell distribution."""
    seed = resolve_seed(seed)
    rng = np.random.default_rng(seed)
    speeds = np.linalg.norm(rng.normal(0, math.sqrt(T / m), size=(n, 3)), axis=1)
    ck = CheckList()
    ks = kstest(speeds, maxwell(scale=math.sqrt(T / m)).cdf)
    ck.add("sampled speeds follow the Maxwell distribution (Kolmogorov–Smirnov)", ks.pvalue > 1e-3, f"KS statistic {ks.statistic:.2e}, p = {ks.pvalue:.2g}")
    mm = mb_moments(T, m)
    ck.add("sample mean speed = √(8kT/πm)", abs(speeds.mean() - mm["mean speed"]) < 5 * speeds.std() / math.sqrt(n), f"{speeds.mean():.5g} vs {mm['mean speed']:.5g}")
    ck.add("equipartition: ⟨mv²/2⟩ = (3/2) kT", abs(0.5 * m * (speeds ** 2).mean() - 1.5 * T) < 5 * 0.5 * m * (speeds ** 2).std() / math.sqrt(n),
           f"{0.5 * m * (speeds ** 2).mean():.5g} vs {1.5 * T:.5g}")
    return ck, seed


# ------------------------------------------------------------------------------------------ ideal gas

def thermal_wavelength(T: float, m: float = 1.0) -> float:
    return math.sqrt(2 * math.pi / (m * T))


def _F(s: float, z: float, kind: str) -> float:
    if kind == "MB":
        return z
    mpmath.mp.dps = 25
    if kind == "FD":
        return float(-mpmath.re(mpmath.polylog(s, -z)))        # for z > 1 the continuation carries a ~0 imaginary part
    return float(mpmath.re(mpmath.polylog(s, z)))


def _F_fast(s: float, lz: float, kind: str) -> float:
    """F_s(z) with ln z = lz, from its integral representation (1/Gamma(s)) int x^(s-1) / (e^(x - lz) -+ 1) dx -- far faster than the
    polylogarithm, used inside the root finders. The polylogarithm (_F) stays as the independent reference for the checks."""
    if kind == "MB":
        return math.exp(lz)
    if kind == "FD":
        f = lambda x: x ** (s - 1) * 0.5 * (1 - math.tanh((x - lz) / 2))  # noqa: E731
        val, _ = quad(f, 0, max(lz, 0.0) + 45.0, points=[lz] if lz > 0 else None, limit=300, epsabs=0, epsrel=1e-11)
    else:
        f = lambda x: x ** (s - 1) * math.exp(-(x - lz)) / -math.expm1(-(x - lz))  # noqa: E731   (no overflow for very negative lz)
        val, _ = quad(f, 0, 60.0, limit=500, epsabs=0, epsrel=1e-11)
    return val / math.gamma(s)


def density(mu: float, T: float, kind: str, g: int = 1, m: float = 1.0) -> float:
    """n(mu, T) from the polylogarithm form."""
    if kind == "BE" and mu >= 0:
        raise StatMechError("Bose-Einstein needs mu < 0 (at mu = 0 the excited states are saturated).")
    return g / thermal_wavelength(T, m) ** 3 * _F(1.5, math.exp(mu / T), kind)


def pressure(mu: float, T: float, kind: str, g: int = 1, m: float = 1.0) -> float:
    if kind == "BE" and mu > 0:
        raise StatMechError("Bose-Einstein needs mu <= 0.")
    return g * T / thermal_wavelength(T, m) ** 3 * _F(2.5, math.exp(mu / T), kind)


def density_quad(mu: float, T: float, kind: str, g: int = 1, m: float = 1.0) -> float:
    """n from numerical integration of the density of states: g/(4 pi^2) (2m)^(3/2) int sqrt(e) f(e) de."""
    f = lambda e: math.sqrt(e) * float(occupation(e, mu, T, kind))  # noqa: E731
    pts = [max(mu, 0.0), max(mu, 0.0) + 10 * T] if mu > 0 else None
    val, _ = quad(f, 0, np.inf, points=None, limit=400, epsabs=1e-14, epsrel=1e-12) if pts is None else \
        (quad(f, 0, mu + 40 * T, points=[mu], limit=400, epsabs=1e-14, epsrel=1e-12))
    return g / (4 * math.pi ** 2) * (2 * m) ** 1.5 * val


def pressure_quad(mu: float, T: float, kind: str, g: int = 1, m: float = 1.0) -> float:
    """P = (2/3) u with u = g/(4 pi^2) (2m)^(3/2) int e^(3/2) f(e) de."""
    f = lambda e: e ** 1.5 * float(occupation(e, mu, T, kind))  # noqa: E731
    upper = np.inf if mu <= 0 else mu + 40 * T
    val, _ = quad(f, 0, upper, limit=400, epsabs=1e-14, epsrel=1e-12)
    return 2 / 3 * g / (4 * math.pi ** 2) * (2 * m) ** 1.5 * val


def fermi_energy(n: float, g: int = 1, m: float = 1.0) -> float:
    return (6 * math.pi ** 2 * n / g) ** (2 / 3) / (2 * m)


def sommerfeld_mu(n: float, T: float, g: int = 1, m: float = 1.0) -> float:
    ef = fermi_energy(n, g, m)
    return ef * (1 - math.pi ** 2 / 12 * (T / ef) ** 2)


def bec_critical_temperature(n: float, g: int = 1, m: float = 1.0) -> float:
    return 2 * math.pi / m * (n / (g * ZETA_32)) ** (2 / 3)


@dataclass
class GasState:
    kind: str
    n: float
    T: float
    g: int
    mu: float
    pressure: float
    energy_density: float
    condensate_fraction: float = 0.0

    @property
    def degeneracy(self) -> float:
        """n lambda^3 / g: the quantum-degeneracy parameter (<< 1 classical)."""
        return self.n * thermal_wavelength(self.T) ** 3 / self.g


def solve_mu(n: float, T: float, kind: str, g: int = 1, m: float = 1.0) -> GasState:
    """The gas state with density n at temperature T for the chosen statistics."""
    if n <= 0 or T <= 0 or g < 1 or m <= 0:
        raise StatMechError("Need n > 0, T > 0, g >= 1 and m > 0.")
    if kind not in KINDS:
        raise StatMechError(f"kind must be one of {', '.join(KINDS)}.")
    lam3 = thermal_wavelength(T, m) ** 3
    target = n * lam3 / g                                       # = F_3/2(z)
    if kind == "MB":
        mu = T * math.log(target)
        return GasState(kind, n, T, g, mu, n * T, 1.5 * n * T)
    if kind == "BE":
        if target >= ZETA_32:
            frac0 = 1 - (T / bec_critical_temperature(n, g, m)) ** 1.5
            p = g * T / lam3 * ZETA_52
            return GasState(kind, n, T, g, 0.0, p, 1.5 * p, frac0)
        lz = brentq(lambda v: _F_fast(1.5, v, "BE") - target, -700.0, -1e-14, xtol=1e-15, rtol=1e-14)
    else:
        lo, hi = -700.0, 1.0
        while _F_fast(1.5, hi, "FD") < target:
            hi *= 2
        lz = brentq(lambda v: _F_fast(1.5, v, "FD") - target, lo, hi, xtol=1e-15, rtol=1e-14)
    mu = T * lz
    p = g * T / lam3 * _F_fast(2.5, lz, kind)
    return GasState(kind, n, T, g, mu, p, 1.5 * p)


def mu_curves(n: float, temps: np.ndarray, g: int = 1) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {k: [] for k in KINDS}
    for t in temps:
        for k in KINDS:
            out[k].append(solve_mu(n, float(t), k, g).mu)
    return {k: np.array(v) for k, v in out.items()}


def equation_of_state(x: np.ndarray, T: float = 1.0, g: int = 1) -> dict[str, np.ndarray]:
    """P/(nT) against the degeneracy parameter x = n lambda^3 / g for the three statistics."""
    out: dict[str, list[float]] = {k: [] for k in KINDS}
    lam3 = thermal_wavelength(T) ** 3
    for xv in x:
        n = float(xv) * g / lam3
        for k in KINDS:
            try:
                out[k].append(solve_mu(n, T, k, g).pressure / (n * T))
            except StatMechError:
                out[k].append(float("nan"))
    return {k: np.array(v) for k, v in out.items()}


# ------------------------------------------------------------------------------------------ checks

@dataclass
class GasReport:
    states: dict[str, GasState]
    checks: CheckList = field(default_factory=CheckList)


def verify_gas_tools(n: float, T: float, g: int = 1) -> GasReport:
    """Solve the three statistics at (n, T) and run every consistency check of the module docstring."""
    states = {k: solve_mu(n, T, k, g) for k in KINDS}
    rep = GasReport(states)
    ck = rep.checks
    for k in ("FD", "BE"):
        st = states[k]
        if k == "BE" and st.condensate_fraction > 0:
            ck.add("BE below T_c: μ = 0 and a condensate holds the rest", st.mu == 0 and 0 < st.condensate_fraction < 1,
                   f"condensate fraction {st.condensate_fraction:.4f} = 1 − (T/T_c)^(3/2), T_c = {bec_critical_temperature(n, g):.4g}")
            ck.add("BE below T_c: P = g T ζ(5/2)/λ³", abs(st.pressure - g * T * ZETA_52 / thermal_wavelength(T) ** 3) < 1e-12 * st.pressure, f"P = {st.pressure:.6g}")
            continue
        back = density(st.mu, T, k, g)
        ck.add(f"{k}: solved μ reproduces the requested n (polylogarithm, independent of the solver)", abs(back - n) < 1e-9 * n, f"n(μ) = {back:.10g}, requested {n:.10g}")
        nq, pq = density_quad(st.mu, T, k, g), pressure_quad(st.mu, T, k, g)
        pp = pressure(st.mu, T, k, g)
        ck.add(f"{k}: polylogarithm n, P = direct integration over the density of states = the solver's P",
               abs(nq - n) < 1e-7 * n and abs(pq - st.pressure) < 1e-7 * st.pressure and abs(pp - st.pressure) < 1e-7 * st.pressure,
               f"n: {nq:.8g} vs {n:.8g}; P: integral {pq:.8g}, polylog {pp:.8g}, solver {st.pressure:.8g}")
    ck.add("P = (2/3) u for every statistics", all(abs(s.pressure - s.energy_density * 2 / 3) < 1e-12 * s.pressure for s in states.values()), "non-relativistic ideal gas")
    # classical limit of the occupations
    y = np.array([8.0, 10.0, 12.0])
    mb = occupation(y * T, 0.0, T, "MB")
    fd = (occupation(y * T, 0.0, T, "FD") - mb) / (mb * np.exp(-y))
    be = (occupation(y * T, 0.0, T, "BE") - mb) / (mb * np.exp(-y))
    ck.add("classical limit: (f_FD − f_MB)/(f_MB e^−y) → −1 and (f_BE − f_MB)/(f_MB e^−y) → +1 as y → ∞",
           bool(np.all(np.abs(fd + 1) < 2e-3) and np.all(np.abs(be - 1) < 2e-3)), f"FD {fd[-1]:.5f}, BE {be[-1]:.5f} at y = 12")
    # second virial coefficient
    xs = np.array([2e-3, 1e-3])
    eos = equation_of_state(xs, 1.0, g)
    slope_fd, slope_be = float(np.mean((eos["FD"] - 1) / xs)), float(np.mean((eos["BE"] - 1) / xs))
    want = 2 ** -2.5
    ck.add("second virial coefficient: P/nT − 1 = ± nλ³/(2^(5/2) g)", abs(slope_fd - want) < 2e-3 and abs(slope_be + want) < 2e-3,
           f"slopes {slope_fd:+.4f} (FD), {slope_be:+.4f} (BE); expected ±{want:.4f}")
    # degenerate Fermi gas
    ef = fermi_energy(1.0, g)
    t_low = 0.02 * ef
    st = solve_mu(1.0, t_low, "FD", g)
    ck.add("Fermi gas: μ = E_F[1 − (π²/12)(T/E_F)²] (Sommerfeld) at T = 0.02 E_F", abs(st.mu - sommerfeld_mu(1.0, t_low, g)) < 2e-5 * ef,
           f"μ = {st.mu:.7g}, Sommerfeld {sommerfeld_mu(1.0, t_low, g):.7g}, E_F = {ef:.7g}")
    st0 = solve_mu(1.0, 0.002 * ef, "FD", g)
    ck.add("Fermi gas: P → (2/5) n E_F as T → 0", abs(st0.pressure - 0.4 * 1.0 * ef) < 1e-3 * 0.4 * ef, f"P = {st0.pressure:.6g}, (2/5)nE_F = {0.4 * ef:.6g}")
    # BEC threshold
    tc = bec_critical_temperature(1.0, g)
    above, below = solve_mu(1.0, 1.001 * tc, "BE", g), solve_mu(1.0, 0.999 * tc, "BE", g)
    ck.add("Bose gas: μ → 0⁻ at T_c = (2π/m)(n/gζ(3/2))^(2/3); a condensate appears just below", above.mu < 0 and abs(above.mu) < 0.05 * tc and below.condensate_fraction > 0,
           f"μ(1.001 T_c) = {above.mu:.3g}, condensate fraction at 0.999 T_c = {below.condensate_fraction:.4f}")
    dil = {k: solve_mu(1e-4, 1.0, k, g).mu for k in KINDS}
    ck.add("dilute limit: MB, FD and BE give the same μ", max(abs(dil["FD"] - dil["MB"]), abs(dil["BE"] - dil["MB"])) < 1e-3, f"μ = {dil['MB']:.5g}, {dil['FD']:.5g}, {dil['BE']:.5g}")
    return rep
