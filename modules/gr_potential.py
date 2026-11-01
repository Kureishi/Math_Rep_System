"""
Effective potentials and the characteristic radii of black-hole spacetimes -- Streamlit-free.

STATIC SPHERICALLY SYMMETRIC metrics  ds^2 = -f dt^2 + dr^2/f + r^2 dOmega^2  (Schwarzschild, Reissner-Nordstrom,
Schwarzschild-de Sitter, or any f(r) typed by the user). A geodesic with conserved E = f dt/dtau and
L = r^2 dphi/dtau obeys

    (dr/dtau)^2 = E^2 - V(r),      V = f (kappa' + L^2/r^2)          kappa' = 1 (massive), 0 (light)

so everything about radial motion is read off V. The characteristic radii follow from f alone:

    horizons          f = 0
    photon sphere     2 f - r f' = 0                (extremum of f / r^2: a circular light orbit)
    circular orbits   L^2(r) = r^3 f' / (2 f - r f'),   E^2(r) = 2 f^2 / (2 f - r f')
    ISCO              d L^2 / dr = 0 (the minimum of L^2 along the circular orbits: inside it none is stable)
    marginally bound  E^2(r) = 1 (a circular orbit with exactly the energy to escape)

For Schwarzschild these give 2M, 3M, 6M, 4M, which the tests check; the other metrics have no textbook
values to compare with, so the radii are found NUMERICALLY (a dense sign-change scan refined by Brent's method)
from the same formulas, and cross-checked: at every circular orbit found, V = E^2 and V' = 0 are verified.

KERR, equatorial plane. For E, L and unit mass the radial equation is r^4 (dr/dtau)^2 = R(r) with
R = [E(r^2+a^2) - aL]^2 - Delta [r^2 + (L - aE)^2]. R is quadratic in E, so the "effective potential" is its
upper root E_+(r; L) (motion is allowed where E >= E_+). Horizons, the ISCO, photon orbits and the marginally
bound orbit have closed forms (Bardeen-Press-Teukolsky), and each is cross-checked numerically: the circular
orbit formulas must satisfy R = R' = 0, and the numerical ISCO (minimum of L along circular orbits) must agree
with the closed form.

SCHWARZSCHILD, exact results (elliptic integrals), used to check the geodesic integrator:
  * perihelion precession of an orbit with turning points r_a, r_p:  4 K(m) / sqrt(2M (u_c - u_a)) - 2 pi,
    with u = 1/r, u_c = 1/2M - u_a - u_p the third root of the orbit cubic;
  * light deflection between radii R:  2 * integral du / sqrt(1/b^2 - u^2 + 2 M u^3) - pi.
"""
import math
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import brentq
from scipy.special import ellipk

from modules.gr_metrics import GRInputError, GRMetric, coordinate_symbol, parameter_symbol, parse_in_symbols

R_SYMBOL = coordinate_symbol("r")


# ------------------------------------------------------------------------------------------ static f(r)

@dataclass
class StaticSpacetime:
    f: sp.Expr
    params: dict[str, float] = field(default_factory=dict)
    name: str = "f(r)"

    def _lam(self, expr: sp.Expr):
        syms = [R_SYMBOL] + [parameter_symbol(k) for k in sorted(self.params)]
        return sp.lambdify(syms, expr, modules="numpy")

    def values(self, params: dict[str, float] | None = None) -> list[float]:
        merged = {**self.params, **(params or {})}
        return [float(merged[k]) for k in sorted(self.params)]

    def func(self, expr: sp.Expr, params: dict[str, float] | None = None):
        fn = self._lam(expr)
        vals = self.values(params)
        return lambda r: np.asarray(fn(np.asarray(r, dtype=float), *vals), dtype=float) * np.ones_like(np.asarray(r, dtype=float))


def static_from_metric(metric: GRMetric) -> StaticSpacetime:
    if metric.f_of_r is None:
        raise GRInputError(f"'{metric.name}' is not a static spherically symmetric f(r) metric.")
    return StaticSpacetime(metric.f_of_r, dict(metric.params), metric.name)


def static_from_text(f_text: str, param_defaults: dict[str, float] | None = None, name: str = "f(r)") -> StaticSpacetime:
    """f(r) typed by the user; every name other than r is a parameter (positive, default 1)."""
    from modules.gr_metrics import _names_in
    names = sorted(_names_in(f_text) - {"r"})
    syms = {"r": R_SYMBOL, **{n: parameter_symbol(n) for n in names}}
    f = parse_in_symbols(f_text, syms)
    if R_SYMBOL not in f.free_symbols and not f.free_symbols:
        f = sp.sympify(f)
    return StaticSpacetime(f, {n: float((param_defaults or {}).get(n, 1.0)) for n in names}, name)


@dataclass
class Marker:
    label: str
    r: float
    kind: str                     # "horizon" | "photon sphere" | "ISCO" | "stability limit" | "marginally bound"
    stable: bool | None = None


def _roots(fn, lo: float, hi: float, n: int = 40000) -> list[float]:
    xs = np.linspace(lo, hi, n)
    with np.errstate(all="ignore"):
        ys = fn(xs)
    out = []
    for i in range(n - 1):
        a, b = ys[i], ys[i + 1]
        if np.isfinite(a) and np.isfinite(b) and a * b < 0 and abs(a) < 1e12 and abs(b) < 1e12:
            try:
                x = brentq(lambda x: float(fn(np.array([x]))[0]), xs[i], xs[i + 1], xtol=1e-14)
            except ValueError:
                continue
            if abs(float(fn(np.array([x]))[0])) < 1e-6 * (1 + abs(a) + abs(b)):     # a sign flip through a pole is not a root
                out.append(x)
        elif np.isfinite(a) and a == 0.0:
            out.append(float(xs[i]))
    return out


def static_markers(st: StaticSpacetime, params: dict[str, float] | None = None, r_max: float = 60.0,
                   r_min: float = 1e-3) -> list[Marker]:
    """Horizons, photon spheres, circular-orbit stability limits and the marginally bound orbit of f(r)."""
    r = R_SYMBOL
    f = st.f
    fp = sp.diff(f, r)
    h = 2 * f - r * fp
    ell = r ** 3 * fp / h                                    # L^2 of the circular orbit at r
    e2 = 2 * f ** 2 / h
    f_n, h_n = st.func(f, params), st.func(h, params)
    ell_n, dell_n, d2ell_n = (st.func(ell, params), st.func(sp.diff(ell, r), params), st.func(sp.diff(ell, r, 2), params))
    e2_n = st.func(e2, params)
    out: list[Marker] = []
    horizons = _roots(f_n, r_min, r_max)
    for k, x in enumerate(horizons):
        out.append(Marker(f"horizon r = {x:.4g}", x, "horizon"))
    r_h = max(horizons) if horizons else r_min
    # photon spheres: h = 0, stable if f/r^2 has a minimum there
    v0 = st.func(f / r ** 2, params)
    for x in _roots(h_n, max(r_min, 1e-3), r_max):
        if f_n(np.array([x]))[0] <= 0:
            continue
        d2 = (v0(x + 1e-4 * x) - 2 * v0(x) + v0(x - 1e-4 * x)) / (1e-4 * x) ** 2
        out.append(Marker(f"photon sphere r = {x:.4g}", x, "photon sphere", stable=bool(d2 > 0)))
    valid = lambda x: bool(f_n(np.array([x]))[0] > 0 and h_n(np.array([x]))[0] > 0 and ell_n(np.array([x]))[0] > 0)  # noqa: E731
    for x in _roots(dell_n, max(r_min, 1e-3), r_max):
        if not valid(x):
            continue
        second = float(d2ell_n(np.array([x]))[0])
        if second > 0:
            out.append(Marker(f"ISCO r = {x:.4g}", x, "ISCO", stable=True))
        else:
            out.append(Marker(f"outer stability limit r = {x:.4g}", x, "stability limit", stable=False))
    for x in _roots(lambda rr: e2_n(rr) - 1.0, max(r_min, 1e-3), r_max):
        if valid(x):
            out.append(Marker(f"marginally bound r = {x:.4g}", x, "marginally bound", stable=False))
    del r_h
    return sorted(out, key=lambda m: m.r)


def veff_static(st: StaticSpacetime, L: float, r_values: np.ndarray, params: dict[str, float] | None = None,
                light: bool = False) -> np.ndarray:
    """V(r) = f (1 + L^2/r^2) for a massive particle, f L^2/r^2 for light: motion needs E^2 >= V."""
    f = st.func(st.f, params)
    rr = np.asarray(r_values, dtype=float)
    return f(rr) * ((0.0 if light else 1.0) + L ** 2 / rr ** 2)


def circular_orbit_static(st: StaticSpacetime, r0: float, params: dict[str, float] | None = None) -> tuple[float, float] | None:
    """(E, L) of the circular geodesic at r0, or None if none exists there."""
    r = R_SYMBOL
    h = st.func(2 * st.f - r * sp.diff(st.f, r), params)(np.array([r0]))[0]
    f0 = st.func(st.f, params)(np.array([r0]))[0]
    fp0 = st.func(sp.diff(st.f, r), params)(np.array([r0]))[0]
    if h <= 0 or f0 <= 0 or fp0 <= 0:
        return None
    return math.sqrt(2 * f0 ** 2 / h), math.sqrt(r0 ** 3 * fp0 / h)


def turning_points_static(st: StaticSpacetime, E: float, L: float, params: dict[str, float] | None = None,
                          r_max: float = 200.0, light: bool = False) -> list[float]:
    """Radii where E^2 = V(r) outside the outer horizon."""
    f = st.func(st.f, params)
    pot = lambda rr: veff_static(st, L, rr, params, light) - E ** 2  # noqa: E731
    hor = _roots(f, 1e-3, r_max)
    lo = (max(hor) if hor else 0.0) + 1e-6
    return _roots(pot, lo, r_max)


# ------------------------------------------------------------------------------------------ Kerr equatorial

def kerr_horizons(M: float, a: float) -> tuple[float, float] | None:
    """(r_-, r_+), or None for a naked singularity (|a| > M)."""
    disc = M * M - a * a
    return None if disc < 0 else (M - math.sqrt(disc), M + math.sqrt(disc))


def kerr_isco(M: float, a: float, prograde: bool = True) -> float:
    """Bardeen-Press-Teukolsky: r = M (3 + Z2 -/+ sqrt((3 - Z1)(3 + Z1 + 2 Z2)))."""
    x = a / M
    z1 = 1 + (1 - x * x) ** (1 / 3) * ((1 + x) ** (1 / 3) + (1 - x) ** (1 / 3))
    z2 = math.sqrt(3 * x * x + z1 * z1)
    root = math.sqrt((3 - z1) * (3 + z1 + 2 * z2))
    return M * (3 + z2 - root) if prograde else M * (3 + z2 + root)


def kerr_photon_orbit(M: float, a: float, prograde: bool = True) -> float:
    """r = 2M (1 + cos(2/3 arccos(-+ a/M)))."""
    return 2 * M * (1 + math.cos(2 / 3 * math.acos(-a / M if prograde else a / M)))


def kerr_marginally_bound(M: float, a: float, prograde: bool = True) -> float:
    s = 1 if prograde else -1
    return 2 * M - s * a + 2 * math.sqrt(M) * math.sqrt(M - s * a)


def kerr_circular(M: float, a: float, r: float, prograde: bool = True) -> tuple[float, float]:
    """(E, L) of the equatorial circular orbit at r (unit mass); L < 0 for a retrograde orbit."""
    s = 1 if prograde else -1
    sq = math.sqrt(M)
    den = r ** 0.75 * math.sqrt(r ** 1.5 - 3 * M * math.sqrt(r) + 2 * s * a * sq)
    E = (r ** 1.5 - 2 * M * math.sqrt(r) + s * a * sq) / den
    L = s * sq * (r * r - 2 * s * a * math.sqrt(M * r) + a * a) / den
    return E, L


def kerr_radial_R(r, E: float, L: float, M: float, a: float, mu2: float = 1.0):
    """R(r) with r^4 (dr/dtau)^2 = R, equatorial plane (mu2 = 1 massive, 0 light)."""
    delta = r * r - 2 * M * r + a * a
    return (E * (r * r + a * a) - a * L) ** 2 - delta * (mu2 * r * r + (L - a * E) ** 2)


def kerr_veff(r, L: float, M: float, a: float, light: bool = False):
    """E_+(r; L): the upper root of R(r) = 0 in E. Motion is allowed where E >= E_+ (for E > 0)."""
    r = np.asarray(r, dtype=float)
    delta = r * r - 2 * M * r + a * a
    A = r ** 4 + a * a * r * r + 2 * M * a * a * r
    mu2 = 0.0 if light else 1.0
    D = 4 * M * M * r * r * a * a * L * L + A * ((r * r - 2 * M * r) * L * L + delta * mu2 * r * r)
    with np.errstate(invalid="ignore"):
        return (2 * M * r * a * L + np.sqrt(D)) / A


@dataclass
class KerrCharacteristics:
    M: float
    a: float
    inner_horizon: float | None
    outer_horizon: float | None
    ergosphere_equator: float
    isco_prograde: float
    isco_retrograde: float
    photon_prograde: float
    photon_retrograde: float
    bound_prograde: float
    bound_retrograde: float
    numeric_isco_error: float          # |closed-form ISCO - numerical minimum of L along circular orbits|
    circular_residual: float           # max |R|, |R'| over the circular orbits at the ISCO, photon-free check


@lru_cache(maxsize=64)
def kerr_characteristics(M: float, a: float) -> KerrCharacteristics:
    if M <= 0 or abs(a) > M:
        raise GRInputError("Kerr needs M > 0 and |a| <= M (otherwise there is no horizon: a naked singularity).")
    a = abs(a)
    hor = kerr_horizons(M, a)
    pro, retro = kerr_isco(M, a, True), kerr_isco(M, a, False)
    # numerical ISCO: minimum of L(r) along prograde circular orbits, bracketed around the closed form
    lo, hi = pro * 0.7, pro * 1.6
    xs = np.linspace(max(lo, kerr_photon_orbit(M, a, True) * 1.0001), hi, 4001)
    ls = np.array([kerr_circular(M, a, x, True)[1] for x in xs])
    num_isco = float(xs[int(np.argmin(ls))])
    resid = 0.0
    for prograde, rc in ((True, pro), (False, retro)):
        E, L = kerr_circular(M, a, rc, prograde)
        h = 1e-6 * rc
        d = (kerr_radial_R(rc + h, E, L, M, a) - kerr_radial_R(rc - h, E, L, M, a)) / (2 * h)
        resid = max(resid, abs(float(kerr_radial_R(rc, E, L, M, a))), abs(float(d)))
    return KerrCharacteristics(M, a, hor[0] if hor else None, hor[1] if hor else None, 2 * M, pro, retro,
                               kerr_photon_orbit(M, a, True), kerr_photon_orbit(M, a, False),
                               kerr_marginally_bound(M, a, True), kerr_marginally_bound(M, a, False),
                               abs(num_isco - pro), resid)


# ------------------------------------------------------------------------------------------ Schwarzschild exact

def schwarzschild_orbit_from_turning_points(M: float, r_a: float, r_p: float) -> tuple[float, float]:
    """(E, L) of the bound orbit with apoapsis r_a and periapsis r_p (unit mass). Raises if r_p lies inside the
    separatrix (no stable bound orbit has that periapsis)."""
    if not (r_p < r_a) or r_p <= 2 * M:
        raise GRInputError("Need 2M < r_p < r_a.")
    ua, up = 1 / r_a, 1 / r_p
    uc = 1 / (2 * M) - ua - up
    if uc <= up:
        raise GRInputError("That periapsis is inside the separatrix: the particle would plunge, not orbit "
                           f"(need r_p larger; with r_a = {r_a:g} and M = {M:g} the limit is where u_c = u_p).")
    L2 = 1 / (ua * up + (ua + up) * uc)
    E2 = 1 - 2 * M * ua * up * uc * L2
    return math.sqrt(E2), math.sqrt(L2)


def schwarzschild_precession(M: float, r_a: float, r_p: float) -> float:
    """Exact advance of the periapsis per orbit (radians): 4 K(m) / sqrt(2M (u_c - u_a)) - 2 pi."""
    ua, up = 1 / r_a, 1 / r_p
    uc = 1 / (2 * M) - ua - up
    if uc <= up:
        raise GRInputError("That periapsis is inside the separatrix: no stable bound orbit.")
    m = (up - ua) / (uc - ua)
    return 4 * float(ellipk(m)) / math.sqrt(2 * M * (uc - ua)) - 2 * math.pi


def weak_field_precession(M: float, r_a: float, r_p: float) -> float:
    """6 pi M / p with p = 2 r_a r_p / (r_a + r_p) the semi-latus rectum: the first-order (Einstein 1915) result."""
    return 6 * math.pi * M * (r_a + r_p) / (2 * r_a * r_p)


def schwarzschild_deflection(M: float, b: float, R: float | None = None) -> float:
    """Exact light deflection for impact parameter b between radius R and R (R = None: from infinity), i.e. the
    total sweep of phi minus pi. Raises for b below the capture threshold 3 sqrt(3) M."""
    if b <= 3 * math.sqrt(3) * M:
        raise GRInputError(f"b = {b:g} is inside the capture cross-section (b < 3*sqrt(3)*M = {3 * math.sqrt(3) * M:.4g}): "
                           "the photon falls in.")
    roots = [x.real for x in np.roots([2 * M, -1, 0, 1 / b ** 2]) if abs(x.imag) < 1e-12 and x.real > 0]
    u0 = min(roots)
    uR = 0.0 if R is None else 1 / R
    if uR >= u0:
        raise GRInputError("The starting radius is inside the periapsis of this ray.")
    # g(u) = 1/b^2 - u^2 + 2 M u^3 = (u0 - u) q(u). Substituting u = u0 - w^2 turns the inverse-square-root
    # endpoint singularity into a smooth integrand: du / sqrt(g) = 2 dw / sqrt(q(u0 - w^2)).
    def q(u: float) -> float:
        return -(2 * M * u * u + (2 * M * u0 - 1) * u + 2 * M * u0 * u0 - u0)

    from scipy.integrate import quad
    w_max = math.sqrt(u0 - uR)
    val, _err = quad(lambda w: 2 / math.sqrt(q(u0 - w * w)), 0, w_max, epsabs=1e-14, epsrel=1e-13, limit=200)
    return 2 * val - math.pi


def weak_field_deflection(M: float, b: float) -> float:
    """4M/b + 15 pi M^2 / (4 b^2): the first two terms of the expansion in M/b."""
    return 4 * M / b + 15 * math.pi * M * M / (4 * b * b)
