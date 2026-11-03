"""
Visual showpieces -- Streamlit-free, no LLM. Three things worth looking at, each with a number that shows the picture is right.

1. BLACK-HOLE SHADOW AND LENSING (Schwarzschild). Every pixel's light ray is traced BACKWARDS from the observer by
   integrating the orbit equation  u'' + u = 3 M u^2  (u = 1/r, derivative with respect to the orbit angle) for all rays
   at once. A ray that reaches the horizon is captured (black: the shadow); one that escapes arrives from a sky direction
   bent by the hole, which is read off a procedural starfield; one that winds more than the integration allows is hugging
   the photon sphere (the bright ring). CHECKS: the shadow radius against the exact b_c = 3 sqrt(3) M, and the weak-field
   deflection against 2M/b (1 + sqrt(1 - b^2/D^2)). Units G = c = 1; lengths in M.
2. ORBIT AND SPACETIME EMBEDDING. A bound orbit about the same hole (Darwin parametrisation: semi-latus rectum p and
   eccentricity e) integrated from the orbit equation u'' + u = M/L^2 + 3 M u^2, drawn on Flamm's paraboloid, the
   surface whose geometry is the hole's spatial geometry on the equatorial plane. CHECKS: a conserved quantity along the
   orbit, and the measured perihelion advance against 6 pi / p.
3. THE CRITICAL POINT. The 2-D Ising model just below, at, and above its exact T_c = 2 / ln(1 + sqrt 2), run side by
   side (modules/sm_lattice.py, GPU-backed for large lattices): ordered domains, clusters of every size, and noise.
"""
import math
from dataclasses import dataclass, field

import numpy as np

from modules import gpu
from modules import sm_lattice as lat
from modules.sm_meanfield import onsager_tc

B_CRIT = 3.0 * math.sqrt(3.0)          # critical impact parameter in units of M


class ShowcaseError(ValueError):
    pass


# ------------------------------------------------------------------------------------------ 1. ray tracing

CAPTURED, ESCAPED, WINDING = 1, 0, 2


def _trace(xp, alpha: np.ndarray, M: float, D: float, dphi: float, phi_max: float):
    """(phi_inf, status) for rays leaving the observer at radius D at angle `alpha` from the direction of the hole."""
    n = alpha.size
    a = xp.maximum(xp.asarray(alpha), 1e-9)
    b = D * xp.sin(a) / math.sqrt(1.0 - 2.0 * M / D)
    u0 = 1.0 / D
    w = xp.sqrt(xp.maximum(1.0 / b ** 2 - u0 ** 2 * (1.0 - 2.0 * M * u0), 0.0))     # du/dphi, moving inward
    u = xp.full(n, u0)
    idx = xp.arange(n)
    phi_inf = xp.full(n, xp.nan)
    status = xp.full(n, WINDING, dtype=xp.int8)
    uc, h, phi = 1.0 / (2.0 * M), dphi, 0.0
    m3 = 3.0 * M
    while phi < phi_max and idx.size:
        k1u, k1w = w, -u + m3 * u * u
        u2, w2 = u + 0.5 * h * k1u, w + 0.5 * h * k1w
        k2u, k2w = w2, -u2 + m3 * u2 * u2
        u3, w3 = u + 0.5 * h * k2u, w + 0.5 * h * k2w
        k3u, k3w = w3, -u3 + m3 * u3 * u3
        u4, w4 = u + h * k3u, w + h * k3w
        k4u, k4w = w4, -u4 + m3 * u4 * u4
        un = u + h / 6.0 * (k1u + 2 * k2u + 2 * k3u + k4u)
        wn = w + h / 6.0 * (k1w + 2 * k2w + 2 * k3w + k4w)
        cap = un >= uc
        esc = (un <= 0.0) & ~cap
        gap = xp.where(esc, u - un, 1.0)
        crossing = phi + h * xp.where(esc, u / gap, 0.0)
        status[idx[cap]] = CAPTURED
        status[idx[esc]] = ESCAPED
        phi_inf[idx[esc]] = crossing[esc]
        keep = ~(cap | esc)
        u, w, idx = un[keep], wn[keep], idx[keep]
        phi += h
    return phi_inf, status


def trace_rays(alpha: np.ndarray, M: float = 1.0, D: float = 30.0, dphi: float = 0.02,
               phi_max: float = 4.0 * math.pi) -> tuple[np.ndarray, np.ndarray, str]:
    """(phi_inf, status, backend name): the polar angle at which each escaping ray reaches infinity (measured from the
    observer's radial line, so an undeflected ray gives pi - alpha), and CAPTURED / ESCAPED / WINDING for each ray."""
    alpha = np.asarray(alpha, dtype=float).ravel()
    if D <= 2.0 * M * 1.5 or M <= 0:
        raise ShowcaseError("The observer must be well outside the horizon (D > 3M) and M must be positive.")
    if np.any(alpha < 0) or np.any(alpha > math.pi / 2):
        raise ShowcaseError("Viewing angles must lie within 90 degrees of the direction of the hole.")
    backend = gpu.select(alpha.size, "raytrace_pixels")
    ran_on_gpu = False

    def on_gpu():
        nonlocal ran_on_gpu
        result = _trace(backend.xp, alpha, M, D, dphi, phi_max)
        ran_on_gpu = True
        return result

    out = gpu.attempt(backend, on_gpu, lambda: _trace(np, alpha, M, D, dphi, phi_max))
    return gpu.to_host(out[0]), gpu.to_host(out[1]), (backend if ran_on_gpu else gpu.CPU).name


def expected_shadow_angle(M: float, D: float) -> float:
    """The exact angular radius of the shadow seen by a static observer at D: sin(alpha_c) = b_c sqrt(1 - 2M/D) / D."""
    return math.asin(B_CRIT * M * math.sqrt(1.0 - 2.0 * M / D) / D)


def weak_deflection(b: float, M: float, D: float) -> float:
    """Light bending for a ray of impact parameter b from an observer at D, with the source at infinity, to first order."""
    return 2.0 * M / b * (1.0 + math.sqrt(1.0 - (b / D) ** 2))


def sky_color(direction: np.ndarray) -> np.ndarray:
    """A deterministic starfield with a lon/lat graticule, for unit vectors (N, 3) whose first axis points along the optical
    axis (the direction behind the hole). Returns float RGB (N, 3) in [0, 1]."""
    tx, ty, tz = direction[:, 2], direction[:, 0], direction[:, 1]            # the axis becomes lon = lat = 0
    lon, lat_ = np.arctan2(ty, tx), np.arcsin(np.clip(tz, -1, 1))

    def line(v, step, width):
        d = np.abs(v / step - np.round(v / step)) * step
        return np.exp(-(d / width) ** 2)

    grid = np.maximum(line(lon, math.pi / 12, 0.010), line(lat_, math.pi / 12, 0.010))
    cell = 0.04
    ci, cj = np.floor(lon / cell), np.floor(lat_ / cell)
    hsh = np.sin(ci * 12.9898 + cj * 78.233) * 43758.5453
    hsh -= np.floor(hsh)
    star = np.where(hsh > 0.990, 0.55 + 0.45 * ((hsh * 91.7) % 1.0), 0.0)
    nebula = np.stack([0.05 + 0.05 * np.sin(2 * lon + 1) * np.cos(lat_),
                       0.04 + 0.04 * np.sin(3 * lat_ + lon),
                       0.10 + 0.07 * np.cos(lon - 2 * lat_)], axis=1)
    rgb = nebula + grid[:, None] * np.array([0.25, 0.42, 0.65]) * 0.7 + star[:, None] * np.array([1.0, 0.95, 0.85])
    return np.clip(rgb, 0, 1)


@dataclass
class LensImage:
    rgb: np.ndarray                         # (n, n, 3) uint8, with the black hole
    rgb_flat: np.ndarray                    # the same view of the same sky with no hole
    status: np.ndarray                      # (n, n) CAPTURED / ESCAPED / WINDING
    n: int
    fov_deg: float
    M: float
    D: float
    backend: str
    shadow_px_expected: float
    shadow_px_measured: float | None        # None when the shadow reaches the image edge
    notes: list[str] = field(default_factory=list)

    @property
    def shadow_error(self) -> float | None:
        if self.shadow_px_measured is None:
            return None
        return abs(self.shadow_px_measured - self.shadow_px_expected) / self.shadow_px_expected


def _directions(beta: np.ndarray, psi: np.ndarray) -> np.ndarray:
    return np.stack([np.sin(beta) * np.cos(psi), np.sin(beta) * np.sin(psi), np.cos(beta)], axis=1)


def render_lens_image(n: int = 240, fov_deg: float = 36.0, M: float = 1.0, D: float = 30.0) -> LensImage:
    """The view of a starfield behind a Schwarzschild black hole of mass M from distance D, `n` x `n` pixels."""
    if not 32 <= n <= 1200:
        raise ShowcaseError("The image size must be between 32 and 1200 pixels.")
    if not 5.0 <= fov_deg <= 120.0:
        raise ShowcaseError("The field of view must be between 5 and 120 degrees.")
    half = math.radians(fov_deg) / 2.0
    xs = (np.arange(n) + 0.5) / n * 2 - 1
    gx, gy = np.meshgrid(xs, -xs)
    rho = np.hypot(gx, gy)
    alpha = rho * half
    psi = np.arctan2(gy, gx)
    phi_inf, status, backend = trace_rays(alpha.ravel(), M, D)
    status = status.reshape(n, n)
    beta = math.pi - phi_inf.reshape(n, n)                    # where the ray came from, measured from behind the hole
    sky = np.zeros((n * n, 3))
    esc = (status == ESCAPED).ravel()
    sky[esc] = sky_color(_directions(beta.ravel()[esc], psi.ravel()[esc]))
    ring = (status == WINDING).ravel()
    sky[ring] = np.array([1.0, 0.82, 0.45])                    # photon-sphere glow: rays that have not escaped yet
    flat = sky_color(_directions(alpha.ravel(), psi.ravel()))
    shadow = (status == CAPTURED)
    expected = expected_shadow_angle(M, D) / half * n / 2.0
    measured = None if (shadow[0].any() or shadow[-1].any() or shadow[:, 0].any() or shadow[:, -1].any()) else \
        math.sqrt(float(shadow.sum()) / math.pi)
    notes = []
    if expected_shadow_angle(M, D) > half:
        notes.append("The shadow is larger than the field of view: widen the view or move the observer back.")
    return LensImage((sky.reshape(n, n, 3) * 255).astype(np.uint8), (flat.reshape(n, n, 3) * 255).astype(np.uint8), status, n,
                     fov_deg, M, D, backend, expected, measured, notes)


# ------------------------------------------------------------------------------------------ 2. orbits and embedding

@dataclass
class Orbit:
    phi: np.ndarray
    r: np.ndarray
    p: float
    e: float
    M: float
    L: float                                # specific angular momentum
    conserved: np.ndarray                   # the conserved combination along the orbit
    precession: float | None                # measured advance of the periapsis per orbit (rad)
    precession_weak_field: float            # 6 pi / p (leading order; the next term is a factor 1 + (18 + e^2) / (4 p))
    drift: float                            # largest relative change of the conserved quantity

    @property
    def x(self) -> np.ndarray:
        return self.r * np.cos(self.phi)

    @property
    def y(self) -> np.ndarray:
        return self.r * np.sin(self.phi)


def orbit(p: float, e: float, n_orbits: int = 3, M: float = 1.0, steps_per_orbit: int = 1500) -> Orbit:
    """A bound equatorial orbit, started at periapsis, with semi-latus rectum p (in M) and eccentricity e.

    Darwin's parametrisation: r = p M / (1 + e cos chi), L^2 = M^2 p^2 / (p - 3 - e^2). Bound and stable only for
    p > 6 + 2e (the separatrix); inside it the body plunges."""
    if not 0.0 <= e < 1.0:
        raise ShowcaseError("The eccentricity must be in [0, 1) for a bound orbit.")
    if p <= 6.0 + 2.0 * e + 0.05:
        raise ShowcaseError(f"p must exceed 6 + 2e = {6 + 2 * e:g} for a stable bound orbit (below that the body plunges in).")
    if not 1 <= n_orbits <= 20:
        raise ShowcaseError("Between 1 and 20 orbits.")
    L2 = (M * p) ** 2 / (p - 3.0 - e * e)
    L = math.sqrt(L2)
    u = (1.0 + e) / (p * M)
    w = 0.0
    c0 = M / L2
    h = 2.0 * math.pi / steps_per_orbit
    n = n_orbits * steps_per_orbit + steps_per_orbit // 2
    us, ws = np.empty(n + 1), np.empty(n + 1)
    us[0], ws[0] = u, w

    def f(uu, ww):
        return ww, -uu + c0 + 3.0 * M * uu * uu

    for i in range(n):
        k1 = f(u, w)
        k2 = f(u + 0.5 * h * k1[0], w + 0.5 * h * k1[1])
        k3 = f(u + 0.5 * h * k2[0], w + 0.5 * h * k2[1])
        k4 = f(u + h * k3[0], w + h * k3[1])
        u += h / 6.0 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        w += h / 6.0 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        us[i + 1], ws[i + 1] = u, w
    phi = np.arange(n + 1) * h
    conserved = ws ** 2 + us ** 2 - 2.0 * M * us ** 3 - 2.0 * M * us / L2
    drift = float(np.max(np.abs(conserved - conserved[0])) / max(abs(conserved[0]), abs(us[0] ** 2)))
    # periapsis = a maximum of u: w goes from + to -
    cross = np.flatnonzero((ws[:-1] > 0) & (ws[1:] <= 0))
    peri = [phi[i] + h * ws[i] / (ws[i] - ws[i + 1]) for i in cross]
    advance = float(np.mean(np.diff(peri)) - 2.0 * math.pi) if len(peri) >= 2 else None
    return Orbit(phi, 1.0 / us, p, e, M, L, conserved, advance, 6.0 * math.pi / p, drift)


def flamm_z(r: np.ndarray | float, M: float = 1.0):
    """Height of Flamm's paraboloid: the embedding of the equatorial plane's spatial geometry, z = 2 sqrt(2M (r - 2M))."""
    return 2.0 * np.sqrt(2.0 * M * np.maximum(np.asarray(r, dtype=float) - 2.0 * M, 0.0))


def flamm_surface(r_max: float, M: float = 1.0, n_r: int = 40, n_phi: int = 72):
    r = np.linspace(2.0 * M, r_max, n_r)
    ph = np.linspace(0, 2 * math.pi, n_phi)
    R, P = np.meshgrid(r, ph)
    return R * np.cos(P), R * np.sin(P), flamm_z(R, M)


# ------------------------------------------------------------------------------------------ 3. the critical point

@dataclass
class CriticalMovie:
    tc: float
    temperatures: list[float]
    runs: list                              # sm_lattice.SingleRun, one per temperature
    labels: list[str]


def critical_movie(L: int = 64, sweeps: int = 240, every: int = 6, seed: int = 7,
                   factors: tuple[float, float, float] = (0.85, 1.0, 1.25)) -> CriticalMovie:
    """Three Ising runs started from the same random state: below, at and above T_c."""
    if not 16 <= L <= 256:
        raise ShowcaseError("The lattice side must be between 16 and 256.")
    if sweeps < 20 or every < 1 or sweeps // every > 400:
        raise ShowcaseError("Need at least 20 sweeps and at most 400 frames.")
    tc = onsager_tc()
    model = lat.LatticeModel("ising", L)
    temps = [f * tc for f in factors]
    runs = [lat.run_single(model, T, sweeps, every, seed=seed) for T in temps]
    labels = [f"T = {f:.2f} T_c  (below)" if f < 1 else f"T = T_c  (critical)" if f == 1 else f"T = {f:.2f} T_c  (above)"
              for f in factors]
    return CriticalMovie(tc, temps, runs, labels)
