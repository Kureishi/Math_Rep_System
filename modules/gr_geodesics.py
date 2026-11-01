"""
Geodesics of any metric (modules/gr_metrics.GRMetric) -- Streamlit-free.

    d x^a / d lambda = u^a,        d u^a / d lambda = - Gamma^a_bc u^b u^c

The Christoffel symbols come from the same symbolic engine as the curvature tools (gr_curvature), lambdified
once into a numpy function; the integration is scipy's DOP853 (or RK45 / LSODA / Radau) at tight tolerances,
in CHUNKS so a progress hook (and with it the app's Stop button, see modules/progress.py) can act between them.

THE BUILT-IN CORRECTNESS TEST. A geodesic is full of quantities that must not change, and the integrator
monitors every one it can identify, reporting the drift:
  * the norm g_ab u^a u^b (= -1 for a massive particle, 0 for light, +1 for a spacelike geodesic);
  * p_k = g_kb u^b for every coordinate x^k the metric does not depend on (a Killing vector): energy
    E = -p_t, angular momentum L = p_phi, comoving momenta in an expanding universe, ...;
  * L^2 = p_theta^2 + p_phi^2 / sin^2(theta) for a spherically symmetric metric (the total angular momentum);
  * the CARTER CONSTANT for Kerr, a hidden symmetry no coordinate reveals:
        Q = p_theta^2 + cos^2(theta) [ a^2 (mu^2 - E^2) + L^2 / sin^2(theta) ].
A wrong Christoffel symbol, a too-loose tolerance or an orbit too near a singularity all show up as drift in
these, which is a much stronger statement than "the curve looks like an orbit".

SCHWARZSCHILD ORBITS are also compared with the EXACT results of gr_potential (the elliptic-integral
periapsis advance, the light-deflection integral), so the integrator is checked against mathematics it shares
no code with.

INITIAL CONDITIONS. Either coordinate velocities (u^t is then fixed by the norm, choosing the future-directed
root), or, for a stationary axisymmetric metric in (t, r, theta, phi) order, the conserved E and L at a radius
in the equatorial plane (u^r then follows from the norm, and an error says so if (E, L) is not allowed there).
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from modules.gr_curvature import _christoffel, invert_metric
from modules.gr_metrics import GR_TIMEOUT, GRInputError, GRMetric, parameter_symbol
from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList
from modules.timeout_utils import run_with_timeout

KAPPA = {"timelike": -1.0, "null": 0.0, "spacelike": 1.0}
METHODS = ("DOP853", "RK45", "LSODA", "Radau")
_SPHERICAL = {"schwarzschild", "rn", "static", "minkowski", "frw"}
_CACHE: dict[tuple, "GeodesicSystem"] = {}


# ------------------------------------------------------------------------------------------ the system

@dataclass
class GeodesicSystem:
    metric: GRMetric
    params: dict[str, float]
    _gamma: Callable
    _g: Callable
    killing: list[int]                       # indices of coordinates the metric does not depend on

    def values(self) -> list[float]:
        return [float(self.params[k]) for k in sorted(self.metric.params)]

    def gamma(self, x: np.ndarray) -> np.ndarray:
        n = self.metric.dim
        with np.errstate(all="ignore"):
            return np.array(self._gamma(*x, *self.values()), dtype=float).reshape(n, n, n)

    def g(self, x: np.ndarray) -> np.ndarray:
        n = self.metric.dim
        with np.errstate(all="ignore"):
            return np.array(self._g(*x, *self.values()), dtype=float).reshape(n, n)

    def g_batch(self, X: np.ndarray) -> np.ndarray:
        """g at each row of X, shape (N, n, n)."""
        n = self.metric.dim
        cols = [X[:, i] for i in range(n)]
        entries = self._g(*cols, *self.values())
        ones = np.ones(len(X))
        return np.array([[np.asarray(entries[i][j], dtype=float) * ones for j in range(n)] for i in range(n)]).transpose(2, 0, 1)

    def rhs(self, _lam: float, y: np.ndarray) -> np.ndarray:
        n = self.metric.dim
        x, u = y[:n], y[n:]
        return np.concatenate([u, -np.einsum("kij,i,j->k", self.gamma(x), u, u)])


def build_system(metric: GRMetric, params: dict[str, float] | None = None, timeout: float = GR_TIMEOUT) -> GeodesicSystem:
    """Christoffel symbols of `metric` lambdified for numerical integration (cached per metric and parameters)."""
    merged = {**metric.params, **{k: float(v) for k, v in (params or {}).items() if k in metric.params}}
    key = (metric.key, str(metric.g), tuple(sorted(merged.items())))
    if key in _CACHE:
        return _CACHE[key]
    n = metric.dim
    coords = metric.coords

    def build():
        ginv = invert_metric(metric.g)
        return _christoffel(metric.g, ginv, n, lambda e, i: sp.diff(e, coords[i]))

    gam = run_with_timeout(build, timeout=timeout, label="Christoffel symbols")
    syms = list(coords) + [parameter_symbol(k) for k in sorted(metric.params)]
    gfn = sp.lambdify(syms, [[gam[k][i][j] for j in range(n)] for k in range(n) for i in range(n)], modules="numpy", cse=True)
    metfn = sp.lambdify(syms, metric.g.tolist(), modules="numpy", cse=True)
    killing = [i for i, c in enumerate(coords) if all(sp.diff(metric.g[a, b], c) == 0 for a in range(n) for b in range(n))]
    system = GeodesicSystem(metric, merged, gfn, metfn, killing)
    if len(_CACHE) > 24:
        _CACHE.clear()
    _CACHE[key] = system
    return system


# ------------------------------------------------------------------------------------------ initial data

def normalise_velocity(system: GeodesicSystem, x0: list[float], spatial_u: list[float], kind: str = "timelike") -> list[float]:
    """The full 4-velocity u = (u^0, *spatial_u) with g_ab u^a u^b = kappa, u^0 > 0 (future-directed)."""
    if kind not in KAPPA:
        raise GRInputError(f"kind must be one of {', '.join(KAPPA)}.")
    n = system.metric.dim
    if len(x0) != n or len(spatial_u) != n - 1:
        raise GRInputError(f"Need {n} position values and {n - 1} spatial velocity components.")
    g = system.g(np.array(x0, dtype=float))
    if not np.all(np.isfinite(g)):
        raise GRInputError("The metric is not finite at that position (a singularity or a coordinate singularity).")
    us = np.array(spatial_u, dtype=float)
    a = g[0, 0]
    b = float(g[0, 1:] @ us)
    c = float(us @ g[1:, 1:] @ us) - KAPPA[kind]
    if abs(a) < 1e-14:
        raise GRInputError("g_tt = 0 here (a horizon or ergosurface in these coordinates): give a different position.")
    disc = b * b - a * c
    if disc < 0:
        raise GRInputError("No velocity with that norm exists for these spatial components here: a spacelike geodesic "
                           "needs a larger spatial part (timelike and null ones always exist outside horizons).")
    roots = [(-b + s * math.sqrt(disc)) / a for s in (1, -1)]
    pos = [r for r in roots if r > 0]
    if not pos:
        raise GRInputError("No future-directed velocity with those spatial components exists here "
                           "(inside an ergoregion or horizon the time coordinate may run backwards).")
    return [min(pos) if a < 0 else max(pos)] + list(us)


def initial_from_conserved(system: GeodesicSystem, r0: float, E: float, L: float, kind: str = "timelike",
                           inward: bool = True, phi0: float = 0.0) -> tuple[list[float], list[float]]:
    """(x0, u0) in the equatorial plane at radius r0 with u_t = -E and u_phi = L (coordinates ordered
    t, r, theta, phi), radial direction inward or outward. Raises if (E, L) is not allowed at r0."""
    names = system.metric.coord_names
    if names[:4] != ["t", "r", "theta", "phi"] or system.metric.dim != 4:
        raise GRInputError("Starting from (E, L) needs coordinates in the order t, r, theta, phi.")
    x0 = [0.0, float(r0), math.pi / 2, float(phi0)]
    g = system.g(np.array(x0))
    if abs(g[0, 1]) > 1e-12 or abs(g[1, 2]) > 1e-12 or abs(g[1, 3]) > 1e-12 or abs(g[0, 2]) > 1e-12 or abs(g[2, 3]) > 1e-12:
        raise GRInputError("This metric has cross terms in r or theta; use coordinate velocities instead.")
    block = g[np.ix_([0, 3], [0, 3])]
    try:
        ut, uphi = np.linalg.solve(block, np.array([-E, L]))
    except np.linalg.LinAlgError:
        raise GRInputError("The (t, phi) block of the metric is singular at that radius.") from None
    rest = KAPPA[kind] - float(np.array([ut, uphi]) @ block @ np.array([ut, uphi]))
    ur2 = rest / g[1, 1]
    if ur2 < -1e-14:
        raise GRInputError(f"(E, L) = ({E:g}, {L:g}) is not allowed at r = {r0:g}: that radius lies in a forbidden "
                           "region (below the effective potential).")
    ur = math.sqrt(max(ur2, 0.0)) * (-1 if inward else 1)
    return x0, [float(ut), ur, 0.0, float(uphi)]


# ------------------------------------------------------------------------------------------ results

@dataclass
class ConservedQuantity:
    name: str
    values: np.ndarray
    drift_abs: float
    drift_rel: float
    passed: bool
    note: str = ""


@dataclass
class GeodesicResult:
    system: GeodesicSystem
    kind: str
    lam: np.ndarray
    x: np.ndarray                                   # (N, n)
    u: np.ndarray                                   # (N, n)
    reason: str                                     # why the integration stopped
    conserved: list[ConservedQuantity] = field(default_factory=list)
    checks: CheckList = field(default_factory=CheckList)
    periapses: list[tuple[float, np.ndarray]] = field(default_factory=list)     # (lambda, state) at r minima
    apoapses: list[tuple[float, np.ndarray]] = field(default_factory=list)
    n_rhs: int = 0
    method: str = "DOP853"
    rtol: float = 1e-10

    def column(self, name: str) -> np.ndarray:
        return self.x[:, self.system.metric.coord_names.index(name)]


def default_r_stop(metric: GRMetric, params: dict[str, float] | None = None) -> float | None:
    """A radius just outside the outer horizon (where Schwarzschild-like coordinate time diverges)."""
    merged = {**metric.params, **(params or {})}
    try:
        if metric.family == "kerr":
            a, M = abs(merged["a"]), merged["M"]
            return 1.002 * (M + math.sqrt(max(M * M - a * a, 0.0)))
        if metric.f_of_r is not None:
            from modules.gr_potential import static_from_metric, static_markers
            hs = [m.r for m in static_markers(static_from_metric(metric), merged) if m.kind == "horizon"]
            return 1.002 * max(hs) if hs else None
    except Exception:  # noqa: BLE001
        return None
    return None


def _event(fn: Callable, terminal: bool = False, direction: float = 0.0) -> Callable:
    """A solve_ivp event function (it reads these two attributes off the function object)."""
    setattr(fn, "terminal", terminal)
    setattr(fn, "direction", direction)
    return fn


def integrate_geodesic(system: GeodesicSystem, x0: list[float], u0: list[float], lam_max: float, *,
                       kind: str = "timelike", method: str = "DOP853", rtol: float = 1e-10, atol: float = 1e-12,
                       n_samples: int = 4000, r_stop: float | None = None, r_return: float | None = None,
                       chunks: int = 24, check_tol: float = 1e-6, progress: ProgressFn | None = None) -> GeodesicResult:
    """Integrate from (x0, u0) for an affine parameter range lam_max (proper time for massive particles).

    r_stop   end the run if r falls to this radius (just outside a horizon, say);
    r_return end the run when r climbs back out to this radius (a flyby: set it to the starting radius);
    the run also stops, with the reason recorded, at a singularity (non-finite state) or step-size failure."""
    if method not in METHODS:
        raise GRInputError(f"method must be one of {', '.join(METHODS)}.")
    if lam_max <= 0:
        raise GRInputError("The parameter range must be positive.")
    n = system.metric.dim
    names = system.metric.coord_names
    ir = names.index("r") if "r" in names else None
    n_samples = int(min(max(n_samples, 50), 200_000))
    chunks = int(min(max(chunks, 1), 200))
    y = np.array(list(x0) + list(u0), dtype=float)

    events: list[Callable] = []
    if ir is not None:
        iu = n + ir
        events += [_event(lambda _t, s: s[iu], direction=1.0),          # u^r goes - to +  -> periapsis
                   _event(lambda _t, s: s[iu], direction=-1.0)]         # + to -           -> apoapsis
        if r_stop is not None:
            events.append(_event(lambda _t, s: s[ir] - r_stop, terminal=True, direction=-1.0))
        if r_return is not None:
            events.append(_event(lambda _t, s: s[ir] - r_return * (1 + 1e-9), terminal=True, direction=1.0))

    lam_all, y_all = [0.0], [y.copy()]
    peris: list[tuple[float, np.ndarray]] = []
    apos: list[tuple[float, np.ndarray]] = []
    reason = "reached the requested parameter range"
    n_rhs = 0
    edges = np.linspace(0.0, lam_max, chunks + 1)
    per_chunk = max(2, n_samples // chunks)
    for k in range(chunks):
        report(progress, f"Integrating (chunk {k + 1} of {chunks})", k / chunks)
        t_eval = np.linspace(edges[k], edges[k + 1], per_chunk + 1)[1:]
        try:
            sol = solve_ivp(system.rhs, (edges[k], edges[k + 1]), y, method=method, rtol=rtol, atol=atol, t_eval=t_eval,
                            events=events or None)
        except (ValueError, FloatingPointError, ZeroDivisionError) as exc:
            reason = f"stopped: the integrator failed ({exc})"
            break
        n_rhs += int(sol.nfev)
        sy = np.asarray(sol.y)
        ok = bool(np.isfinite(sy).all()) if sy.size else True
        if len(sol.t) and sy.size and ok:
            lam_all += list(sol.t)
            y_all += list(sy.T)
        if events and sol.t_events:
            for (tt, yy) in zip(sol.t_events[0], sol.y_events[0]):
                peris.append((float(tt), yy.copy()))
            for (tt, yy) in zip(sol.t_events[1], sol.y_events[1]):
                apos.append((float(tt), yy.copy()))
        if sol.status == 1:
            hit = [i for i, te in enumerate(sol.t_events) if len(te)]
            if 2 in hit and r_stop is not None:
                reason = f"reached the stopping radius r = {r_stop:g}"
            else:
                reason = f"returned to r = {r_return:g}"
            lam_all.append(float(sol.t_events[hit[-1]][-1]))
            y_all.append(sol.y_events[hit[-1]][-1].copy())
            break
        if sol.status == -1 or not ok:
            reason = f"stopped: {sol.message}" if sol.status == -1 else "stopped: the state became non-finite (a singularity)"
            break
        y = sy[:, -1].copy()
    else:
        report(progress, "Integrating", 1.0)
    lam_arr = np.array(lam_all)
    y_arr = np.array(y_all)
    result = GeodesicResult(system, kind, lam_arr, y_arr[:, :n], y_arr[:, n:], reason, periapses=peris, apoapses=apos,
                            n_rhs=n_rhs, method=method, rtol=rtol)
    evaluate_conserved(result, check_tol)
    return result


# ------------------------------------------------------------------------------------------ conserved quantities

def covariant_momentum(system: GeodesicSystem, x: np.ndarray, u: np.ndarray) -> np.ndarray:
    """p_a = g_ab u^b along the path, shape (N, n)."""
    return np.einsum("nab,nb->na", system.g_batch(x), u)


def evaluate_conserved(result: GeodesicResult, check_tol: float = 1e-6) -> None:
    """Fills result.conserved and result.checks (see the module docstring for the list)."""
    system, metric = result.system, result.system.metric
    names = metric.coord_names
    p = covariant_momentum(system, result.x, result.u)
    g = system.g_batch(result.x)
    kappa = KAPPA[result.kind]
    quantities: list[tuple[str, np.ndarray, str]] = []
    norm = np.einsum("na,nab,nb->n", result.u, g, result.u)
    quantities.append((f"norm  g_ab u^a u^b  (should stay {kappa:g})", norm - kappa, "deviation from the required value"))
    for k in system.killing:
        nm = names[k]
        if nm == "t":
            quantities.append(("energy  E = -p_t", -p[:, k], "∂_t is Killing"))
        elif nm == "phi":
            quantities.append(("angular momentum  L = p_φ", p[:, k], "∂_φ is Killing"))
        else:
            quantities.append((f"momentum  p_{nm}", p[:, k], f"the metric does not depend on {nm}"))
    if metric.family in _SPHERICAL and "theta" in names and "phi" in names:
        it, ip = names.index("theta"), names.index("phi")
        s = np.sin(result.x[:, it])
        with np.errstate(divide="ignore", invalid="ignore"):
            quantities.append(("total angular momentum²  L² = p_θ² + p_φ²/sin²θ", p[:, it] ** 2 + p[:, ip] ** 2 / s ** 2,
                               "spherical symmetry"))
    if metric.family == "kerr" and "t" in names:
        a = abs(result.system.params["a"])
        it, ip, itt = names.index("theta"), names.index("phi"), names.index("t")
        E, L = -p[:, itt], p[:, ip]
        mu2 = -kappa
        s2 = np.sin(result.x[:, it]) ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            carter = p[:, it] ** 2 + np.cos(result.x[:, it]) ** 2 * (a * a * (mu2 - E ** 2) + L ** 2 / s2)
        quantities.append(("Carter constant  Q", carter, "the hidden symmetry of Kerr"))
    result.conserved = []
    result.checks = CheckList()
    for name, vals, note in quantities:
        finite = vals[np.isfinite(vals)]
        if finite.size == 0:
            continue
        ref = 0.0 if name.startswith("norm") else float(finite[0])
        drift = float(np.max(np.abs(finite - ref)))
        scale = max(1.0, abs(ref), float(np.max(np.abs(finite))))
        ok = drift <= check_tol * scale
        cq = ConservedQuantity(name, vals, drift, drift / scale, ok, note)
        result.conserved.append(cq)
        result.checks.add(name, ok, f"max drift {drift:.2e} (relative {drift / scale:.1e}; tolerance {check_tol:.0e})")
    result.checks.add("reached the requested range", result.reason.startswith("reached the requested"), result.reason)


# ------------------------------------------------------------------------------------------ analysis

def periapsis_advance(result: GeodesicResult) -> list[float]:
    """Angle swept between successive periapses minus 2 pi (radians), one value per completed radial period.
    Positive = the orbit's periapsis advances in the direction of motion (Mercury's 43 arcsec per century)."""
    names = result.system.metric.coord_names
    if "phi" not in names:
        return []
    ip = names.index("phi")
    phis = [float(s[ip]) for _, s in result.periapses]
    return [abs(b - a) - 2 * math.pi for a, b in zip(phis, phis[1:])]


def to_cartesian(metric: GRMetric, x: np.ndarray, params: dict[str, float] | None = None) -> np.ndarray:
    """Positions as (N, 3) points in a plottable space, following metric.embedding (see gr_metrics)."""
    names = metric.coord_names
    N = len(x)
    col = lambda nm: x[:, names.index(nm)]  # noqa: E731
    merged = {**metric.params, **(params or {})}
    out = np.zeros((N, 3))
    if metric.embedding in ("spherical", "comoving", "oblate") and {"r", "theta", "phi"} <= set(names):
        r, th, ph = col("r"), col("theta"), col("phi")
        rho = np.sqrt(r * r + merged.get("a", 0.0) ** 2) if metric.embedding == "oblate" else r
        out[:, 0], out[:, 1], out[:, 2] = rho * np.sin(th) * np.cos(ph), rho * np.sin(th) * np.sin(ph), r * np.cos(th)
    elif metric.embedding == "cartesian" and {"x", "y", "z"} <= set(names):
        out[:, 0], out[:, 1], out[:, 2] = col("x"), col("y"), col("z")
    else:
        spatial = [i for i in range(1, metric.dim)][:3]
        for j, i in enumerate(spatial):
            out[:, j] = x[:, i]
    return out
