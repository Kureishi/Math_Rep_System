"""
Dynamics and chaos: N-body orbits with symplectic integrators, the double pendulum, the Lorenz system, and Lyapunov exponents --
Streamlit-free, no LLM.

THE BUILT-IN CHECK IS CONSERVATION. A conservative system has quantities that must not change (energy, momentum, angular momentum, the
Laplace-Runge-Lenz vector), and a numerical method's honesty shows in how it breaks them:
  * a SYMPLECTIC integrator (leapfrog, Yoshida's 4th order) does not conserve energy exactly but keeps its error BOUNDED, oscillating
    around the true value for as long as you run it; a non-symplectic one (RK4, Euler) leaks energy steadily. The drift classifier of
    modules/cc_common.py tells the two apart, and the lab reports which one it saw and whether that matches what the method promises.
  * the observed ORDER of each integrator (the slope of the error against the step size on log-log axes) is measured against a
    reference solution from an adaptive 8th-order solver, and compared with the order the method claims.
  * the figure-eight three-body orbit must come back to where it started after one period; the Lagrange equilateral solution must
    stay equilateral; a Kepler orbit's period must obey Kepler's third law.
  * LYAPUNOV EXPONENTS are estimated for whole ENSEMBLES of nearby trajectories at once (the Benettin / QR method on batched tangent
    matrices) and come with a standard error across the ensemble. Two exact identities then check the estimate: the exponents of the
    Lorenz system sum to the trace of its Jacobian, -(sigma + 1 + beta); those of a Hamiltonian system sum to zero and come in +/- pairs.
  * the double pendulum's equations are DERIVED from its Lagrangian with SymPy, and the fast hand-vectorised right-hand side that the
    ensembles use is verified against that derivation at random states.

Ensembles are one batched Runge-Kutta step over every member, so they run on the GPU backend (modules/gpu.py) when there are enough of
them. Everything is deterministic given the seed, which is returned in every result (the project's seed convention).

Units: G = 1 for the N-body problems (the figure-eight uses the standard dimensionless initial conditions); the pendulum takes SI-like
numbers (masses, lengths, g); Lorenz is dimensionless.
"""
import math
from dataclasses import dataclass, field
from typing import Any
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from modules import gpu
from modules.cc_common import DriftReport, drift_report
from modules.quantum_common import CheckList
from modules.sm_common import resolve_seed


class DynamicsError(ValueError):
    """Bad input, with a reason a student can act on."""


# ------------------------------------------------------------------------------------------ N-body

INTEGRATORS: dict[str, dict[str, Any]] = {
    "leapfrog": dict(order=2, symplectic=True, label="Leapfrog (velocity Verlet), 2nd order, symplectic"),
    "yoshida4": dict(order=4, symplectic=True, label="Yoshida, 4th order, symplectic"),
    "rk4": dict(order=4, symplectic=False, label="Runge-Kutta 4, 4th order, NOT symplectic"),
    "euler": dict(order=1, symplectic=False, label="Explicit Euler, 1st order, NOT symplectic"),
}
_W1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_W0 = -(2.0 ** (1.0 / 3.0)) * _W1
_YOSHIDA = (_W1, _W0, _W1)

FIGURE_EIGHT_PERIOD = 6.32591398
MAX_BODIES = 2000


@dataclass
class NBodySystem:
    name: str
    masses: np.ndarray            # (N,)
    positions: np.ndarray         # (N, dim)
    velocities: np.ndarray        # (N, dim)
    softening: float = 0.0
    period: float | None = None   # a natural time scale (orbital period), when there is one
    note: str = ""
    seed: int | None = None


def _com_frame(m: np.ndarray, r: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mt = m.sum()
    return r - (m[:, None] * r).sum(0) / mt, v - (m[:, None] * v).sum(0) / mt


def preset_system(name: str, n: int = 40, seed: int | None = None, eccentricity: float = 0.6) -> NBodySystem:
    """The built-in systems: 'figure-eight', 'kepler', 'lagrange', 'pythagorean', 'cluster'."""
    if name == "figure-eight":
        x = np.array([[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0.0, 0.0]])
        v3 = np.array([-0.93240737, -0.86473146])
        v = np.array([-v3 / 2, -v3 / 2, v3])
        return NBodySystem("Figure-eight (Chenciner-Montgomery)", np.ones(3), x, v, 0.0, FIGURE_EIGHT_PERIOD,
                           "Three equal masses chasing each other along one figure-eight curve; periodic, and stable to small changes.")
    if name == "kepler":
        if not 0.0 <= eccentricity < 0.95:
            raise DynamicsError("Use an eccentricity from 0 to 0.95.")
        M, mm, a = 1.0, 1e-3, 1.0
        r0 = a * (1 + eccentricity)                                    # start at apocentre
        vrel = math.sqrt((M + mm) * (2 / r0 - 1 / a))
        m = np.array([M, mm])
        r, v = np.array([[0.0, 0.0], [r0, 0.0]]), np.array([[0.0, 0.0], [0.0, vrel]])
        r, v = _com_frame(m, r, v)
        return NBodySystem(f"Kepler orbit (e = {eccentricity:g})", m, r, v, 0.0, 2 * math.pi * math.sqrt(a ** 3 / (M + mm)),
                           "A light body around a heavy one: closed ellipse, Kepler's third law, conserved Laplace-Runge-Lenz vector.")
    if name == "lagrange":
        s = 1.0
        ang = np.array([90.0, 210.0, 330.0]) * math.pi / 180
        rad = s / math.sqrt(3.0)
        r = rad * np.stack([np.cos(ang), np.sin(ang)], axis=1)
        omega = math.sqrt(3.0 / s ** 3)                                 # omega^2 = 3 G m / s^3 for three unit masses
        v = omega * rad * np.stack([-np.sin(ang), np.cos(ang)], axis=1)
        return NBodySystem("Lagrange equilateral triangle", np.ones(3), r, v, 0.0, 2 * math.pi / omega,
                           "Three equal masses at the corners of an equilateral triangle that rotates rigidly: an exact solution.")
    if name == "pythagorean":
        m = np.array([3.0, 4.0, 5.0])
        r = np.array([[1.0, 3.0], [-2.0, -1.0], [1.0, -1.0]])
        r, v = _com_frame(m, r, np.zeros((3, 2)))
        return NBodySystem("Pythagorean three-body (3-4-5)", m, r, v, 0.0, None,
                           "Masses 3, 4, 5 released from rest at the corners of a 3-4-5 triangle: chaotic, with violent close encounters. A fixed step "
                           "cannot follow them, and the energy check says so.")
    if name == "cluster":
        if not 3 <= n <= MAX_BODIES:
            raise DynamicsError(f"Use between 3 and {MAX_BODIES} bodies.")
        sd = resolve_seed(seed)
        rng = np.random.default_rng(sd)
        r = rng.normal(size=(n, 3))
        m = np.full(n, 1.0 / n)
        v = rng.normal(size=(n, 3))
        eps = 0.1
        pot = potential_energy(m, r, eps)
        ke = 0.5 * float((m[:, None] * v * v).sum())
        v *= math.sqrt(0.5 * abs(pot) / ke)                              # virial ratio 2K/|U| = 1 (a cluster in equilibrium)
        r, v = _com_frame(m, r, v)
        return NBodySystem(f"Random cluster of {n} bodies", m, r, v, eps, None,
                           "Bodies drawn from a Gaussian cloud and scaled to virial equilibrium, with a small softening length.", seed=sd)
    raise DynamicsError("Unknown system.")


def accelerations(m: np.ndarray, r, eps: float, xp=np):
    """a_i = sum_j m_j (r_j - r_i) / (|r_j - r_i|^2 + eps^2)^(3/2), with G = 1; direct summation over all pairs (one N x N array per coordinate)."""
    n, dim = r.shape
    comps = [r[None, :, k] - r[:, None, k] for k in range(dim)]
    eye = xp.eye(n)
    d2 = sum(c * c for c in comps) + eps * eps + eye           # + eye keeps the diagonal finite; it is masked out below
    inv3 = d2 ** -1.5 * (1 - eye)                               # a body does not attract itself
    w = inv3 * m[None, :]
    return xp.stack([(c * w).sum(1) for c in comps], axis=1)


def potential_energy(m: np.ndarray, r: np.ndarray, eps: float = 0.0) -> float:
    d = r[None, :, :] - r[:, None, :]
    dist = np.sqrt((d * d).sum(-1) + eps * eps)
    mm = m[:, None] * m[None, :]
    iu = np.triu_indices(len(m), 1)
    return float(-(mm[iu] / dist[iu]).sum())


def kinetic_energy(m: np.ndarray, v: np.ndarray) -> float:
    return 0.5 * float((m[:, None] * v * v).sum())


def angular_momentum(m: np.ndarray, r: np.ndarray, v: np.ndarray) -> np.ndarray:
    if r.shape[1] == 2:
        return np.array([float((m * (r[:, 0] * v[:, 1] - r[:, 1] * v[:, 0])).sum())])
    return (m[:, None] * np.cross(r, v)).sum(0)


def _step(kind: str, m, r, v, dt: float, eps: float, acc):
    if kind == "euler":
        a = acc(r)
        return r + dt * v, v + dt * a
    if kind == "leapfrog":
        v = v + 0.5 * dt * acc(r)
        r = r + dt * v
        v = v + 0.5 * dt * acc(r)
        return r, v
    if kind == "yoshida4":
        for w in _YOSHIDA:
            v = v + 0.5 * w * dt * acc(r)
            r = r + w * dt * v
            v = v + 0.5 * w * dt * acc(r)
        return r, v
    if kind == "rk4":
        k1r, k1v = v, acc(r)
        k2r, k2v = v + 0.5 * dt * k1v, acc(r + 0.5 * dt * k1r)
        k3r, k3v = v + 0.5 * dt * k2v, acc(r + 0.5 * dt * k2r)
        k4r, k4v = v + dt * k3v, acc(r + dt * k3r)
        return r + dt * (k1r + 2 * k2r + 2 * k3r + k4r) / 6, v + dt * (k1v + 2 * k2v + 2 * k3v + k4v) / 6
    raise DynamicsError(f"Unknown integrator {kind!r}.")


@dataclass
class NBodyRun:
    system: NBodySystem
    integrator: str
    dt: float
    n_steps: int
    times: np.ndarray
    positions: np.ndarray            # (R, N, dim)
    energy: np.ndarray
    momentum: np.ndarray             # (R, dim) total linear momentum
    angular: np.ndarray              # (R, k)
    lrl: np.ndarray | None           # (R, 2) Laplace-Runge-Lenz vector for a two-body problem
    energy_drift: DriftReport
    return_error: float | None       # distance to the initial state after one period, when the run covers one
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"
    notes: list[str] = field(default_factory=list)


def run_nbody(system: NBodySystem, integrator: str = "leapfrog", dt: float = 1e-3, t_end: float | None = None, n_records: int = 400,
              tol: float = 1e-4, backend: gpu.Backend | None = None) -> NBodyRun:
    """Integrate with a fixed step and monitor the conserved quantities."""
    if integrator not in INTEGRATORS:
        raise DynamicsError(f"Choose one of: {', '.join(INTEGRATORS)}.")
    if not dt > 0:
        raise DynamicsError("The time step must be positive.")
    t_end = float(t_end if t_end is not None else (5 * system.period if system.period else 10.0))
    if t_end <= 0:
        raise DynamicsError("The end time must be positive.")
    n_steps = int(math.ceil(t_end / dt))
    if n_steps > 400_000:
        raise DynamicsError("That is more than 400,000 steps: raise the time step or shorten the run.")
    n = len(system.masses)
    backend = backend or (gpu.select(n, "nbody_particles"))
    xp = backend.xp
    records = np.unique(np.linspace(0, n_steps, min(n_records, n_steps + 1)).astype(int))
    m_h = system.masses
    eps = system.softening

    def run(xp_mod, to_host):
        m = xp_mod.asarray(m_h)
        r = xp_mod.asarray(system.positions.copy())
        v = xp_mod.asarray(system.velocities.copy())
        acc = lambda x: accelerations(m, x, eps, xp_mod)          # noqa: E731
        out_r, out_v = [to_host(r).copy()], [to_host(v).copy()]
        rec = set(records[1:].tolist())
        for step in range(1, n_steps + 1):
            r, v = _step(integrator, m, r, v, dt, eps, acc)
            if step in rec:
                out_r.append(to_host(r).copy())
                out_v.append(to_host(v).copy())
        return np.array(out_r), np.array(out_v)

    R, V = gpu.attempt(backend, lambda: run(xp, gpu.to_host), lambda: run(np, lambda a: a))
    used = gpu.last_used() or gpu.CPU.name
    t = records * dt
    energy = np.array([kinetic_energy(m_h, V[i]) + potential_energy(m_h, R[i], eps) for i in range(len(t))])
    mom = np.array([(m_h[:, None] * V[i]).sum(0) for i in range(len(t))])
    ang = np.array([angular_momentum(m_h, R[i], V[i]) for i in range(len(t))])
    lrl = None
    if n == 2:
        rel_r, rel_v = R[:, 1] - R[:, 0], V[:, 1] - V[:, 0]
        mu = float(m_h.sum())
        if rel_r.shape[1] == 2:
            h = rel_r[:, 0] * rel_v[:, 1] - rel_r[:, 1] * rel_v[:, 0]
            lrl = np.stack([rel_v[:, 1] * h - mu * rel_r[:, 0] / np.linalg.norm(rel_r, axis=1),
                            -rel_v[:, 0] * h - mu * rel_r[:, 1] / np.linalg.norm(rel_r, axis=1)], axis=1)
    drift = drift_report("energy", t, energy, scale=max(abs(energy[0]), 1e-300))
    spec = INTEGRATORS[integrator]
    ck = CheckList()
    notes: list[str] = []
    ck.add("Energy is conserved to within the tolerance", drift.max_rel <= tol, f"largest relative change {drift.max_rel:.2e} (tolerance {tol:g}); {drift.kind}")
    if spec["symplectic"]:
        ck.add("The energy error stays bounded (the symplectic signature)", not drift.secular,
               "it oscillates instead of growing" if not drift.secular else f"it grows steadily (net change {drift.secular_change:+.2e}): the step is too large for this orbit")
    else:
        notes.append(f"{integrator.upper()} is not symplectic, so it is not expected to keep the energy error bounded: the drift was classed as '{drift.kind}'. "
                     "Switch to leapfrog or Yoshida to see the difference.")
    p_scale = max(float(np.max(np.abs(m_h[:, None] * system.velocities))), 1e-12)
    p_dev = float(np.max(np.abs(mom - mom[0])))
    ck.add("Total momentum is conserved", p_dev <= 1e-9 * max(1.0, p_scale) * max(1, n_steps / 1000) or p_dev <= 1e-8, f"largest change {p_dev:.1e}")
    l_scale = max(float(np.max(np.abs(ang))), float((m_h * np.linalg.norm(system.positions, axis=1) * np.linalg.norm(system.velocities, axis=1)).sum()), 1e-12)
    l_dev = float(np.max(np.abs(ang - ang[0])))
    if not (integrator == "euler"):
        ck.add("Angular momentum is conserved", l_dev <= max(tol, 1e-6) * l_scale, f"largest relative change {l_dev / l_scale:.1e}")
    if lrl is not None:
        scale = max(float(np.max(np.linalg.norm(lrl, axis=1))), 1e-12)
        dev = float(np.max(np.linalg.norm(lrl - lrl[0], axis=1)))
        ck.add("The Laplace-Runge-Lenz vector is conserved (the orbit does not precess)", dev <= max(tol, 1e-5) * max(scale, 1e-3) * 10 or dev < 1e-6,
               f"largest change {dev:.1e} against its size {scale:.2f}")
    return_error = None
    if system.period is not None and t[-1] >= system.period:
        k = int(round(t[-1] / system.period))
        # the nearest record to k whole periods
        i = int(np.argmin(np.abs(t - k * system.period)))
        if abs(t[i] - k * system.period) < 2 * max(dt, t[1] - t[0]):
            if system.name.startswith("Figure-eight"):
                return_error = float(np.max(np.linalg.norm(R[i] - R[0], axis=1)))
                ck.add("The orbit returns to its starting point after each period", return_error < 1e-3 * max(1.0, 1.0),
                       f"after {k} period(s) (T = {system.period:.8g}) the largest distance from the start is {return_error:.1e}")
            elif system.name.startswith("Lagrange"):
                sides = [np.linalg.norm(R[:, a] - R[:, b], axis=1) for a, b in ((0, 1), (1, 2), (0, 2))]
                spread = float(max(np.max(np.abs(s - 1.0)) for s in sides))
                return_error = spread
                ck.add("The triangle stays equilateral", spread < 1e-3, f"every side stays within {spread:.1e} of 1 over the run")
            elif n == 2:
                rel = R[:, 1] - R[:, 0]
                return_error = float(np.linalg.norm(rel[i] - rel[0]) / np.linalg.norm(rel[0]))
                ck.add("The period obeys Kepler's third law", return_error < 1e-2, f"after {k} predicted period(s) the relative distance has returned to within {return_error:.1e} of the start")
    elif system.name.startswith("Lagrange"):
        sides = [np.linalg.norm(R[:, a] - R[:, b], axis=1) for a, b in ((0, 1), (1, 2), (0, 2))]
        spread = float(max(np.max(np.abs(s - 1.0)) for s in sides))
        ck.add("The triangle stays equilateral", spread < 1e-3, f"every side stays within {spread:.1e} of 1 over the run")
    return NBodyRun(system, integrator, dt, n_steps, t, R, energy, mom, ang, lrl, drift, return_error, ck, used, notes)


# ------------------------------------------------------------------------------------------ integrator order study

@dataclass
class OrderStudy:
    integrator: str
    dts: np.ndarray
    position_error: np.ndarray
    energy_error: np.ndarray
    observed_order: float
    nominal_order: int


def _reference_kepler(system: NBodySystem, t_end: float) -> np.ndarray:
    m, eps = system.masses, system.softening
    n, dim = system.positions.shape

    def f(_t, y):
        r, v = y[: n * dim].reshape(n, dim), y[n * dim:].reshape(n, dim)
        return np.concatenate([v.ravel(), accelerations(m, r, eps).ravel()])
    sol = solve_ivp(f, (0, t_end), np.concatenate([system.positions.ravel(), system.velocities.ravel()]), method="DOP853", rtol=1e-13, atol=1e-14)
    return sol.y[: n * dim, -1].reshape(n, dim)


def order_study(system: NBodySystem, integrators: tuple[str, ...] = ("euler", "leapfrog", "yoshida4", "rk4"), dts: tuple[float, ...] = (0.04, 0.02, 0.01, 0.005),
                orbits: float = 1.0) -> tuple[list[OrderStudy], CheckList]:
    """Error of the final position against an 8th-order adaptive reference, for each integrator at each step size; the slope on log-log axes is
    the observed order, compared with the nominal one."""
    if system.period is None:
        raise DynamicsError("The order study needs a periodic system (use the Kepler orbit).")
    t_end = orbits * system.period
    ref = _reference_kepler(system, t_end)
    ck = CheckList()
    out = []
    for kind in integrators:
        pos_err, en_err = [], []
        for dt in dts:
            steps = int(round(t_end / dt))
            h = t_end / steps
            m, eps = system.masses, system.softening
            acc = lambda x: accelerations(m, x, eps)               # noqa: E731
            r, v = system.positions.copy(), system.velocities.copy()
            e0 = kinetic_energy(m, v) + potential_energy(m, r, eps)
            emax = 0.0
            for s in range(steps):
                r, v = _step(kind, m, r, v, h, eps, acc)
                if s % 10 == 0:
                    emax = max(emax, abs(kinetic_energy(m, v) + potential_energy(m, r, eps) - e0))
            pos_err.append(float(np.max(np.linalg.norm(r - ref, axis=1))))
            en_err.append(max(emax / abs(e0), 1e-17))
        slope = float(np.polyfit(np.log(dts), np.log(pos_err), 1)[0])
        nominal = int(INTEGRATORS[kind]["order"])
        out.append(OrderStudy(kind, np.array(dts), np.array(pos_err), np.array(en_err), slope, nominal))
        ck.add(f"{kind}: the error shrinks at order {nominal}", abs(slope - nominal) < 0.5,
               f"observed slope {slope:.2f} on log-log axes (nominal {nominal}); final-position error {pos_err[0]:.1e} at dt = {dts[0]:g} down to {pos_err[-1]:.1e} at dt = {dts[-1]:g}")
    return out, ck


# ------------------------------------------------------------------------------------------ double pendulum

@dataclass(frozen=True)
class PendulumParams:
    m1: float = 1.0
    m2: float = 1.0
    l1: float = 1.0
    l2: float = 1.0
    g: float = 9.81

    def validate(self) -> None:
        for name in ("m1", "m2", "l1", "l2", "g"):
            if not getattr(self, name) > 0:
                raise DynamicsError(f"{name} must be positive.")


def pendulum_rhs(state, p: PendulumParams, xp=np):
    """d/dt of (theta1, theta2, omega1, omega2), hand-vectorised: `state` has shape (..., 4)."""
    t1, t2, w1, w2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    d = t1 - t2
    sd, cd = xp.sin(d), xp.cos(d)
    den = 2 * p.m1 + p.m2 - p.m2 * xp.cos(2 * d)
    a1 = (-p.g * (2 * p.m1 + p.m2) * xp.sin(t1) - p.m2 * p.g * xp.sin(t1 - 2 * t2) - 2 * sd * p.m2 * (w2 ** 2 * p.l2 + w1 ** 2 * p.l1 * cd)) / (p.l1 * den)
    a2 = (2 * sd * (w1 ** 2 * p.l1 * (p.m1 + p.m2) + p.g * (p.m1 + p.m2) * xp.cos(t1) + w2 ** 2 * p.l2 * p.m2 * cd)) / (p.l2 * den)
    return xp.stack([w1, w2, a1, a2], axis=-1)


def pendulum_energy(state, p: PendulumParams, xp=np):
    t1, t2, w1, w2 = state[..., 0], state[..., 1], state[..., 2], state[..., 3]
    kin = 0.5 * (p.m1 + p.m2) * p.l1 ** 2 * w1 ** 2 + 0.5 * p.m2 * p.l2 ** 2 * w2 ** 2 + p.m2 * p.l1 * p.l2 * w1 * w2 * xp.cos(t1 - t2)
    pot = -(p.m1 + p.m2) * p.g * p.l1 * xp.cos(t1) - p.m2 * p.g * p.l2 * xp.cos(t2)
    return kin + pot


@lru_cache(maxsize=1)
def pendulum_sympy():
    """The double pendulum's equations of motion DERIVED from its Lagrangian: (rhs, jacobian) as SymPy objects plus the symbols."""
    t = sp.Symbol("t")
    m1, m2, l1, l2, g = sp.symbols("m1 m2 l1 l2 g", positive=True)
    th1, th2 = sp.Function("th1")(t), sp.Function("th2")(t)
    x1, y1 = l1 * sp.sin(th1), -l1 * sp.cos(th1)
    x2, y2 = x1 + l2 * sp.sin(th2), y1 - l2 * sp.cos(th2)
    T = sp.Rational(1, 2) * m1 * (x1.diff(t) ** 2 + y1.diff(t) ** 2) + sp.Rational(1, 2) * m2 * (x2.diff(t) ** 2 + y2.diff(t) ** 2)
    V = m1 * g * y1 + m2 * g * y2
    Lag = T - V
    eqs = [sp.diff(Lag.diff(q.diff(t)), t) - Lag.diff(q) for q in (th1, th2)]
    a1, a2 = sp.symbols("a1 a2")
    eqs = [e.subs({th1.diff(t, 2): a1, th2.diff(t, 2): a2}) for e in eqs]
    sol = sp.solve(eqs, [a1, a2], dict=True)[0]
    q1, q2, w1, w2 = sp.symbols("q1 q2 w1 w2")
    rep = {th1.diff(t): w1, th2.diff(t): w2}
    rep2 = {th1: q1, th2: q2}
    acc1 = sp.simplify(sol[a1].subs(rep).subs(rep2))      # simplified once (about 3 s), which keeps the Jacobian compact for the ensembles
    acc2 = sp.simplify(sol[a2].subs(rep).subs(rep2))
    state = sp.Matrix([q1, q2, w1, w2])
    rhs = sp.Matrix([w1, w2, acc1, acc2])
    return rhs, rhs.jacobian(state), (q1, q2, w1, w2), (m1, m2, l1, l2, g)


def pendulum_is_derived() -> bool:
    """Has the SymPy derivation already run in this process (so the next use is instant)? The first use takes about 3.5 s, which the page explains with a spinner."""
    return pendulum_sympy.cache_info().currsize > 0


def verify_pendulum_rhs(p: PendulumParams, samples: int = 25, seed: int = 0) -> tuple[float, float]:
    """(largest relative difference of the RHS, of the Jacobian is not computed here) between the hand-vectorised right-hand side and the
    one derived from the Lagrangian with SymPy, at random states. Returns (rhs_error, 0.0)."""
    rhs, _, st_syms, par_syms = pendulum_sympy()
    f = sp.lambdify([*st_syms, *par_syms], rhs, "numpy")
    rng = np.random.default_rng(seed)
    worst = 0.0
    for _ in range(samples):
        s = rng.uniform(-3, 3, size=4)
        ref = np.array(f(*s, p.m1, p.m2, p.l1, p.l2, p.g), dtype=float).ravel()
        mine = pendulum_rhs(s, p)
        worst = max(worst, float(np.max(np.abs(ref - mine) / np.maximum(1.0, np.abs(ref)))))
    return worst, 0.0


@dataclass
class PendulumRun:
    params: PendulumParams
    times: np.ndarray
    states: np.ndarray            # (R, 4)
    energy: np.ndarray
    energy_drift: DriftReport
    checks: CheckList = field(default_factory=CheckList)


def run_pendulum(p: PendulumParams, state0: tuple[float, float, float, float], t_end: float = 30.0, n_points: int = 1500, tol: float = 1e-7) -> PendulumRun:
    """One trajectory with an adaptive 8th-order integrator at tight tolerance; the energy drift is the check."""
    p.validate()
    if not 0 < t_end <= 2000:
        raise DynamicsError("Use a duration between 0 and 2000.")
    y0 = np.array(state0, dtype=float)
    ts = np.linspace(0, t_end, int(n_points))
    sol = solve_ivp(lambda _t, y: pendulum_rhs(y, p), (0, t_end), y0, method="DOP853", t_eval=ts, rtol=1e-12, atol=1e-12)
    states = sol.y.T
    en = pendulum_energy(states, p)
    drift = drift_report("energy", ts, en)
    ck = CheckList()
    ck.add("Energy is conserved", drift.max_rel <= tol, f"largest relative change {drift.max_rel:.2e} (tolerance {tol:g}); {drift.kind}")
    err, _ = verify_pendulum_rhs(p)
    ck.add("The fast equations match the ones derived from the Lagrangian", err < 1e-10, f"largest difference at 25 random states: {err:.1e} (SymPy derivation of the Euler-Lagrange equations)")
    return PendulumRun(p, ts, states, en, drift, ck)


def pendulum_positions(p: PendulumParams, states: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x1, y1 = p.l1 * np.sin(states[:, 0]), -p.l1 * np.cos(states[:, 0])
    return x1, y1, x1 + p.l2 * np.sin(states[:, 1]), y1 - p.l2 * np.cos(states[:, 1])


# ------------------------------------------------------------------------------------------ Lorenz

@dataclass(frozen=True)
class LorenzParams:
    sigma: float = 10.0
    rho: float = 28.0
    beta: float = 8.0 / 3.0


LORENZ_LAMBDA1_LITERATURE = 0.9056            # Sprott / Wolf et al. for (10, 28, 8/3)
LORENZ_KAPLAN_YORKE_LITERATURE = 2.0627


def lorenz_rhs(s, p: LorenzParams, xp=np):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return xp.stack([p.sigma * (y - x), x * (p.rho - z) - y, x * y - p.beta * z], axis=-1)


def lorenz_jacobian(s, p: LorenzParams, xp=np):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    zero = xp.zeros_like(x)
    rows = [[zero - p.sigma, zero + p.sigma, zero], [p.rho - z, zero - 1.0, -x], [y, x, zero - p.beta]]
    return xp.stack([xp.stack(r, axis=-1) for r in rows], axis=-2)


def run_lorenz(p: LorenzParams, state0: tuple[float, float, float] = (1.0, 1.0, 1.0), t_end: float = 40.0, n_points: int = 6000) -> tuple[np.ndarray, np.ndarray]:
    ts = np.linspace(0, t_end, int(n_points))
    sol = solve_ivp(lambda _t, y: lorenz_rhs(y, p), (0, t_end), np.array(state0, float), method="DOP853", t_eval=ts, rtol=1e-11, atol=1e-11)
    return ts, sol.y.T


# ------------------------------------------------------------------------------------------ batched integration and Lyapunov exponents

@lru_cache(maxsize=4)
def _pendulum_jacobian_fn(module: str):
    _, jac, st_syms, par_syms = pendulum_sympy()
    return sp.lambdify([*st_syms, *par_syms], [list(jac.row(i)) for i in range(4)], module, cse=True)


def pendulum_jacobian(s, p: PendulumParams, xp=np, module: str = "numpy"):
    f = _pendulum_jacobian_fn(module)
    rows = f(s[..., 0], s[..., 1], s[..., 2], s[..., 3], p.m1, p.m2, p.l1, p.l2, p.g)
    zero = xp.zeros_like(s[..., 0])
    return xp.stack([xp.stack([xp.asarray(e) + zero for e in row], axis=-1) for row in rows], axis=-2)


def rk4_batch(f, y, dt: float):
    k1 = f(y)
    k2 = f(y + 0.5 * dt * k1)
    k3 = f(y + 0.5 * dt * k2)
    k4 = f(y + dt * k3)
    return y + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6


@dataclass
class LyapunovResult:
    system: str
    exponents: np.ndarray            # mean over the ensemble
    errors: np.ndarray               # standard error across the ensemble
    members: int
    t_run: float
    dt: float
    seed: int
    kaplan_yorke: float | None
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"
    notes: list[str] = field(default_factory=list)


def _gram_schmidt(Q, xp):
    """Modified Gram-Schmidt on the columns of a batch of matrices; returns (Q, log of the column norms)."""
    n = Q.shape[-1]
    cols: list[Any] = []
    logs: list[Any] = []
    for k in range(n):
        v = Q[..., k]
        for q in cols:
            v = v - (q * v).sum(-1, keepdims=True) * q
        r = xp.sqrt((v * v).sum(-1))
        logs.append(xp.log(r))
        cols.append(v / r[..., None])
    return xp.stack(cols, axis=-1), xp.stack(logs, axis=-1)


def kaplan_yorke(exps: np.ndarray) -> float | None:
    """The Kaplan-Yorke (Lyapunov) dimension: j + (lambda_1 + ... + lambda_j) / |lambda_{j+1}|."""
    e = np.sort(exps)[::-1]
    csum = np.cumsum(e)
    if csum[0] < 0:
        return 0.0
    j = int(np.max(np.nonzero(csum >= 0)[0]))
    if j >= len(e) - 1:
        return float(len(e))
    return float(j + 1 + csum[j] / abs(e[j + 1]))


def lyapunov_spectrum(system: str, params: LorenzParams | PendulumParams, state0: np.ndarray, members: int = 128, t_transient: float = 10.0, t_run: float = 20.0,
                      dt: float = 0.01, renorm_steps: int = 10, t_frame: float = 5.0, spread: float = 1e-3, seed: int | None = None, backend: gpu.Backend | None = None) -> LyapunovResult:
    """The full Lyapunov spectrum of the Lorenz system or the double pendulum, as the mean and standard error over an ENSEMBLE of nearby initial
    conditions: every member carries its own tangent frame, advanced together with it by one batched RK4 step, re-orthonormalised by
    Gram-Schmidt every `renorm_steps` steps (Benettin et al. 1980)."""
    if system not in ("lorenz", "pendulum"):
        raise DynamicsError("Choose 'lorenz' or 'pendulum'.")
    if not 8 <= members <= 1_000_000:
        raise DynamicsError("Use between 8 and 1,000,000 ensemble members.")
    if t_run <= 0 or dt <= 0 or t_transient < 0:
        raise DynamicsError("Times and the step must be positive.")
    steps_total = int((t_transient + t_run) / dt)
    if steps_total > 400_000:
        raise DynamicsError("That is more than 400,000 steps: raise dt or shorten the run.")
    if system == "pendulum":
        params.validate()                                                  # type: ignore[union-attr]
    sd = resolve_seed(seed)
    backend = backend or gpu.select(members, "ensemble_trajectories")
    s0 = np.asarray(state0, dtype=float)
    n = len(s0)
    rng = np.random.default_rng(sd)
    init = s0[None, :] + spread * rng.standard_normal((members, n))

    P: Any = params

    def run(xp, to_host):
        if system == "lorenz":
            f = lambda y: lorenz_rhs(y, P, xp)                           # noqa: E731
            jac = lambda y: lorenz_jacobian(y, P, xp)                    # noqa: E731
        else:
            f = lambda y: pendulum_rhs(y, P, xp)                         # noqa: E731
            jac = lambda y: pendulum_jacobian(y, P, xp, backend.module if xp is not np else "numpy")  # noqa: E731
        y = xp.asarray(init)
        for _ in range(int(t_transient / dt)):
            y = rk4_batch(f, y, dt)
        Q = xp.broadcast_to(xp.eye(n), (members, n, n)) + 0.0
        Q = xp.array(Q)
        acc = xp.zeros((members, n))
        n_seg = int(t_run / dt) // renorm_steps
        n_frame = int(t_frame / dt) // renorm_steps          # segments used only to let the tangent frame align with the Oseledets directions

        def aug(state):
            yy, QQ = state
            return f(yy), xp.matmul(jac(yy), QQ)

        for seg in range(n_frame + n_seg):
            for _ in range(renorm_steps):
                k1 = aug((y, Q))
                k2 = aug((y + 0.5 * dt * k1[0], Q + 0.5 * dt * k1[1]))
                k3 = aug((y + 0.5 * dt * k2[0], Q + 0.5 * dt * k2[1]))
                k4 = aug((y + dt * k3[0], Q + dt * k3[1]))
                y = y + dt * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]) / 6
                Q = Q + dt * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]) / 6
            Q, logs = _gram_schmidt(Q, xp)
            if seg >= n_frame:
                acc = acc + logs
        return to_host(acc), n_seg * renorm_steps * dt

    acc, span = gpu.attempt(backend, lambda: run(backend.xp, gpu.to_host), lambda: run(np, lambda a: a))
    used = gpu.last_used() or gpu.CPU.name
    per_member = acc / span
    exps = per_member.mean(0)
    errs = per_member.std(0, ddof=1) / math.sqrt(members)
    order = np.argsort(exps)[::-1]
    exps, errs = exps[order], errs[order]
    ck = CheckList()
    notes: list[str] = []
    ky = None
    if system == "lorenz":
        lp: LorenzParams = params                                           # type: ignore[assignment]
        trace = -(lp.sigma + 1.0 + lp.beta)
        total = float(exps.sum())
        ck.add("The exponents sum to the trace of the Jacobian, -(sigma + 1 + beta)", abs(total - trace) < 5e-3 * abs(trace) + 3 * float(np.sqrt((errs ** 2).sum())),
               f"sum = {total:.4f}, trace = {trace:.4f}: volumes in phase space shrink at exactly this rate, whatever the trajectory")
        ck.add("One exponent is zero (motion along the flow)", abs(exps[1]) < 0.03 + 3 * errs[1], f"the middle exponent is {exps[1]:+.4f} ± {errs[1]:.4f}")
        if (lp.sigma, lp.rho, lp.beta) == (10.0, 28.0, 8.0 / 3.0):
            tol = max(0.05, 3 * errs[0])
            ck.add("The largest exponent matches the published value 0.9056", abs(exps[0] - LORENZ_LAMBDA1_LITERATURE) < tol,
                   f"{exps[0]:.4f} ± {errs[0]:.4f} from {members} trajectories (published 0.9056 for sigma = 10, rho = 28, beta = 8/3)")
        ky = kaplan_yorke(exps)
    else:
        total = float(exps.sum())
        tot_err = float(np.sqrt((errs ** 2).sum()))
        ck.add("The exponents sum to zero (a Hamiltonian system conserves phase-space volume)", abs(total) < 0.05 + 3 * tot_err, f"sum = {total:+.4f} ± {tot_err:.4f}")
        pair = abs(exps[0] + exps[3]) + abs(exps[1] + exps[2])
        ck.add("The exponents come in plus-and-minus pairs", pair < 0.1 + 3 * tot_err, f"λ1 + λ4 = {exps[0] + exps[3]:+.4f}, λ2 + λ3 = {exps[1] + exps[2]:+.4f}")
        ck.add("Two exponents are zero (energy and time translation)", abs(exps[1]) < 0.05 + 3 * errs[1] and abs(exps[2]) < 0.05 + 3 * errs[2],
               f"the middle exponents are {exps[1]:+.4f} and {exps[2]:+.4f}")
    if exps[0] > 3 * errs[0] and exps[0] > 0.05:
        notes.append(f"The largest exponent is {exps[0]:.3f} ± {errs[0]:.3f}: nearby trajectories separate like exp({exps[0]:.2f} t), so a 10-fold smaller initial error "
                     f"buys only ln(10)/{exps[0]:.2f} = {math.log(10) / exps[0]:.1f} extra time units of prediction.")
    else:
        notes.append("The largest exponent is consistent with zero: this orbit is regular (no exponential sensitivity detected over the run).")
    if per_member[:, 0].std() / max(abs(exps[0]), 1e-9) > 1.0:
        notes.append("The ensemble members disagree widely: run longer or use more members for a sharper estimate.")
    return LyapunovResult(system, exps, errs, members, span, dt, sd, ky, ck, used, notes)


@dataclass
class DivergenceCurve:
    times: np.ndarray
    log_separation: np.ndarray        # ensemble mean of ln(d(t)/d0)
    spread: np.ndarray
    slope: float
    fit_window: tuple[float, float]
    members: int
    seed: int
    backend: str = "CPU (NumPy)"


def divergence_curve(system: str, params: LorenzParams | PendulumParams, state0: np.ndarray, members: int = 256, d0: float = 1e-8, t_end: float = 25.0, dt: float = 0.01,
                     t_transient: float = 10.0, seed: int | None = None, backend: gpu.Backend | None = None) -> DivergenceCurve:
    """An ensemble of pairs of nearby trajectories: the mean of ln(separation) against time. Its initial slope (before the separation saturates at the size of the
    attractor) estimates the largest Lyapunov exponent -- a second, independent route to the number lyapunov_spectrum gives."""
    sd = resolve_seed(seed)
    backend = backend or gpu.select(members, "ensemble_trajectories")
    rng = np.random.default_rng(sd)
    s0 = np.asarray(state0, float)
    n = len(s0)
    init = s0[None, :] + 1e-3 * rng.standard_normal((members, n))
    direction = rng.standard_normal((members, n))
    direction /= np.linalg.norm(direction, axis=1, keepdims=True)

    def run(xp, to_host):
        P: Any = params
        f = (lambda y: lorenz_rhs(y, P, xp)) if system == "lorenz" else (lambda y: pendulum_rhs(y, P, xp))
        y = xp.asarray(init)
        for _ in range(int(t_transient / dt)):
            y = rk4_batch(f, y, dt)
        z = y + d0 * xp.asarray(direction)
        out = []
        n_steps = int(t_end / dt)
        stride = max(1, n_steps // 250)
        for k in range(n_steps + 1):
            if k % stride == 0:
                out.append(to_host(xp.log(xp.sqrt(((z - y) ** 2).sum(-1)) / d0)))
            y = rk4_batch(f, y, dt)
            z = rk4_batch(f, z, dt)
        return np.array(out), stride

    logs, stride = gpu.attempt(backend, lambda: run(backend.xp, gpu.to_host), lambda: run(np, lambda a: a))
    used = gpu.last_used() or gpu.CPU.name
    t = np.arange(len(logs)) * stride * dt
    mean = logs.mean(1)
    sat = float(np.log(1e-2 / d0))                                           # stop the fit well before saturation
    below = np.nonzero(mean < sat)[0]
    hi = int(below[-1]) if len(below) else len(t) - 1
    lo = int(np.searchsorted(t, 1.0))
    hi = max(hi, lo + 5)
    slope = float(np.polyfit(t[lo:hi], mean[lo:hi], 1)[0])
    return DivergenceCurve(t, mean, logs.std(1), slope, (float(t[lo]), float(t[min(hi, len(t) - 1)])), members, sd, used)


# ------------------------------------------------------------------------------------------ double-pendulum flip-time map

@dataclass
class FlipMap:
    theta1: np.ndarray
    theta2: np.ndarray
    flip_time: np.ndarray            # (n, n); t_end where the pendulum never flipped
    t_end: float
    energy_error_median: float
    energy_error_p99: float
    energy_error_max: float
    members: int
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"


def flip_map(p: PendulumParams, n: int = 64, t_end: float = 10.0, dt: float = 0.005, backend: gpu.Backend | None = None) -> FlipMap:
    """The classic chaos picture: start the pendulum from every (theta1, theta2) on a grid, at rest, and colour each start by how long it takes
    until a bob first goes over the top. All n x n trajectories are one batched RK4; the energy error of every member is the check."""
    p.validate()
    if not 8 <= n <= 1500:
        raise DynamicsError("Use a grid of 8 to 1500 points per side.")
    if t_end * 1.0 / dt > 100_000:
        raise DynamicsError("Too many steps: raise dt or shorten the run.")
    backend = backend or gpu.select(n * n, "ensemble_trajectories")
    th = np.linspace(-math.pi, math.pi, n)
    T1, T2 = np.meshgrid(th, th)
    init = np.stack([T1.ravel(), T2.ravel(), np.zeros(n * n), np.zeros(n * n)], axis=1)

    def run(xp, to_host):
        f = lambda y: pendulum_rhs(y, p, xp)                              # noqa: E731
        y = xp.asarray(init)
        e0 = pendulum_energy(y, p, xp)
        flip = xp.full(y.shape[0], t_end)
        steps = int(round(t_end / dt))
        for k in range(steps):
            y = rk4_batch(f, y, dt)
            over = (xp.abs(y[:, 0]) > math.pi) | (xp.abs(y[:, 1]) > math.pi)
            flip = xp.where(over & (flip >= t_end), (k + 1) * dt, flip)
        err = xp.abs(pendulum_energy(y, p, xp) - e0) / xp.maximum(1.0, xp.abs(e0))
        return to_host(flip), to_host(err)

    flip, err = gpu.attempt(backend, lambda: run(backend.xp, gpu.to_host), lambda: run(np, lambda a: a))
    used = gpu.last_used() or gpu.CPU.name
    ck = CheckList()
    p99 = float(np.percentile(err, 99))
    ck.add("Energy is conserved for the ensemble (99 % of members)", p99 < 1e-3, f"relative energy error: median {np.median(err):.1e}, 99th percentile {p99:.1e}, worst {err.max():.1e} (tolerance 1e-3 for the 99th)")
    ck.add("The map is symmetric under (θ1, θ2) → (−θ1, −θ2)", _flip_symmetry(flip.reshape(n, n)) < 0.05,
           f"{_flip_symmetry(flip.reshape(n, n)):.1%} of the grid points differ from their mirror image by more than 5% of the run time")
    return FlipMap(th, th, flip.reshape(n, n), t_end, float(np.median(err)), p99, float(err.max()), n * n, ck, used)


def _flip_symmetry(z: np.ndarray) -> float:
    mirror = z[::-1, ::-1]
    # the grid has an odd or even number of points; the mirror of index i is n-1-i, exact for a symmetric linspace
    diff = np.abs(z - mirror) > 0.05 * float(np.max(z))
    return float(np.mean(diff))
