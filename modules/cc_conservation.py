"""
Conservation-law monitors and symplectic integrators -- Streamlit-free.

THE IDEA. A numerical simulation is only as trustworthy as the quantities it should conserve. This module (a) FINDS conserved
quantities of an ordinary-differential-equation system automatically, (b) tracks them along a numerical solution and reports
the drift (shared classifier, cc_common.DriftReport: secular versus bounded), and (c) offers the integrators that make the
distinction matter.

FINDING INVARIANTS. For dy/dt = f(y) look for Q(y) = sum_j c_j b_j(y) with dQ/dt = sum_j c_j (grad b_j . f) = 0, where the b_j are
a library of basis functions: monomials up to a chosen degree, ln(y_i) (these give the Lotka-Volterra invariant), and any extra
functions the user supplies (sin, 1/r, ...). Evaluating grad b_j . f at many random points gives a matrix whose NULL SPACE (by SVD)
is the set of conserved combinations; coefficients are then rationalised and the result is VERIFIED by exact symbolic
differentiation (dQ/dt simplifies to 0), or, where SymPy cannot close it, numerically along a trajectory (and labelled as such).
Autonomous systems only (f does not depend on t); parameters take numeric values, so the invariants found are for those values.

INTEGRATORS for a Hamiltonian system H(q, p), dq/dt = dH/dp, dp/dt = -dH/dq:
  * leapfrog / velocity Verlet (separable H = T(p) + V(q)): second order, symplectic;
  * Yoshida 4th order: a composition of leapfrog steps with weights (1/(2 - 2^(1/3)), -2^(1/3)/(2 - 2^(1/3)), same);
  * implicit midpoint (any H): second order, symplectic, solved by fixed-point iteration;
  * RK4 (fixed step) and scipy RK45 / DOP853 (adaptive) as the non-symplectic comparisons.
A symplectic method conserves a "shadow" Hamiltonian H~ = H + O(dt^p) exactly, so H oscillates in a BOUNDED band forever; a
non-symplectic method of the same or higher order drifts SECULARLY. `compare_integrators` shows both on the same problem, with
the cost (right-hand-side evaluations) and the error against a high-accuracy reference solution.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from modules.cc_common import CrossCuttingError, DriftReport, drift_report


# ------------------------------------------------------------------------------------------ finding invariants

@dataclass
class Invariant:
    expr: sp.Expr
    degree: int
    verified_exactly: bool
    note: str = ""


def _monomials(n: int, degree: int) -> list[tuple[int, ...]]:
    out: list[tuple[int, ...]] = []

    def rec(i: int, left: int, cur: list[int]) -> None:
        if i == n:
            if sum(cur) > 0:
                out.append(tuple(cur))
            return
        for k in range(left + 1):
            rec(i + 1, left - k, cur + [k])
    rec(0, degree, [])
    return out


def find_invariants(rhs: list[sp.Expr], state: list[sp.Symbol], max_degree: int = 2, include_logs: bool = False,
                    extra_basis: list[sp.Expr] | None = None, seed: int = 0, n_points: int | None = None) -> list[Invariant]:
    """Conserved quantities of dy/dt = rhs(y) in the span of the basis described in the module docstring."""
    n = len(state)
    if n < 1 or len(rhs) != n:
        raise CrossCuttingError("Give one right-hand side per state variable.")
    if not 1 <= max_degree <= 6:
        raise CrossCuttingError("max_degree must be between 1 and 6.")
    for f in rhs:
        if f.free_symbols - set(state):
            raise CrossCuttingError("The right-hand sides may depend only on the state variables (and numbers): the system must be "
                                    "autonomous with numeric parameters.")
    basis: list[tuple[sp.Expr, int]] = []
    for mono in _monomials(n, max_degree):
        basis.append((sp.Mul(*[s ** k for s, k in zip(state, mono)]), sum(mono)))
    if include_logs:
        basis += [(sp.log(s), 1) for s in state]
    basis += [(sp.sympify(e), 1) for e in (extra_basis or [])]
    derivs = [sum(sp.diff(b, s) * f for s, f in zip(state, rhs)) for b, _ in basis]
    rng = np.random.default_rng(seed)
    m = n_points or max(4 * len(basis), 60)
    fns = [sp.lambdify(state, d, "numpy") for d in derivs]
    pts = rng.uniform(0.4, 2.2, size=(m, n))
    cols = []
    for fn in fns:
        with np.errstate(all="ignore"):
            col = np.broadcast_to(np.asarray(fn(*pts.T), dtype=float), (m,)).copy()
        cols.append(col)
    A = np.column_stack(cols)
    if not np.all(np.isfinite(A)):
        raise CrossCuttingError("The basis or right-hand side is not finite on the sampling region (0.4 < y < 2.2): drop the offending basis function.")
    scale = np.maximum(np.abs(A).max(axis=0), 1e-12)
    _, s, vt = np.linalg.svd(A / scale, full_matrices=True)
    tol = 1e-9 * (s[0] if len(s) else 1.0)
    rank = int((s > tol).sum())
    null = vt[rank:] / scale                                              # rows: coefficient vectors c with A c = 0
    if len(null) == 0:
        return []
    # sparsify: row-reduce the null space so each vector is as simple as possible
    N = sp.Matrix(np.round(null, 12).tolist())
    rr, _ = _rref_numeric(np.asarray(null, dtype=float))
    found: list[Invariant] = []
    for row in rr:
        if np.max(np.abs(row)) < 1e-9:
            continue
        coeffs = [sp.nsimplify(float(c), rational=True, tolerance=1e-9) if abs(c) > 1e-9 else sp.Integer(0) for c in row]
        q = sum(c * b for c, (b, _) in zip(coeffs, basis))
        if q == 0 or q.is_number:
            continue
        deg = max((d for c, (_, d) in zip(coeffs, basis) if c != 0), default=0)
        exact, note = _verify_invariant(q, rhs, state, seed)
        found.append(Invariant(sp.simplify(q), deg, exact, note))
    del N
    return _functionally_independent(found, state, seed)


def _functionally_independent(found: list[Invariant], state: list[sp.Symbol], seed: int) -> list[Invariant]:
    """Drop invariants that are functions of lower-degree ones (Q^2 of a conserved Q is conserved too, but adds nothing): keep an
    invariant only if its gradient is linearly independent of the gradients already kept, at random points."""
    if len(found) < 2:
        return found
    rng = np.random.default_rng(seed + 7)
    pts = rng.uniform(0.5, 2.0, size=(8, len(state)))

    def grads(inv: Invariant) -> np.ndarray:                       # (n_points, n_state)
        out = []
        for s_ in state:
            fn = sp.lambdify(state, sp.diff(inv.expr, s_), "numpy")
            with np.errstate(all="ignore"):
                out.append(np.broadcast_to(np.asarray(fn(*pts.T), dtype=float), (len(pts),)))
        return np.array(out).T

    kept: list[Invariant] = []
    kept_grads: list[np.ndarray] = []
    for inv in sorted(found, key=lambda i: (i.degree, len(str(i.expr)))):
        g = grads(inv)
        trial = kept_grads + [g]
        # functionally independent iff the Jacobian of (kept + new) has full row rank at SOME generic point
        full = any(np.linalg.matrix_rank(np.array([t[k] for t in trial]), tol=1e-8 * max(1.0, np.abs(g[k]).max())) == len(trial)
                   for k in range(len(pts)))
        if full:
            kept.append(inv)
            kept_grads.append(g)
    return kept


def _rref_numeric(M: np.ndarray, tol: float = 1e-9) -> tuple[np.ndarray, list[int]]:
    A = M.copy()
    rows, cols = A.shape
    pivots: list[int] = []
    r = 0
    for c in range(cols):
        if r >= rows:
            break
        k = int(np.argmax(np.abs(A[r:, c]))) + r
        if abs(A[k, c]) < tol:
            continue
        A[[r, k]] = A[[k, r]]
        A[r] /= A[r, c]
        for i in range(rows):
            if i != r:
                A[i] -= A[i, c] * A[r]
        pivots.append(c)
        r += 1
    return A[:r], pivots


def _verify_invariant(q: sp.Expr, rhs: list[sp.Expr], state: list[sp.Symbol], seed: int) -> tuple[bool, str]:
    dq = sum(sp.diff(q, s) * f for s, f in zip(state, rhs))
    try:
        if sp.simplify(dq) == 0:
            return True, "dQ/dt = 0 exactly (symbolic)"
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(seed + 1)
    fn = sp.lambdify(state, dq, "numpy")
    pts = rng.uniform(0.4, 2.2, size=(50, len(state)))
    with np.errstate(all="ignore"):
        res = float(np.max(np.abs(np.broadcast_to(np.asarray(fn(*pts.T), dtype=float), (50,)))))
    return False, f"dQ/dt = 0 numerically (max |dQ/dt| = {res:.1e} at 50 random points); SymPy could not close it exactly"


# ------------------------------------------------------------------------------------------ monitoring a run

@dataclass
class MonitoredRun:
    t: np.ndarray
    y: np.ndarray                      # (n_state, N)
    invariants: list[Invariant]
    reports: list[DriftReport]
    n_rhs: int
    method: str


def integrate_monitored(rhs: list[sp.Expr], state: list[sp.Symbol], y0: list[float], t_end: float, invariants: list[Invariant],
                        method: str = "DOP853", rtol: float = 1e-10, atol: float = 1e-12, n_samples: int = 2000) -> MonitoredRun:
    """Integrate the system and measure the drift of each invariant along the solution."""
    if t_end <= 0 or len(y0) != len(state):
        raise CrossCuttingError("Need t_end > 0 and one initial value per state variable.")
    fns = [sp.lambdify(state, f, "numpy") for f in rhs]
    sol = solve_ivp(lambda _t, y: [float(f(*y)) for f in fns], (0.0, t_end), list(map(float, y0)), method=method, rtol=rtol, atol=atol,
                    t_eval=np.linspace(0, t_end, n_samples))
    if not sol.success:
        raise CrossCuttingError(f"Integration failed: {sol.message}")
    reports = []
    for k, inv in enumerate(invariants):
        qf = sp.lambdify(state, inv.expr, "numpy")
        with np.errstate(all="ignore"):
            vals = np.broadcast_to(np.asarray(qf(*sol.y), dtype=float), sol.t.shape)
        reports.append(drift_report(f"Q{k + 1} = {inv.expr}", sol.t, vals))
    return MonitoredRun(sol.t, sol.y, invariants, reports, int(sol.nfev), method)


# ------------------------------------------------------------------------------------------ Hamiltonian integrators

@dataclass
class HamiltonianFlow:
    """Numerical H(q, p) with gradients, for integration: all arrays are 1-D of length n_dof."""
    n_dof: int
    H: Callable[[np.ndarray, np.ndarray], float]
    dHdq: Callable[[np.ndarray, np.ndarray], np.ndarray]
    dHdp: Callable[[np.ndarray, np.ndarray], np.ndarray]
    separable: bool = False                      # H = T(p) + V(q)


def leapfrog_step(f: HamiltonianFlow, q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    p = p - 0.5 * dt * f.dHdq(q, p)
    q = q + dt * f.dHdp(q, p)
    p = p - 0.5 * dt * f.dHdq(q, p)
    return q, p


_Y1 = 1.0 / (2.0 - 2.0 ** (1.0 / 3.0))
_Y0 = -(2.0 ** (1.0 / 3.0)) * _Y1


def yoshida4_step(f: HamiltonianFlow, q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    for w in (_Y1, _Y0, _Y1):
        q, p = leapfrog_step(f, q, p, w * dt)
    return q, p


def midpoint_step(f: HamiltonianFlow, q: np.ndarray, p: np.ndarray, dt: float, tol: float = 1e-13, max_iter: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """Implicit midpoint: z' = z + dt F((z + z')/2), solved by fixed-point iteration (converges for dt small enough)."""
    qn, pn = q + dt * f.dHdp(q, p), p - dt * f.dHdq(q, p)
    settled = 0
    for _ in range(max_iter):
        qm, pm = 0.5 * (q + qn), 0.5 * (p + pn)
        q2, p2 = q + dt * f.dHdp(qm, pm), p - dt * f.dHdq(qm, pm)
        small = max(np.max(np.abs(q2 - qn)), np.max(np.abs(p2 - pn))) < tol * (1.0 + max(np.max(np.abs(q2)), np.max(np.abs(p2))))
        qn, pn = q2, p2
        if small:
            settled += 1                              # two more sweeps after meeting the tolerance: each shrinks the residual by ~dt, so the
            if settled == 3:                          # per-step error is far below rounding (a residual of 1e-13 per step would accumulate into
                return qn, pn                         # a spurious secular energy drift over 10^5 steps)
    raise CrossCuttingError("The implicit midpoint iteration did not converge: reduce the time step.")


def rk4_step(f: HamiltonianFlow, q: np.ndarray, p: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    def F(qq: np.ndarray, pp: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return f.dHdp(qq, pp), -f.dHdq(qq, pp)
    k1 = F(q, p)
    k2 = F(q + 0.5 * dt * k1[0], p + 0.5 * dt * k1[1])
    k3 = F(q + 0.5 * dt * k2[0], p + 0.5 * dt * k2[1])
    k4 = F(q + dt * k3[0], p + dt * k3[1])
    return (q + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0]), p + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1]))


FIXED_STEPPERS: dict[str, tuple[Callable, int, bool, str]] = {
    # name: (stepper, order, symplectic, needs separable H)
    "leapfrog": (leapfrog_step, 2, True, "leapfrog (Störmer–Verlet)"),
    "yoshida4": (yoshida4_step, 4, True, "Yoshida 4th order"),
    "midpoint": (midpoint_step, 2, True, "implicit midpoint"),
    "rk4": (rk4_step, 4, False, "RK4 (fixed step)"),
}
ADAPTIVE = ("RK45", "DOP853")


@dataclass
class IntegratorRun:
    method: str
    label: str
    symplectic: bool
    order: int | None
    t: np.ndarray
    q: np.ndarray                      # (N, n_dof)
    p: np.ndarray
    energy: np.ndarray
    drift: DriftReport
    n_rhs: int
    end_error: float = float("nan")    # |z_end - z_reference_end|


def integrate_fixed(f: HamiltonianFlow, method: str, q0: np.ndarray, p0: np.ndarray, t_end: float, dt: float, n_out: int = 2000) -> IntegratorRun:
    stepper, order, symplectic, label = FIXED_STEPPERS[method]
    if method in ("leapfrog", "yoshida4") and not f.separable:
        raise CrossCuttingError(f"{label} needs a separable Hamiltonian H = T(p) + V(q); use the implicit midpoint rule for this one.")
    n = int(round(t_end / dt))
    if n < 10 or n > 20_000_000:
        raise CrossCuttingError("Need between 10 and 2e7 steps (adjust dt or the duration).")
    stride = max(1, n // n_out)
    q, p = np.array(q0, dtype=float), np.array(p0, dtype=float)
    ts, qs, ps, es = [0.0], [q.copy()], [p.copy()], [f.H(q, p)]
    evals_per_step = {"leapfrog": 2, "yoshida4": 6, "midpoint": 6, "rk4": 4}[method]
    for k in range(n):
        q, p = stepper(f, q, p, dt)
        if (k + 1) % stride == 0:
            ts.append((k + 1) * dt)
            qs.append(q.copy())
            ps.append(p.copy())
            es.append(f.H(q, p))
    t, e = np.array(ts), np.array(es)
    return IntegratorRun(method, label, symplectic, order, t, np.array(qs), np.array(ps), e, drift_report("H", t, e), n * evals_per_step)


def integrate_adaptive(f: HamiltonianFlow, method: str, q0: np.ndarray, p0: np.ndarray, t_end: float, rtol: float = 1e-8, atol: float = 1e-10,
                       n_out: int = 2000) -> IntegratorRun:
    nd = f.n_dof

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        q, p = z[:nd], z[nd:]
        return np.concatenate([f.dHdp(q, p), -f.dHdq(q, p)])
    sol = solve_ivp(rhs, (0.0, t_end), np.concatenate([q0, p0]), method=method, rtol=rtol, atol=atol, t_eval=np.linspace(0, t_end, n_out))
    if not sol.success:
        raise CrossCuttingError(f"Integration failed: {sol.message}")
    q, p = sol.y[:nd].T, sol.y[nd:].T
    e = np.array([f.H(a, b) for a, b in zip(q, p)])
    return IntegratorRun(method, f"{method} (adaptive, rtol {rtol:g})", False, 5 if method == "RK45" else 8, sol.t, q, p, e, drift_report("H", sol.t, e), int(sol.nfev))


def reference_solution(f: HamiltonianFlow, q0: np.ndarray, p0: np.ndarray, t_end: float) -> np.ndarray:
    """z(t_end) from DOP853 at rtol 1e-13: the benchmark the other integrators' end points are compared with."""
    nd = f.n_dof
    sol = solve_ivp(lambda _t, z: np.concatenate([f.dHdp(z[:nd], z[nd:]), -f.dHdq(z[:nd], z[nd:])]), (0.0, t_end), np.concatenate([q0, p0]),
                    method="DOP853", rtol=1e-13, atol=1e-14)
    return sol.y[:, -1]


@dataclass
class IntegratorComparison:
    runs: list[IntegratorRun]
    reference_end: np.ndarray
    notes: list[str] = field(default_factory=list)


def compare_integrators(f: HamiltonianFlow, q0: np.ndarray, p0: np.ndarray, t_end: float, dt: float,
                        methods: list[str] | None = None, adaptive: tuple[str, ...] = ("RK45",)) -> IntegratorComparison:
    """Every requested method on the same problem. Fixed-step methods use dt; adaptive ones their tolerance."""
    methods = methods or (["leapfrog", "yoshida4"] if f.separable else []) + ["midpoint", "rk4"]
    ref = reference_solution(f, q0, p0, t_end)
    runs: list[IntegratorRun] = []
    notes: list[str] = []
    for m in methods:
        try:
            run = integrate_fixed(f, m, q0, p0, t_end, dt)
        except CrossCuttingError as exc:
            notes.append(f"{m}: {exc}")
            continue
        runs.append(run)
    for m in adaptive:
        runs.append(integrate_adaptive(f, m, q0, p0, t_end))
    nd = f.n_dof
    for r in runs:
        z = np.concatenate([r.q[-1], r.p[-1]])
        r.end_error = float(np.linalg.norm(z - ref)) if abs(r.t[-1] - t_end) < 1e-9 * t_end + dt else float("nan")
    return IntegratorComparison(runs, ref, notes)


# ------------------------------------------------------------------------------------------ monitors for the app's other simulations

def reports_from_wavepacket(res: object) -> list[DriftReport]:
    """Norm and energy of a wavepacket run (modules/quantum_dynamics.DynamicsResult), plus momentum when the potential is flat (free particle)."""
    t = np.asarray(res.series_times)                                              # type: ignore[attr-defined]
    out = [drift_report("norm ∫|ψ|²dx", t, np.asarray(res.norm)),                 # type: ignore[attr-defined]
           drift_report("energy ⟨H⟩", t, np.asarray(res.energy))]                 # type: ignore[attr-defined]
    pot = np.asarray(res.potential)                                               # type: ignore[attr-defined]
    if np.ptp(pot) < 1e-12:
        out.append(drift_report("momentum ⟨p⟩", t, np.asarray(res.mean_p)))       # type: ignore[attr-defined]
    return out


def reports_from_geodesic(res: object) -> list[DriftReport]:
    """The conserved quantities of a modules/gr_geodesics.GeodesicResult as drift reports."""
    lam = np.asarray(res.lam)                                                     # type: ignore[attr-defined]
    out = []
    for c in res.conserved:                                                       # type: ignore[attr-defined]
        v = np.asarray(c.values, dtype=float)
        ok = np.isfinite(v)
        if ok.sum() >= 8:
            is_norm = c.name.startswith("norm")
            out.append(drift_report(c.name, lam[ok], v[ok], 0.0 if is_norm else None, scale=1.0 if is_norm else None))
    return out


def monitor_ivp(setup: object, max_degree: int = 2, rtol: float = 1e-10) -> MonitoredRun | str:
    """Automatic conservation monitor for the app's ODE pipeline (modules/ode_utils.IVPSetup): search for invariants of the system the
    solver just solved and measure their drift along an independent numerical integration. Returns a MonitoredRun, or a one-line reason
    why nothing was done (not autonomous, nothing found, ...)."""
    exprs = getattr(setup, "rhs_exprs", None)
    funcs = getattr(setup, "state_funcs", None)
    t = getattr(setup, "t_symbol", None)
    if not exprs or not funcs or t is None:
        return "no symbolic right-hand side available for this system"
    names = [str(f.func) for f in funcs]
    state = [sp.Symbol(n, real=True) for n in names]
    sub = dict(zip(funcs, state))
    rhs = [e.xreplace(sub) for e in exprs]
    if any(t in e.free_symbols for e in rhs):
        return "the system is not autonomous (its right-hand side depends on t explicitly), so the invariant search does not apply"
    if any(e.free_symbols - set(state) for e in rhs):
        return "the right-hand side has symbols without numeric values"
    y0 = list(getattr(setup, "y0"))
    invs: list[Invariant] = []
    try:
        invs = find_invariants(rhs, state, max_degree, False)
        if not invs and all(v > 0 for v in y0):
            invs = find_invariants(rhs, state, 1, True)
    except CrossCuttingError as exc:
        return str(exc)
    if not invs:
        return "no conserved polynomial (or ln-polynomial) quantity of degree <= %d was found: for a dissipative system that is the right answer" % max_degree
    return integrate_monitored(rhs, state, y0, float(getattr(setup, "window")) * 4, invs, "DOP853", rtol, rtol * 1e-2, 800)
