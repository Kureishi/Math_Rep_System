"""
Hamiltonian / Lagrangian front-end -- Streamlit-free.

Type L(q, qdot) or H(q, p) and get, symbolically:
  * the EULER-LAGRANGE equations  d/dt (dL/dqdot_i) - dL/dq_i = 0, solved for the accelerations (needs a non-singular Hessian
    d2L/dqdot_i dqdot_j; a singular one means a constraint and is reported, not guessed);
  * the LEGENDRE transform p_i = dL/dqdot_i, H = p.qdot - L (when the momenta can be inverted in closed form);
  * HAMILTON's equations  dq/dt = dH/dp,  dp/dt = -dH/dq;
  * CONSERVED QUANTITIES from symmetries (Noether): cyclic coordinates (p_i conserved when L does not depend on q_i), energy (when L does
    not depend on t), and a SEARCH over linear point symmetries xi_i = sum_j A_ij q_j + b_i + beta_i t with a gauge function
    Lambda = sum lambda_i q_i + mu t, which finds translations, rotations, Galilean boosts and the uniform-gravity "boost" by solving a
    linear system (the Noether condition  sum_i (dL/dq_i xi_i + dL/dqdot_i dxi_i/dt) = dLambda/dt  holds identically in q, qdot).
    The charge is Q = sum_i p_i xi_i - Lambda;
  * PHASE PORTRAITS (one degree of freedom) and POINCARE SECTIONS (two degrees of freedom: the crossings of a section q_s = c with
    p_s > 0, plotted in the remaining coordinates), from a high-accuracy integration of Hamilton's equations.

SELF-CHECKS (`verify_system`): the Legendre transform is consistent (q̇ from H equals the velocity you started with, and Hamilton's
dp/dt equals d/dt of dL/dqdot from the Euler-Lagrange accelerations, at random states); H is conserved along a trajectory; every
Noether charge is conserved along the SAME trajectory (and symbolically, dQ/dt = 0 on the equations of motion); and the flow is
SYMPLECTIC: the numerical Jacobian J of the time-t flow satisfies J^T Omega J = Omega (hence Liouville's theorem det J = 1).
"""
import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.optimize import brentq

from modules.cc_common import CrossCuttingError, drift_report, names_in, parse_with
from modules.cc_conservation import HamiltonianFlow
from modules.quantum_common import CheckList
from modules.timeout_utils import ComputationTimeoutError, run_with_timeout

T_SYM = sp.Symbol("t", real=True)


@dataclass
class NoetherCharge:
    kind: str                          # "cyclic coordinate" | "energy" | "translation" | "rotation" | "boost" | "symmetry"
    description: str
    charge: sp.Expr                    # in (q, qdot) for a Lagrangian system, (q, p) for a Hamiltonian one
    xi: list[sp.Expr] | None = None
    verified: bool = False
    note: str = ""


@dataclass
class MechanicalSystem:
    name: str
    coords: list[str]
    q: list[sp.Symbol]
    qd: list[sp.Symbol]
    p: list[sp.Symbol]
    params: dict[str, float]
    L: sp.Expr | None
    H: sp.Expr | None
    accelerations: list[sp.Expr] | None = None          # q̈_i(q, q̇), from Euler-Lagrange
    dq: list[sp.Expr] | None = None                     # Hamilton: dq_i/dt
    dp: list[sp.Expr] | None = None
    separable: bool = False
    time_dependent: bool = False
    source: str = "lagrangian"
    notes: list[str] = field(default_factory=list)

    @property
    def n(self) -> int:
        return len(self.q)

    def param_subs(self) -> dict[sp.Symbol, float]:
        return {sp.Symbol(k, positive=True): v for k, v in self.params.items()}


# ------------------------------------------------------------------------------------------ construction

def _symbols(coords: list[str], text: str, params: dict[str, float] | None) -> tuple[dict[str, sp.Symbol], dict[str, float]]:
    if not coords or len(set(coords)) != len(coords) or not all(c.isidentifier() for c in coords):
        raise CrossCuttingError("Give distinct coordinate names, e.g. 'x, y'.")
    if len(coords) > 4:
        raise CrossCuttingError("At most 4 degrees of freedom.")
    q = {c: sp.Symbol(c, real=True) for c in coords}
    qd = {f"{c}dot": sp.Symbol(f"{c}dot", real=True) for c in coords}
    p = {f"p_{c}": sp.Symbol(f"p_{c}", real=True) for c in coords}
    reserved = set(q) | set(qd) | set(p) | {"t"}
    extra = sorted(names_in(text) - reserved)
    values = {k: float((params or {}).get(k, 1.0)) for k in extra}
    syms: dict[str, sp.Symbol] = {**q, **qd, **p, "t": T_SYM}
    syms.update({k: sp.Symbol(k, positive=True) for k in extra})
    return syms, values


def _is_separable(H: sp.Expr, q: list[sp.Symbol], p: list[sp.Symbol]) -> bool:
    return all(sp.simplify(sp.diff(H, a, b)) == 0 for a in q for b in p)


def from_hamiltonian(H_text: str, coords: list[str], params: dict[str, float] | None = None, name: str = "Hamiltonian system") -> MechanicalSystem:
    syms, values = _symbols(coords, H_text, params)
    H = parse_with(H_text, syms)
    q, qd, p = [syms[c] for c in coords], [syms[f"{c}dot"] for c in coords], [syms[f"p_{c}"] for c in coords]
    if T_SYM in H.free_symbols:
        raise CrossCuttingError("H may not depend on t explicitly here (autonomous systems only).")
    if H.free_symbols & set(qd):
        raise CrossCuttingError("H is a function of q and p: use p_<name> for momenta, not <name>dot.")
    sys_ = MechanicalSystem(name, coords, q, qd, p, values, None, H, dq=[sp.diff(H, pi) for pi in p], dp=[-sp.diff(H, qi) for qi in q],
                            separable=_is_separable(H, q, p), source="hamiltonian")
    return sys_


def from_lagrangian(L_text: str, coords: list[str], params: dict[str, float] | None = None, name: str = "Lagrangian system",
                    timeout: float = 30.0) -> MechanicalSystem:
    syms, values = _symbols(coords, L_text, params)
    L = parse_with(L_text, syms)
    q, qd, p = [syms[c] for c in coords], [syms[f"{c}dot"] for c in coords], [syms[f"p_{c}"] for c in coords]
    if L.free_symbols & set(p):
        raise CrossCuttingError("L is a function of q and qdot: use <name>dot for velocities, not p_<name>.")
    n = len(q)
    time_dep = T_SYM in L.free_symbols
    M = sp.Matrix(n, n, lambda i, j: sp.diff(L, qd[i], qd[j]))
    if M.det().simplify() == 0:
        raise CrossCuttingError("The Hessian d²L/dq̇dq̇ is singular: the system has constraints (the Lagrangian is degenerate), so the "
                                "accelerations are not determined by the Euler–Lagrange equations alone.")
    rhs = sp.Matrix([sp.diff(L, q[i]) - sp.diff(L, qd[i], T_SYM) - sum(qd[j] * sp.diff(L, qd[i], q[j]) for j in range(n)) for i in range(n)])
    acc = M.LUsolve(rhs)

    def simp(e: sp.Expr) -> sp.Expr:
        try:
            return run_with_timeout(sp.simplify, e, timeout=timeout, label="simplify")
        except ComputationTimeoutError:
            return e
    accs = [simp(a) for a in acc]
    sys_ = MechanicalSystem(name, coords, q, qd, p, values, L, None, accelerations=accs, time_dependent=time_dep)
    # Legendre transform
    mom = [sp.diff(L, v) for v in qd]
    try:
        b = sp.Matrix([m.subs({v: 0 for v in qd}) for m in mom])
        if all(sp.simplify(sp.diff(M[i, j], v)) == 0 for i in range(n) for j in range(n) for v in qd):       # quadratic in the velocities
            qdot_of_p = M.LUsolve(sp.Matrix(p) - b)
        else:
            sol = sp.solve([pi - mi for pi, mi in zip(p, mom)], qd, dict=True)
            if len(sol) != 1:
                raise ValueError("no unique inverse")
            qdot_of_p = sp.Matrix([sol[0][v] for v in qd])
        sub = dict(zip(qd, list(qdot_of_p)))
        H = simp(sum(pi * sub[v] for pi, v in zip(p, qd)) - L.subs(sub, simultaneous=True))
        sys_.H = H
        sys_.dq = [sp.diff(H, pi) for pi in p]
        sys_.dp = [-sp.diff(H, qi) for qi in q]
        sys_.separable = _is_separable(H, q, p) and not time_dep
    except Exception as exc:  # noqa: BLE001
        sys_.notes.append(f"No closed-form Legendre transform ({exc}); the Euler–Lagrange form is used for dynamics.")
    return sys_


# ------------------------------------------------------------------------------------------ numerics

def to_flow(sys_: MechanicalSystem) -> HamiltonianFlow:
    """Numerical H, dH/dq, dH/dp (parameters substituted) for integration."""
    if sys_.H is None:
        raise CrossCuttingError("This system has no Hamiltonian (the Legendre transform was not available).")
    sub = sys_.param_subs()
    H = sys_.H.subs(sub)
    args = sys_.q + sys_.p
    fH = sp.lambdify(args, H, "numpy")
    gq = sp.lambdify(args, [sp.diff(H, a) for a in sys_.q], "numpy")
    gp = sp.lambdify(args, [sp.diff(H, a) for a in sys_.p], "numpy")
    return HamiltonianFlow(sys_.n, lambda q, p: float(fH(*q, *p)), lambda q, p: np.array(gq(*q, *p), dtype=float),
                           lambda q, p: np.array(gp(*q, *p), dtype=float), sys_.separable)


# ------------------------------------------------------------------------------------------ Noether

def _total_derivative(expr: sp.Expr, sys_: MechanicalSystem) -> sp.Expr:
    """d/dt along solutions: dexpr/dt + sum qdot dexpr/dq + sum qddot dexpr/dqdot."""
    assert sys_.accelerations is not None
    return (sp.diff(expr, T_SYM) + sum(sp.diff(expr, q) * v for q, v in zip(sys_.q, sys_.qd))
            + sum(sp.diff(expr, v) * a for v, a in zip(sys_.qd, sys_.accelerations)))


def _charge_conserved(Q: sp.Expr, sys_: MechanicalSystem) -> tuple[bool, str]:
    dq = _total_derivative(Q, sys_)
    try:
        if sp.simplify(dq) == 0:
            return True, "dQ/dt = 0 on the equations of motion (symbolic)"
    except Exception:  # noqa: BLE001
        pass
    rng = np.random.default_rng(3)
    sub = sys_.param_subs()
    fn = sp.lambdify(sys_.q + sys_.qd + [T_SYM], dq.subs(sub), "numpy")
    pts = rng.uniform(0.3, 1.5, size=(40, 2 * sys_.n + 1))
    with np.errstate(all="ignore"):
        res = float(np.max(np.abs(np.broadcast_to(np.asarray(fn(*pts.T), dtype=float), (40,)))))
    return res < 1e-9, f"dQ/dt = 0 numerically (max {res:.1e} at 40 random states)"


def noether_charges(sys_: MechanicalSystem, search: bool = True) -> list[NoetherCharge]:
    """Conserved quantities of a Lagrangian system from its symmetries (module docstring)."""
    if sys_.L is None or sys_.accelerations is None:
        raise CrossCuttingError("The Noether analysis needs a Lagrangian. For a Hamiltonian system, look for cyclic coordinates "
                                "(dH/dq_i = 0) and use the automatic invariant finder.")
    L, n = sys_.L, sys_.n
    out: list[NoetherCharge] = []
    mom = [sp.diff(L, v) for v in sys_.qd]
    for i, q in enumerate(sys_.q):
        if sp.simplify(sp.diff(L, q)) == 0:
            ok, note = _charge_conserved(mom[i], sys_)
            out.append(NoetherCharge("cyclic coordinate", f"L does not depend on {sys_.coords[i]}: its momentum is conserved", sp.simplify(mom[i]), None, ok, note))
    if not sys_.time_dependent:
        h = sp.simplify(sum(v * m for v, m in zip(sys_.qd, mom)) - L)
        ok, note = _charge_conserved(h, sys_)
        out.append(NoetherCharge("energy", "L does not depend on t: h = Σ q̇ ∂L/∂q̇ − L is conserved", h, None, ok, note))
    if search and not sys_.time_dependent:
        out += _symmetry_search(sys_, [c.charge for c in out])
    return out


def _symmetry_search(sys_: MechanicalSystem, known: list[sp.Expr]) -> list[NoetherCharge]:
    n, L = sys_.n, sys_.L
    assert L is not None
    names: list[str] = []
    A = [[sp.Symbol(f"A{i}{j}") for j in range(n)] for i in range(n)]
    b = [sp.Symbol(f"b{i}") for i in range(n)]
    beta = [sp.Symbol(f"beta{i}") for i in range(n)]
    lam = [sp.Symbol(f"lam{i}") for i in range(n)]
    mu = sp.Symbol("mu")
    unknowns = [a for row in A for a in row] + b + beta + lam + [mu]
    xi = [sum(A[i][j] * sys_.q[j] for j in range(n)) + b[i] + beta[i] * T_SYM for i in range(n)]
    xid = [sum(A[i][j] * sys_.qd[j] for j in range(n)) + beta[i] for i in range(n)]
    cond = sum(sp.diff(L, sys_.q[i]) * xi[i] + sp.diff(L, sys_.qd[i]) * xid[i] for i in range(n)) - (sum(lam[i] * sys_.qd[i] for i in range(n)) + mu)
    cond = cond.subs(sys_.param_subs())
    cols = []
    vars_ = sys_.q + sys_.qd + [T_SYM]
    rng = np.random.default_rng(5)
    pts = rng.uniform(0.3, 1.6, size=(max(8 * len(unknowns), 80), len(vars_)))
    for u in unknowns:
        fn = sp.lambdify(vars_, sp.diff(cond, u), "numpy")
        with np.errstate(all="ignore"):
            cols.append(np.broadcast_to(np.asarray(fn(*pts.T), dtype=float), (len(pts),)).copy())
    M = np.column_stack(cols)
    if not np.all(np.isfinite(M)):
        return []
    _, s, vt = np.linalg.svd(M, full_matrices=True)
    rank = int((s > 1e-9 * max(s[0], 1e-300)).sum())
    null = vt[rank:]
    if len(null) == 0:
        return []
    from modules.cc_conservation import _rref_numeric
    rr, _ = _rref_numeric(np.asarray(null, dtype=float))
    out: list[NoetherCharge] = []
    mom = [sp.diff(L, v) for v in sys_.qd]
    for row in rr:
        vec = [sp.nsimplify(float(c), rational=True, tolerance=1e-9) if abs(c) > 1e-9 else sp.Integer(0) for c in row]
        sub = dict(zip(unknowns, vec))
        xi_v = [sp.simplify(x.subs(sub)) for x in xi]
        lam_v, mu_v = sum(sub[lam[i]] * 0 for i in range(n)), sub[mu]
        Lam = sum(sub[lam[i]] * sys_.q[i] for i in range(n)) + mu_v * T_SYM
        Q = sp.simplify(sum(m * x for m, x in zip(mom, xi_v)) - Lam)
        del lam_v
        if Q == 0 or Q.is_number or any(sp.simplify(Q - k) == 0 or sp.simplify(Q + k) == 0 for k in known):
            continue
        a_mat = sp.Matrix(n, n, lambda i, j: sub[A[i][j]])
        if all(sub[beta[i]] == 0 for i in range(n)) and a_mat == sp.zeros(n, n) and any(sub[b[i]] != 0 for i in range(n)):
            kind = "translation"
        elif a_mat != sp.zeros(n, n) and a_mat == -a_mat.T and all(sub[beta[i]] == 0 for i in range(n)):
            kind = "rotation"
        elif any(sub[beta[i]] != 0 for i in range(n)):
            kind = "boost"
        else:
            kind = "symmetry"
        ok, note = _charge_conserved(Q, sys_)
        desc = "δq = (" + ", ".join(str(x) for x in xi_v) + ")" + (f", gauge Λ = {sp.simplify(Lam)}" if Lam != 0 else "")
        out.append(NoetherCharge(kind, desc, Q, xi_v, ok, note))
    return out


# ------------------------------------------------------------------------------------------ phase space

@dataclass
class PhasePortrait:
    q: np.ndarray
    p: np.ndarray
    H: np.ndarray                                     # H on the (q, p) grid
    trajectories: list[tuple[np.ndarray, np.ndarray]]
    energies: list[float]


def phase_portrait(sys_: MechanicalSystem, q_range: tuple[float, float], p_range: tuple[float, float], energies: list[float] | None = None,
                   n_grid: int = 160, t_max: float = 60.0) -> PhasePortrait:
    """One degree of freedom: the level sets of H and trajectories started on a few of them."""
    if sys_.n != 1:
        raise CrossCuttingError("A phase portrait needs one degree of freedom; use a Poincaré section for two.")
    flow = to_flow(sys_)
    q = np.linspace(*q_range, n_grid)
    p = np.linspace(*p_range, n_grid)
    Hg = np.array([[flow.H(np.array([qq]), np.array([pp])) for qq in q] for pp in p])
    es = energies or list(np.linspace(Hg.min(), Hg.max(), 8)[1:-1])
    trajs: list[tuple[np.ndarray, np.ndarray]] = []
    used: list[float] = []
    for E in es:
        # start on the level set at p = 0 (a turning point) or at q = mid (best effort): find q with H(q, 0) = E
        hq = np.array([flow.H(np.array([qq]), np.array([0.0])) for qq in q]) - E
        starts = [q[i] for i in range(len(q) - 1) if hq[i] * hq[i + 1] < 0]
        z0 = [None, 0.0]
        if starts:
            i0 = next(i for i in range(len(q) - 1) if q[i] == starts[0])
            root = brentq(lambda qq: flow.H(np.array([qq]), np.array([0.0])) - E, q[i0], q[i0 + 1], xtol=1e-14)
            z0 = [root, 0.0]
        else:                                     # no turning point in range (a rotating / unbound orbit): start at the left edge with p > 0
            hp = np.array([flow.H(np.array([q[0]]), np.array([pp])) for pp in p]) - E
            cross = [i for i in range(len(p) - 1) if hp[i] * hp[i + 1] < 0 and p[i] >= 0]
            if not cross:
                continue
            j = cross[0]
            z0 = [q[0], brentq(lambda pp: flow.H(np.array([q[0]]), np.array([pp])) - E, p[j], p[j + 1], xtol=1e-14)]
        sol = solve_ivp(lambda _t, z: [float(flow.dHdp(np.array([z[0]]), np.array([z[1]]))[0]), -float(flow.dHdq(np.array([z[0]]), np.array([z[1]]))[0])],
                        (0, t_max), z0, method="DOP853", rtol=1e-10, atol=1e-12, t_eval=np.linspace(0, t_max, 1200))
        trajs.append((sol.y[0], sol.y[1]))
        used.append(E)
    return PhasePortrait(q, p, Hg, trajs, used)


@dataclass
class PoincareSection:
    points: list[np.ndarray]                         # one (n_cross, 2) array per initial condition
    labels: list[str]
    axes: tuple[str, str]
    energy_errors: list[float]
    section: str


def initial_on_energy(sys_: MechanicalSystem, energy: float, fixed: dict[str, float], solve_for: str, search: tuple[float, float] = (0.0, 5.0),
                      n_scan: int = 4000) -> np.ndarray:
    """A full phase-space point (q..., p...) with H = energy, where `fixed` gives the values of all coordinates/momenta except `solve_for`
    (found as the root of H - E in `search`)."""
    flow = to_flow(sys_)
    order = sys_.coords + [f"p_{c}" for c in sys_.coords]
    if solve_for not in order or set(order) - set(fixed) - {solve_for}:
        raise CrossCuttingError("Fix every phase-space variable except the one to solve for (" + ", ".join(order) + ").")

    def make(v: float) -> np.ndarray:
        d = {**fixed, solve_for: v}
        return np.array([d[k] for k in order], dtype=float)

    def f(v: float) -> float:
        z = make(v)
        return flow.H(z[:sys_.n], z[sys_.n:]) - energy
    grid = np.linspace(*search, n_scan)
    vals = np.array([f(v) for v in grid])
    for i in range(len(grid) - 1):
        if vals[i] * vals[i + 1] < 0:
            return make(brentq(f, grid[i], grid[i + 1], xtol=1e-14))
    raise CrossCuttingError(f"No value of {solve_for} in [{search[0]:g}, {search[1]:g}] puts the point on the energy surface H = {energy:g}: "
                            "that initial condition is not accessible at this energy.")


def poincare_section(sys_: MechanicalSystem, starts: list[np.ndarray], section: str = "x", value: float = 0.0, n_crossings: int = 300,
                     t_max: float = 4000.0, rtol: float = 1e-11) -> PoincareSection:
    """Crossings of {section coordinate = value} with its momentum positive, projected on the other (q, p) pair. Two degrees of freedom."""
    if sys_.n != 2:
        raise CrossCuttingError("Poincaré sections are implemented for two degrees of freedom.")
    if section not in sys_.coords:
        raise CrossCuttingError(f"section must be one of {', '.join(sys_.coords)}.")
    flow = to_flow(sys_)
    s = sys_.coords.index(section)
    o = 1 - s
    pts: list[np.ndarray] = []
    labels: list[str] = []
    errs: list[float] = []

    def rhs(_t: float, z: np.ndarray) -> np.ndarray:
        return np.concatenate([flow.dHdp(z[:2], z[2:]), -flow.dHdq(z[:2], z[2:])])

    def event(_t: float, z: np.ndarray) -> float:
        return z[s] - value
    event.direction = 1.0                                    # type: ignore[attr-defined]
    for z0 in starts:
        e0 = flow.H(z0[:2], z0[2:])
        collected: list[tuple[float, float]] = []
        sol = solve_ivp(rhs, (0, t_max), z0, method="DOP853", rtol=rtol, atol=rtol * 1e-2, events=event, dense_output=False)
        for ye in sol.y_events[0]:
            if ye[2 + s] > 0:
                collected.append((ye[o], ye[2 + o]))
            if len(collected) >= n_crossings:
                break
        pts.append(np.array(collected))
        errs.append(float(max((abs(flow.H(ye[:2], ye[2:]) - e0) for ye in sol.y_events[0]), default=0.0)))
        labels.append(f"E = {e0:.4g}, {sys_.coords[o]}₀ = {z0[o]:.3g}")
    return PoincareSection(pts, labels, (sys_.coords[o], f"p_{sys_.coords[o]}"), errs, f"{section} = {value:g}, p_{section} > 0")


# ------------------------------------------------------------------------------------------ verification

def symplectic_defect(flow: HamiltonianFlow, z0: np.ndarray, t: float = 0.5, eps: float = 1e-6) -> tuple[float, float]:
    """(max |J^T Omega J - Omega|, |det J - 1|) for the time-t flow map, J by central finite differences of the DOP853 flow."""
    n = flow.n_dof

    def phi(z: np.ndarray) -> np.ndarray:
        sol = solve_ivp(lambda _t, y: np.concatenate([flow.dHdp(y[:n], y[n:]), -flow.dHdq(y[:n], y[n:])]), (0, t), z, method="DOP853", rtol=1e-12, atol=1e-14)
        return sol.y[:, -1]
    J = np.zeros((2 * n, 2 * n))
    for k in range(2 * n):
        dz = np.zeros(2 * n)
        dz[k] = eps
        J[:, k] = (phi(z0 + dz) - phi(z0 - dz)) / (2 * eps)
    Om = np.block([[np.zeros((n, n)), np.eye(n)], [-np.eye(n), np.zeros((n, n))]])
    return float(np.max(np.abs(J.T @ Om @ J - Om))), float(abs(np.linalg.det(J) - 1))


def verify_system(sys_: MechanicalSystem, z0: np.ndarray | None = None, t_end: float = 20.0, charges: list[NoetherCharge] | None = None) -> CheckList:
    """Run the self-checks listed in the module docstring; z0 = (q..., p...) (default: a generic point)."""
    ck = CheckList()
    n = sys_.n
    rng = np.random.default_rng(11)
    sub = sys_.param_subs()
    if sys_.L is not None and sys_.H is not None and sys_.accelerations is not None:
        pts = rng.uniform(0.3, 1.2, size=(12, 2 * n))
        mom = [sp.diff(sys_.L, v) for v in sys_.qd]
        fH_dp = sp.lambdify(sys_.q + sys_.p, [sp.diff(sys_.H, pi).subs(sub) for pi in sys_.p], "numpy")
        fmom = sp.lambdify(sys_.q + sys_.qd, [m.subs(sub) for m in mom], "numpy")
        # dp/dt from Hamilton vs d/dt(dL/dqdot) from the Euler-Lagrange accelerations
        pdot_el = [(sum(sp.diff(m, q) * v for q, v in zip(sys_.q, sys_.qd)) + sum(sp.diff(m, v) * a for v, a in zip(sys_.qd, sys_.accelerations))) for m in mom]
        f_el = sp.lambdify(sys_.q + sys_.qd, [e.subs(sub) for e in pdot_el], "numpy")
        f_hdq = sp.lambdify(sys_.q + sys_.p, [(-sp.diff(sys_.H, q)).subs(sub) for q in sys_.q], "numpy")
        worst_v = worst_p = 0.0
        for row in pts:
            q, v = row[:n], row[n:]
            p = np.array(fmom(*q, *v), dtype=float)
            vH = np.array(fH_dp(*q, *p), dtype=float)
            worst_v = max(worst_v, float(np.max(np.abs(vH - v))))
            worst_p = max(worst_p, float(np.max(np.abs(np.array(f_hdq(*q, *p), dtype=float) - np.array(f_el(*q, *v), dtype=float)))))
        ck.add("Legendre transform consistent: ∂H/∂p returns the velocity you started with", worst_v < 1e-8, f"largest mismatch {worst_v:.1e} at 12 random states")
        ck.add("Hamilton's dp/dt = d/dt(∂L/∂q̇) from the Euler–Lagrange accelerations", worst_p < 1e-8, f"largest mismatch {worst_p:.1e} at 12 random states")
    if sys_.H is None:
        return ck
    flow = to_flow(sys_)
    z = np.array(z0 if z0 is not None else np.full(2 * n, 0.5), dtype=float)
    nd = n
    sol = solve_ivp(lambda _t, y: np.concatenate([flow.dHdp(y[:nd], y[nd:]), -flow.dHdq(y[:nd], y[nd:])]), (0, t_end), z, method="DOP853", rtol=1e-12, atol=1e-14,
                    t_eval=np.linspace(0, t_end, 800))
    e = np.array([flow.H(y[:nd], y[nd:]) for y in sol.y.T])
    r = drift_report("H", sol.t, e)
    ck.add("H conserved along a DOP853 trajectory", r.max_rel < 1e-9, r.summary())
    if charges and sys_.L is not None:
        fm = sp.lambdify(sys_.q + sys_.p, [sp.diff(sys_.L, v).subs(sub) for v in sys_.qd], "numpy")
        for c in charges:
            if c.kind == "energy":
                continue
            # charges are expressed in (q, qdot); evaluate with qdot from the trajectory via dH/dp
            qdot_traj = np.array([flow.dHdp(y[:nd], y[nd:]) for y in sol.y.T])
            fq = sp.lambdify(sys_.q + sys_.qd + [T_SYM], c.charge.subs(sub), "numpy")
            vals = np.array([float(np.broadcast_to(np.asarray(fq(*y[:nd], *qd, tt), dtype=float), ())) for y, qd, tt in zip(sol.y.T, qdot_traj, sol.t)])
            rr = drift_report(f"Noether charge ({c.kind})", sol.t, vals)
            ck.add(f"Noether charge conserved along the trajectory: {c.kind} {sp.simplify(c.charge)}", rr.max_rel < 1e-8 or rr.max_abs < 1e-9, rr.summary())
        del fm
    defect, det_err = symplectic_defect(flow, z, t=0.5)
    ck.add("the flow is symplectic: JᵀΩJ = Ω (hence det J = 1, Liouville)", defect < 1e-6 and det_err < 1e-6, f"max |JᵀΩJ − Ω| = {defect:.1e}, |det J − 1| = {det_err:.1e}")
    return ck


# ------------------------------------------------------------------------------------------ presets

PRESETS: dict[str, tuple[str, list[str], dict[str, float], str, str]] = {
    # key: (L or H text, coords, params, "L" | "H", description)
    "harmonic": ("m*xdot^2/2 - k*x^2/2", ["x"], {"m": 1.0, "k": 1.0}, "L", "Harmonic oscillator."),
    "pendulum": ("m*l^2*thdot^2/2 + m*g*l*cos(th)", ["th"], {"m": 1.0, "l": 1.0, "g": 1.0}, "L", "Simple pendulum (th = angle from the downward vertical)."),
    "gravity": ("m*xdot^2/2 - m*g*x", ["x"], {"m": 1.0, "g": 1.0}, "L", "A particle in uniform gravity: the Noether search finds the 'boost' charge m*xdot + m*g*t."),
    "free2d": ("m*(xdot^2 + ydot^2)/2", ["x", "y"], {"m": 1.0}, "L", "Free particle in the plane: translations, rotation, boosts."),
    "central": ("m*(xdot^2 + ydot^2)/2 + k/sqrt(x^2 + y^2)", ["x", "y"], {"m": 1.0, "k": 1.0}, "L", "Kepler problem: angular momentum from the rotation symmetry."),
    "polar_central": ("m*(rdot^2 + r^2*thdot^2)/2 + k/r", ["r", "th"], {"m": 1.0, "k": 1.0}, "L", "Kepler in polar coordinates: th is cyclic."),
    "coupled": ("m*(x1dot^2 + x2dot^2)/2 - k*(x1^2 + x2^2)/2 - kc*(x1 - x2)^2/2", ["x1", "x2"], {"m": 1.0, "k": 1.0, "kc": 0.5}, "L", "Two coupled oscillators."),
    "henon_heiles": ("(p_x^2 + p_y^2)/2 + (x^2 + y^2)/2 + x^2*y - y^3/3", ["x", "y"], {}, "H", "Hénon–Heiles: regular motion at low energy, chaos near E = 1/6."),
    "double_well": ("p_x^2/2 + a*x^4/4 - b*x^2/2", ["x"], {"a": 1.0, "b": 1.0}, "H", "Double-well oscillator."),
    "double_pendulum": ("(m1 + m2)*l1^2*a1dot^2/2 + m2*l2^2*a2dot^2/2 + m2*l1*l2*a1dot*a2dot*cos(a1 - a2) + (m1 + m2)*g*l1*cos(a1) + m2*g*l2*cos(a2)",
                        ["a1", "a2"], {"m1": 1.0, "m2": 1.0, "l1": 1.0, "l2": 1.0, "g": 1.0}, "L", "The double pendulum (chaotic)."),
}


def preset_system(key: str, params: dict[str, float] | None = None) -> MechanicalSystem:
    if key not in PRESETS:
        raise CrossCuttingError(f"Unknown preset '{key}'. Available: {', '.join(PRESETS)}.")
    text, coords, defaults, kind, desc = PRESETS[key]
    merged = {**defaults, **(params or {})}
    sys_ = from_lagrangian(text, coords, merged, key) if kind == "L" else from_hamiltonian(text, coords, merged, key)
    sys_.notes.append(desc)
    return sys_
