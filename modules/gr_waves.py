"""
Linearised gravity and gravitational waves -- Streamlit-free.

Weak field: g = eta + h with |h| << 1, eta = diag(-1, 1, 1, 1), coordinates (t, x, y, z).

  Linearised Riemann   R_abcd = 1/2 (d_b d_c h_ad + d_a d_d h_bc - d_a d_c h_bd - d_b d_d h_ac)
  Ricci                R_bd = eta^ac R_abcd
  Gauge                h -> h - d_a xi_b - d_b xi_a leaves R_abcd unchanged (checked symbolically for any xi)
  Lorenz gauge         d^a hbar_ab = 0 with hbar = h - eta h / 2, in which  box hbar_ab = -16 pi T_ab
  TT gauge             h_0a = 0, h^i_i = 0, d_j h_ij = 0: only the two physical polarisations remain

PLANE WAVES. h_ij = A_+ e+_ij cos(w t - k n.x) + A_x ex_ij cos(w t - k n.x + delta), with e+ = e1 e1 - e2 e2 and
ex = e1 e2 + e2 e1 for unit vectors e1, e2 perpendicular to the propagation direction n (e1 x e2 = n).
It satisfies the vacuum equation R_ab = 0 exactly when the wave is null, w = k: `check_plane_wave` verifies the
gauge conditions and the field equation symbolically, and an OFF-shell wave (w != k) is correctly reported as
NOT a solution (the residual is proportional to w^2 - k^2). `ricci_first_order_full_theory` also gets the first-order
Ricci tensor a second way, from the full non-linear Christoffel/Riemann machinery of gr_curvature applied to
eta + eps h, and the two must agree: the linear formulas are an approximation, so they are tested against the
theory they approximate.

TEST PARTICLES. In TT gauge free particles initially at rest stay at fixed coordinates (Gamma^i_00 = 0 exactly),
and their proper separation oscillates: L(t) = (delta_ij + h_ij/2) L0_j. The ring of test masses is drawn from that,
and the geodesic-deviation equation  xi'' = (1/2) h''_ij xi_j  is integrated numerically as a cross-check.

QUADRUPOLE FORMULA (G and c are parameters; geometric units G = c = 1 by default):
    Q_ij = sum m (x_i x_j - delta_ij r^2 / 3)
    h^TT_ij = (2G / c^4 D) [d^2 Q_ij/dt^2]^TT (retarded time),   P = (G / 5 c^5) <d^3 Q_ij/dt^3  d^3 Q_ij/dt^3>
For a CIRCULAR BINARY the symbolic result is compared with the textbook amplitude 4 G mu Omega^2 a^2 / (c^4 D)
(times (1 + cos^2 i)/2 for h_+, cos i for h_x) and Peters' power (32/5) G^4 m1^2 m2^2 (m1 + m2) / (c^5 a^5).
INSPIRAL: Peters' decay da/dt = -(64/5) G^3 m1 m2 (m1 + m2) / (c^5 a^3) is integrated numerically and compared with its
closed form a(t) = (a0^4 - 4 beta t)^(1/4), the merger time a0^4 / (4 beta) and the chirp rate
df/dt = (96/5) pi^(8/3) (G Mc / c^3)^(5/3) f^(11/3).
Scope: the Newtonian quadrupole (lowest) order; no post-Newtonian corrections and no ringdown.
"""
import math
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp

from modules.gr_curvature import curvature_from_components, evaluate, is_zero, simplify_expr
from modules.gr_metrics import GRInputError, coordinate_symbol
from modules.quantum_common import CheckList

T_, X_, Y_, Z_ = (coordinate_symbol(n) for n in ("t", "x", "y", "z"))
COORDS = (T_, X_, Y_, Z_)
ETA = sp.diag(-1, 1, 1, 1)


# ------------------------------------------------------------------------------------------ linearised gravity

def linearized_riemann(h: sp.Matrix, coords: tuple[sp.Symbol, ...] = COORDS) -> list:
    """R_abcd[a][b][c][d] to first order in h."""
    n = len(coords)
    d2 = lambda e, i, j: sp.diff(e, coords[i], coords[j])  # noqa: E731
    return [[[[sp.Rational(1, 2) * (d2(h[a, d], b, c) + d2(h[b, c], a, d) - d2(h[b, d], a, c) - d2(h[a, c], b, d))
               for d in range(n)] for c in range(n)] for b in range(n)] for a in range(n)]


def linearized_ricci(h: sp.Matrix, coords: tuple[sp.Symbol, ...] = COORDS) -> sp.Matrix:
    R = linearized_riemann(h, coords)
    n = len(coords)
    eta = sp.diag(*([-1] + [1] * (n - 1)))
    return sp.Matrix(n, n, lambda b, d: sum(eta[a, a] * R[a][b][a][d] for a in range(n)))


def gauge_transform(h: sp.Matrix, xi: list[sp.Expr], coords: tuple[sp.Symbol, ...] = COORDS) -> sp.Matrix:
    """h'_ab = h_ab - d_a xi_b - d_b xi_a (xi_a with the index DOWN)."""
    n = len(coords)
    return sp.Matrix(n, n, lambda a, b: h[a, b] - sp.diff(xi[b], coords[a]) - sp.diff(xi[a], coords[b]))


def riemann_gauge_invariant(h: sp.Matrix, xi: list[sp.Expr], coords: tuple[sp.Symbol, ...] = COORDS) -> bool:
    """True if the linearised Riemann tensor of h and of the gauge-transformed h agree exactly."""
    a, b = linearized_riemann(h, coords), linearized_riemann(gauge_transform(h, xi, coords), coords)
    n = len(coords)
    return all(sp.simplify(a[i][j][k][l] - b[i][j][k][l]) == 0 for i in range(n) for j in range(n)
               for k in range(n) for l in range(n))


def tt_project(h_spatial: np.ndarray, n_dir: np.ndarray) -> np.ndarray:
    """Transverse-traceless part of a symmetric 3x3 matrix for propagation along n: P h P - P tr(P h) / 2,
    P = 1 - n n^T."""
    n = np.asarray(n_dir, dtype=float)
    n = n / np.linalg.norm(n)
    P = np.eye(3) - np.outer(n, n)
    ph = P @ np.asarray(h_spatial, dtype=float) @ P
    return ph - 0.5 * P * np.trace(P @ np.asarray(h_spatial, dtype=float))


@dataclass
class PlaneWave:
    h: sp.Matrix                                  # 4x4 in (t, x, y, z)
    direction: sp.Matrix
    omega: sp.Expr
    k: sp.Expr
    e_plus: sp.Matrix
    e_cross: sp.Matrix
    amplitudes: tuple[sp.Expr, sp.Expr]
    phase: sp.Expr


def polarization_basis(n_dir: list[float]) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """(n, e1, e2): an orthonormal right-handed triad with e1 x e2 = n."""
    n = sp.Matrix([sp.nsimplify(v, rational=True) for v in n_dir])
    norm = sp.sqrt(sum(v * v for v in n))
    if norm == 0:
        raise GRInputError("The propagation direction must not be the zero vector.")
    n = (n / norm).applyfunc(sp.simplify)
    helper = sp.Matrix([1, 0, 0]) if abs(float(n[0])) < 0.9 else sp.Matrix([0, 1, 0])
    e1 = helper - (helper.dot(n)) * n
    e1 = (e1 / sp.sqrt(e1.dot(e1))).applyfunc(sp.simplify)
    e2 = n.cross(e1).applyfunc(sp.simplify)
    return n, e1, e2


def plane_wave(n_dir: list[float], a_plus: sp.Expr | float = 1, a_cross: sp.Expr | float = 0,
               omega: sp.Expr | float = 1, wave_number: sp.Expr | float | None = None,
               cross_phase: sp.Expr | float = 0) -> PlaneWave:
    """A TT plane wave travelling along n_dir. `wave_number` defaults to omega (a null wave: a solution of the
    vacuum equations); anything else gives an off-shell wave used to show the check can fail."""
    n, e1, e2 = polarization_basis(n_dir)
    omega, k = sp.sympify(omega), sp.sympify(omega if wave_number is None else wave_number)
    ap, ac = sp.sympify(a_plus), sp.sympify(a_cross)
    phase = omega * T_ - k * (n[0] * X_ + n[1] * Y_ + n[2] * Z_)
    ep = e1 * e1.T - e2 * e2.T
    ec = e1 * e2.T + e2 * e1.T
    hs = ap * ep * sp.cos(phase) + ac * ec * sp.cos(phase + sp.sympify(cross_phase))
    h = sp.zeros(4, 4)
    h[1:, 1:] = hs
    return PlaneWave(h, n, omega, k, ep, ec, (ap, ac), phase)


def _sample_points(k: int = 3) -> list[dict[sp.Symbol, float]]:
    return [{T_: 0.3 + 0.17 * i, X_: 0.5 - 0.11 * i, Y_: 0.8 + 0.09 * i, Z_: 1.2 - 0.23 * i} for i in range(k)]


def check_plane_wave(w: PlaneWave) -> CheckList:
    """TT conditions, Lorenz gauge and the linearised vacuum equations for a plane wave."""
    ck = CheckList()
    h = w.h
    pts = _sample_points()
    ok = all(is_zero(h[0, a], pts)[0] for a in range(4))
    ck.add("TT gauge: h_0a = 0", ok, "no time-time or time-space components")
    tr = h[1, 1] + h[2, 2] + h[3, 3]
    ck.add("TT gauge: traceless  h_ii = 0", is_zero(tr, pts)[0], "the trace of the spatial part vanishes")
    div = [sum(sp.diff(h[i, j], COORDS[i]) for i in range(1, 4)) for j in range(1, 4)]
    ck.add("TT gauge: transverse  d_i h_ij = 0", all(is_zero(d, pts)[0] for d in div), "divergence of the spatial part vanishes")
    hbar = h - ETA * (sum(ETA[a, a] * h[a, a] for a in range(4))) / 2
    lorenz = [sum(ETA[a, a] * sp.diff(hbar[a, b], COORDS[a]) for a in range(4)) for b in range(4)]
    ck.add("Lorenz gauge  d^a hbar_ab = 0", all(is_zero(d, pts)[0] for d in lorenz), "follows from TT; checked directly")
    ric = linearized_ricci(h)
    ok = all(is_zero(ric[a, b], pts)[0] for a in range(4) for b in range(a, 4))
    on_shell = is_zero(sp.sympify(w.omega) - sp.sympify(w.k))[0]
    ck.add("linearised vacuum equations  R_ab = 0", ok,
           "satisfied" if ok else f"not satisfied: the wave is off-shell (omega = {w.omega}, k = {w.k}; needs omega = k)")
    if ok != on_shell:
        ck.add("consistency: R_ab = 0 exactly when omega = k", False, "the two disagree")
    return ck


def ricci_first_order_full_theory(h: sp.Matrix, coords: tuple[sp.Symbol, ...] = COORDS) -> sp.Matrix:
    """d/d(eps) of the exact Ricci tensor of eta + eps*h at eps = 0, from the full non-linear Christoffel and Riemann
    formulas (no linearisation anywhere; nothing is simplified until after the derivative in eps). Used to validate
    `linearized_ricci`. The exact inverse metric is used, so this is not 'h small' algebra in disguise."""
    eps = sp.Symbol("epsilon", positive=True)
    n = len(coords)
    g = ETA + eps * h
    ginv = g.inv()
    d = lambda e, i: sp.diff(e, coords[i])  # noqa: E731
    gam = [[[sum(ginv[k, l] * (d(g[l, j], i) + d(g[l, i], j) - d(g[i, j], l)) for l in range(n)) / 2
             for j in range(n)] for i in range(n)] for k in range(n)]
    def riem(r, s_, m, k):
        return (d(gam[r][k][s_], m) - d(gam[r][m][s_], k)
                + sum(gam[r][m][l] * gam[l][k][s_] - gam[r][k][l] * gam[l][m][s_] for l in range(n)))
    out = sp.zeros(n, n)
    for s_ in range(n):
        for k in range(s_, n):
            ric = sum(riem(r, s_, r, k) for r in range(n))
            out[s_, k] = out[k, s_] = sp.diff(ric, eps).subs(eps, 0)
    return out


def full_theory_agrees(h: sp.Matrix, coords: tuple[sp.Symbol, ...] = COORDS) -> tuple[bool, float]:
    """(agree, largest |difference|) between `linearized_ricci` and `ricci_first_order_full_theory`, at sample points."""
    lin, full = linearized_ricci(h, coords), ricci_first_order_full_theory(h, coords)
    pts = _sample_points(3)
    worst = 0.0
    for i in range(len(coords)):
        for j in range(i, len(coords)):
            for p in pts:
                v = evaluate(lin[i, j] - full[i, j], p)
                worst = max(worst, abs(v) if v == v else float("inf"))
    return worst < 1e-9, worst


# ------------------------------------------------------------------------------------------ test particles

def ring_positions(omega: float, a_plus: float, a_cross: float, times: np.ndarray, n_points: int = 16,
                   cross_phase: float = 0.0, radius: float = 1.0) -> np.ndarray:
    """Proper-detector-frame positions (x, y) of a ring of test masses hit by a wave along z, shape (T, P, 2):
    xi_i(t) = (delta_ij + h_ij(t)/2) xi0_j with h_xx = -h_yy = A_+ cos(wt), h_xy = A_x cos(wt + delta)."""
    ang = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    xi0 = radius * np.column_stack([np.cos(ang), np.sin(ang)])
    out = np.empty((len(times), n_points, 2))
    for k, t in enumerate(times):
        hp, hx = a_plus * math.cos(omega * t), a_cross * math.cos(omega * t + cross_phase)
        H = np.array([[hp, hx], [hx, -hp]])
        out[k] = xi0 + 0.5 * xi0 @ H.T
    return out


def geodesic_deviation_ring(omega: float, a_plus: float, a_cross: float, times: np.ndarray, n_points: int = 16,
                            cross_phase: float = 0.0, radius: float = 1.0) -> tuple[np.ndarray, float]:
    """The same ring from the geodesic-deviation equation xi'' = (1/2) h''(t) xi0, integrated numerically, and the
    largest difference from the closed form of `ring_positions`. To first order in h the source may be evaluated at
    the unperturbed separation xi0; (evaluating it at xi itself would add O(h^2) terms that grow secularly, which
    the linear equation does not claim to describe), and the closed form is then exact, so any difference is
    integration error."""
    ang = np.linspace(0, 2 * math.pi, n_points, endpoint=False)
    xi0 = radius * np.column_stack([np.cos(ang), np.sin(ang)])

    def hmat(t: float, order: int) -> np.ndarray:
        if order == 0:
            hp, hx = a_plus * math.cos(omega * t), a_cross * math.cos(omega * t + cross_phase)
        elif order == 1:
            hp, hx = -omega * a_plus * math.sin(omega * t), -omega * a_cross * math.sin(omega * t + cross_phase)
        else:
            hp, hx = -omega ** 2 * a_plus * math.cos(omega * t), -omega ** 2 * a_cross * math.cos(omega * t + cross_phase)
        return np.array([[hp, hx], [hx, -hp]])

    def rhs(t: float, y: np.ndarray) -> np.ndarray:
        return np.concatenate([y[2 * n_points:], (0.5 * xi0 @ hmat(t, 2).T).ravel()])

    y0 = np.concatenate([(xi0 + 0.5 * xi0 @ hmat(times[0], 0).T).ravel(), (0.5 * xi0 @ hmat(times[0], 1).T).ravel()])
    sol = solve_ivp(rhs, (times[0], times[-1]), y0, t_eval=times, rtol=1e-11, atol=1e-13, method="DOP853")
    pos = sol.y[:2 * n_points].T.reshape(len(times), n_points, 2)
    exact = ring_positions(omega, a_plus, a_cross, times, n_points, cross_phase, radius)
    return pos, float(np.max(np.abs(pos - exact)))


@dataclass
class RestCheck:
    max_coordinate_drift: float                  # how far a particle released at rest moves in TT coordinates
    max_separation_error: float                  # proper separation vs (1 + h/2) L0, relative, first order in h
    separation_t: np.ndarray
    separation: np.ndarray
    separation_first_order: np.ndarray


@lru_cache(maxsize=32)
def free_particle_check(a_plus: float, a_cross: float, omega: float, duration: float, separation: float = 1.0) -> RestCheck:
    """Uses the GEODESIC INTEGRATOR on the exact TT-form metric
        ds^2 = -dt^2 + (1 + A+ cos w(t-z)) dx^2 + (1 - A+ cos w(t-z)) dy^2 + 2 Ax cos(w(t-z)) dx dy + dz^2
    to show that a particle released at rest keeps its coordinates (Gamma^i_00 = 0), and that the proper distance
    between two such particles a coordinate distance L0 apart along x is  L0 sqrt(g_xx) ~ L0 (1 + A+ cos(wt)/2)."""
    from modules.gr_geodesics import build_system, integrate_geodesic, normalise_velocity
    from modules.gr_metrics import GRMetric
    t, x, y, z = COORDS
    A, B, w = sp.Symbol("Ap", positive=True), sp.Symbol("Ax", positive=True), sp.Symbol("w", positive=True)
    ph = sp.cos(w * (t - z))
    g = sp.Matrix([[-1, 0, 0, 0], [0, 1 + A * ph, B * ph, 0], [0, B * ph, 1 - A * ph, 0], [0, 0, 0, 1]])
    metric = GRMetric("tt_wave", "TT plane wave", COORDS, g, {"Ap": a_plus, "Ax": a_cross, "w": omega},
                      "A plane gravitational wave in TT form.", family="", sample={"t": 0.3, "x": 0.5, "y": 0.8, "z": 1.2})
    system = build_system(metric)
    drift = 0.0
    xs = []
    for x0 in (0.0, separation):
        pos = [0.0, x0, 0.0, 0.0]
        u0 = normalise_velocity(system, pos, [0.0, 0.0, 0.0], "timelike")
        res = integrate_geodesic(system, pos, u0, duration, n_samples=400, chunks=8)
        drift = max(drift, float(np.max(np.abs(res.x[:, 1:] - np.array(pos[1:])))))
        xs.append(res)
    tt = xs[0].x[:, 0]
    gxx = 1 + a_plus * np.cos(omega * tt)
    exact = separation * np.sqrt(gxx)                         # proper length along x at z = 0, y = 0 (g_xy does not enter)
    first = separation * (1 + 0.5 * a_plus * np.cos(omega * tt))
    err = float(np.max(np.abs(exact - first)) / separation)
    return RestCheck(drift, err, tt, exact, first)


# ------------------------------------------------------------------------------------------ quadrupole

def sample_series(expr: sp.Expr, t: sp.Symbol, times: np.ndarray) -> np.ndarray:
    """expr(t) at each time, vectorised (lambdified once rather than substituted point by point)."""
    fn = sp.lambdify(t, expr, modules="numpy")
    return np.broadcast_to(np.asarray(fn(np.asarray(times, dtype=float)), dtype=float), np.shape(times)).copy()


@dataclass
class QuadrupoleResult:
    t: sp.Symbol
    Q: sp.Matrix
    Q2: sp.Matrix                                 # d^2 Q / dt^2
    Q3: sp.Matrix                                 # d^3 Q / dt^3
    h_plus: sp.Expr
    h_cross: sp.Expr
    power: sp.Expr
    G: sp.Expr
    c: sp.Expr
    distance: sp.Expr


def observer_basis(inclination: sp.Expr, azimuth: sp.Expr = 0) -> tuple[sp.Matrix, sp.Matrix, sp.Matrix]:
    """(n, theta_hat, phi_hat) for an observer at polar angle `inclination` from the z axis."""
    th, ph = sp.sympify(inclination), sp.sympify(azimuth)
    n = sp.Matrix([sp.sin(th) * sp.cos(ph), sp.sin(th) * sp.sin(ph), sp.cos(th)])
    that = sp.Matrix([sp.cos(th) * sp.cos(ph), sp.cos(th) * sp.sin(ph), -sp.sin(th)])
    phat = sp.Matrix([-sp.sin(ph), sp.cos(ph), 0])
    return n, that, phat


def quadrupole_radiation(masses: list[sp.Expr], positions: list[list[sp.Expr]], t: sp.Symbol, *,
                         inclination: sp.Expr | float = 0, azimuth: sp.Expr | float = 0,
                         distance: sp.Expr | float = 1, G: sp.Expr | float = 1, c: sp.Expr | float = 1) -> QuadrupoleResult:
    """Quadrupole radiation of point masses on given trajectories x_i(t) (lists of three expressions in t)."""
    if len(masses) != len(positions) or not masses:
        raise GRInputError("Give one position (x, y, z as functions of t) per mass.")
    if any(len(p) != 3 for p in positions):
        raise GRInputError("Each position needs three components (x, y, z).")
    G, c, D = sp.sympify(G), sp.sympify(c), sp.sympify(distance)
    Q = sp.zeros(3, 3)
    for m, p in zip(masses, positions):
        r2 = sum(v * v for v in p)
        for i in range(3):
            for j in range(3):
                Q[i, j] += sp.sympify(m) * (p[i] * p[j] - (r2 / 3 if i == j else 0))
    Q = Q.applyfunc(sp.simplify)
    Q2 = Q.diff(t, 2).applyfunc(sp.simplify)
    Q3 = Q.diff(t, 3).applyfunc(sp.simplify)
    _n, that, phat = observer_basis(inclination, azimuth)
    hp = (G / (c ** 4 * D)) * sum(Q2[i, j] * (that[i] * that[j] - phat[i] * phat[j]) for i in range(3) for j in range(3))
    hx = (2 * G / (c ** 4 * D)) * sum(Q2[i, j] * that[i] * phat[j] for i in range(3) for j in range(3))
    P = (G / (5 * c ** 5)) * sum(Q3[i, j] ** 2 for i in range(3) for j in range(3))
    return QuadrupoleResult(t, Q, Q2, Q3, sp.simplify(hp), sp.simplify(hx), sp.simplify(P), G, c, D)


def circular_binary_positions(m1: sp.Expr, m2: sp.Expr, a: sp.Expr, t: sp.Symbol, G: sp.Expr = 1) -> tuple[list, sp.Expr]:
    """Centre-of-mass positions of two masses on a circular orbit of separation a in the x-y plane, and Omega."""
    M = m1 + m2
    Om = sp.sqrt(G * M / a ** 3)
    c_, s_ = sp.cos(Om * t), sp.sin(Om * t)
    return [[m2 / M * a * c_, m2 / M * a * s_, 0], [-m1 / M * a * c_, -m1 / M * a * s_, 0]], Om


@dataclass
class BinaryRadiation:
    m1: float
    m2: float
    a: float
    inclination: float
    distance: float
    G: float
    c: float
    omega: float
    h_amplitude: float                           # 4 G mu Omega^2 a^2 / (c^4 D)
    h_plus_amplitude: float
    h_cross_amplitude: float
    power: float
    quad: QuadrupoleResult
    checks: CheckList = field(default_factory=CheckList)


@lru_cache(maxsize=32)
def circular_binary(m1: float, m2: float, a: float, inclination: float = 0.0, distance: float = 1.0, G: float = 1.0,
                    c: float = 1.0) -> BinaryRadiation:
    """Waveform and power of a circular binary from the symbolic quadrupole, checked against closed forms."""
    if min(m1, m2, a, distance) <= 0:
        raise GRInputError("Masses, separation and distance must be positive.")
    t = sp.Symbol("t", real=True)
    pos, om = circular_binary_positions(sp.nsimplify(m1), sp.nsimplify(m2), sp.nsimplify(a), t, sp.nsimplify(G))
    q = quadrupole_radiation([sp.nsimplify(m1), sp.nsimplify(m2)], pos, t, inclination=inclination,
                             distance=distance, G=G, c=c)
    M = m1 + m2
    mu = m1 * m2 / M
    omega = math.sqrt(G * M / a ** 3)
    h0 = 4 * G * mu * omega ** 2 * a ** 2 / (c ** 4 * distance)
    amp_p, amp_x = h0 * (1 + math.cos(inclination) ** 2) / 2, h0 * abs(math.cos(inclination))
    p_formula = 32 / 5 * G ** 4 * m1 ** 2 * m2 ** 2 * M / (c ** 5 * a ** 5)
    res = BinaryRadiation(m1, m2, a, inclination, distance, G, c, omega, h0, amp_p, amp_x, p_formula, q)
    ts = np.append(np.linspace(0, 2 * math.pi / omega, 400), math.pi / (4 * omega))     # includes the h_x peak
    hp = sample_series(q.h_plus, t, ts)
    hx = sample_series(q.h_cross, t, ts)
    pw = float(sample_series(q.power, t, np.array([0.0]))[0])
    scale = max(h0, 1e-300)
    res.checks.add("h₊ amplitude = 4Gμ Ω²a² (1+cos²ι)/2 / (c⁴D)", abs(np.max(np.abs(hp)) - amp_p) < 1e-9 * scale,
                   f"symbolic {np.max(np.abs(hp)):.6e}, formula {amp_p:.6e}")
    res.checks.add("h× amplitude = 4Gμ Ω²a² cos ι / (c⁴D)", abs(np.max(np.abs(hx)) - amp_x) < 1e-9 * scale,
                   f"symbolic {np.max(np.abs(hx)):.6e}, formula {amp_x:.6e}")
    res.checks.add("radiated power = (32/5) G⁴ m₁²m₂²(m₁+m₂)/(c⁵a⁵)  (Peters)", abs(pw - p_formula) < 1e-9 * p_formula,
                   f"symbolic {pw:.6e}, formula {p_formula:.6e}")
    freq = np.fft.rfft(hp[:400] - hp[:400].mean())
    peak = int(np.argmax(np.abs(freq)))
    res.checks.add("gravitational-wave frequency = 2 × orbital frequency", peak == 2,
                   f"strongest harmonic of h₊ over one orbit: {peak} (expected 2)" if max(abs(hp)) > 0 else "h₊ is zero")
    return res


# ------------------------------------------------------------------------------------------ inspiral

@dataclass
class Inspiral:
    t: np.ndarray
    a: np.ndarray                                 # numerical
    a_exact: np.ndarray
    f_gw: np.ndarray                              # GW frequency 2 * f_orb
    h_amp: np.ndarray                             # strain amplitude 4 (G Mc)^(5/3) (pi f)^(2/3) / (c^4 D)
    merger_time: float
    chirp_mass: float
    beta: float
    checks: CheckList = field(default_factory=CheckList)


@lru_cache(maxsize=32)
def peters_inspiral(m1: float, m2: float, a0: float, G: float = 1.0, c: float = 1.0, distance: float = 1.0,
                    a_end_factor: float = 0.05, n: int = 600) -> Inspiral:
    """Shrinking circular orbit under radiation reaction, from a0 until a = a_end_factor * a0, integrated numerically
    and compared with the closed forms (module docstring). The numerical run is the reference; the closed form is the check."""
    if min(m1, m2, a0, distance) <= 0 or not 0 < a_end_factor < 1:
        raise GRInputError("Masses, separation and distance must be positive and 0 < end fraction < 1.")
    M = m1 + m2
    beta = 64 / 5 * G ** 3 * m1 * m2 * M / c ** 5
    t_merge = a0 ** 4 / (4 * beta)
    a_end = a_end_factor * a0
    t_end = (a0 ** 4 - a_end ** 4) / (4 * beta)
    ev = lambda t, y: y[0] - a_end  # noqa: E731
    ev.terminal = True                                                # type: ignore[attr-defined]
    ts = np.linspace(0, t_end, n)
    sol = solve_ivp(lambda t, y: [-beta / y[0] ** 3], (0, t_end), [a0], t_eval=ts, rtol=1e-13, atol=1e-16, method="DOP853")
    a_num = sol.y[0]
    a_ex = (a0 ** 4 - 4 * beta * sol.t) ** 0.25
    f_gw = np.sqrt(G * M / a_num ** 3) / math.pi                       # = 2 * Omega / (2 pi)
    mc = (m1 * m2) ** 0.6 / M ** 0.2
    h = 4 * (G * mc) ** (5 / 3) * (math.pi * f_gw) ** (2 / 3) / (c ** 4 * distance)
    res = Inspiral(sol.t, a_num, a_ex, f_gw, h, t_merge, mc, beta)
    err = float(np.max(np.abs(a_num - a_ex) / a_ex))
    res.checks.add("numerical a(t) matches the closed form (a₀⁴ − 4βt)^¼", err < 1e-7, f"max relative difference {err:.1e}")
    fdot_num = np.gradient(f_gw, sol.t)
    fdot = 96 / 5 * math.pi ** (8 / 3) * (G * mc / c ** 3) ** (5 / 3) * f_gw ** (11 / 3)
    mid = slice(n // 10, n - n // 10)
    rel = float(np.max(np.abs(fdot_num[mid] - fdot[mid]) / fdot[mid]))
    res.checks.add("chirp rate df/dt = (96/5) π^(8/3) (G𝓜/c³)^(5/3) f^(11/3)", rel < 5e-3,
                   f"max relative difference {rel:.1e} (finite differences of the numerical f(t))")
    res.checks.add("frequency increases monotonically (a chirp)", bool(np.all(np.diff(f_gw) > 0)), "f_gw(t) is strictly increasing")
    return res
