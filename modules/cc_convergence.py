"""
Convergence studies: automatic refinement with an order-of-convergence estimate -- Streamlit-free.

`convergence_study(solve, h0, ...)` calls `solve(h)` on a SEQUENCE OF REFINEMENTS h_k = h0 / ratio^k and returns:
  * the error at each level (against an exact `reference` if given, otherwise against the Richardson extrapolation of the three finest
    levels), in a chosen norm;
  * the OBSERVED ORDER p: pairwise p_k = ln(d_k / d_{k+1}) / ln(ratio) from successive differences d_k = |u_k - u_{k+1}| (needs no
    reference at all), summarised by the median of the last three, and the least-squares slope of ln e against ln h;
  * the Richardson-extrapolated value u_inf = u_n + (u_n - u_{n-1}) / (ratio^p - 1);
  * Roache's GRID CONVERGENCE INDEX GCI = 1.25 |u_n - u_{n-1}| / (ratio^p - 1) (relative): an error BAR for the finest solution;
  * checks: errors fall monotonically until the rounding floor, the observed order matches the expected one, the sequence is in the
    ASYMPTOTIC RANGE (the last two pairwise orders agree), and (with a reference) the exact value lies within the GCI band.
Levels whose error has reached the rounding floor (~1e-13 of the solution) are excluded from the fits and reported, because an
"order" fitted through round-off is meaningless.

READY-MADE STUDIES, each verified against the known order of its method:
  * `ode_convergence`          Euler (1), Heun (2), midpoint (2), RK4 (4) on any first-order system with an exact solution;
  * `hamiltonian_convergence`  leapfrog (2), Yoshida (4), implicit midpoint (2), RK4 (4) from modules/cc_conservation;
  * `poisson_convergence`      the app's finite-difference Poisson solver against a MANUFACTURED solution (second order on a rectangle);
  * `wavepacket_convergence`   the app's split-step wavepacket solver in a harmonic trap against the exact classical-oscillation law
                               (Strang splitting: second order in dt);
  * `sde_strong_convergence`   Euler-Maruyama (strong order 1/2 for multiplicative noise) and Milstein (1) on geometric Brownian motion,
                               with the Brownian increments shared between levels so the paths are comparable.
"""
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.cc_common import CrossCuttingError
from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList

FLOOR = 1e-13


@dataclass
class ConvergenceResult:
    name: str
    h: np.ndarray
    values: list
    errors: np.ndarray
    error_basis: str                     # "reference" | "Richardson extrapolation"
    pairwise_orders: np.ndarray
    observed_order: float
    fit_order: float | None
    extrapolated: object
    gci: float | None
    expected_order: float | None
    at_floor: list[bool] = field(default_factory=list)
    checks: CheckList = field(default_factory=CheckList)


def _max_norm(a: np.ndarray) -> float:
    return float(np.max(np.abs(a)))


def convergence_study(solve: Callable[[float], object], h0: float, n_levels: int = 5, ratio: float = 2.0, reference: object = None,
                      expected_order: float | None = None, order_tol: float = 0.25, norm: Callable[[np.ndarray], float] = _max_norm,
                      name: str = "convergence study", progress: ProgressFn | None = None) -> ConvergenceResult:
    if n_levels < 4:
        raise CrossCuttingError("Need at least 4 refinement levels to judge an order of convergence.")
    if ratio <= 1 or h0 <= 0:
        raise CrossCuttingError("Need h0 > 0 and a refinement ratio greater than 1.")
    h = np.array([h0 / ratio ** k for k in range(n_levels)])
    vals = []
    for k, hk in enumerate(h):
        report(progress, f"Level {k + 1} of {n_levels}: h = {hk:.4g}", k / n_levels)
        vals.append(np.asarray(solve(float(hk)), dtype=float))
    d = np.array([norm(vals[k] - vals[k + 1]) for k in range(n_levels - 1)])
    scale = max(norm(vals[-1]), 1e-300)
    with np.errstate(divide="ignore", invalid="ignore"):
        pairwise = np.log(d[:-1] / d[1:]) / math.log(ratio)
    pairwise = np.where(np.isfinite(pairwise), pairwise, np.nan)
    tail = pairwise[-3:][np.isfinite(pairwise[-3:])]
    p_obs = float(np.median(tail)) if len(tail) else float("nan")
    if reference is not None:
        ref = np.asarray(reference, dtype=float)
        err = np.array([norm(v - ref) for v in vals])
        basis = "reference"
        extrap = None
    else:
        p_use = p_obs if np.isfinite(p_obs) and p_obs > 0.2 else (expected_order or 1.0)
        extrap = vals[-1] + (vals[-1] - vals[-2]) / (ratio ** p_use - 1)
        err = np.array([norm(v - extrap) for v in vals])
        basis = "Richardson extrapolation"
    floor = FLOOR * max(scale, 1.0 if scale < 1e-3 else scale)
    at_floor = [bool(e <= floor) for e in err]
    use = [k for k in range(n_levels) if not at_floor[k] and err[k] > 0]
    fit = None
    if len(use) >= 3:
        fit = float(np.polyfit(np.log(h[use]), np.log(err[use]), 1)[0])
    p_gci = p_obs if np.isfinite(p_obs) and p_obs > 0.2 else (expected_order or 1.0)
    gci = float(1.25 * (d[-1] / scale) / (ratio ** p_gci - 1)) if d[-1] > 0 else 0.0
    if reference is None:
        extrap_out: object = extrap
    else:
        extrap_out = vals[-1] + (vals[-1] - vals[-2]) / (ratio ** p_gci - 1)
    res = ConvergenceResult(name, h, vals, err, basis, pairwise, p_obs, fit, extrap_out, gci, expected_order, at_floor)
    ck = res.checks
    live = [k for k in range(n_levels) if not at_floor[k]]
    mono = all(err[a] > err[b] for a, b in zip(live, live[1:]) if a + 1 == b) if len(live) > 1 else True
    ck.add("the error decreases at every refinement (until the rounding floor)", mono,
           "errors: " + ", ".join(f"{e:.2e}" for e in err) + (f"; {sum(at_floor)} level(s) at the rounding floor" if any(at_floor) else ""))
    good = pairwise[np.isfinite(pairwise)]
    if len(good) >= 2:
        asym = abs(good[-1] - good[-2]) < max(0.25, 0.1 * abs(good[-1]))
        ck.add("asymptotic range reached (the last two observed orders agree)", bool(asym), f"last orders {good[-2]:.3f}, {good[-1]:.3f}")
    if expected_order is not None:
        ck.add(f"observed order = expected order {expected_order:g}", bool(np.isfinite(p_obs) and abs(p_obs - expected_order) <= order_tol),
               f"observed {p_obs:.3f}" + (f" (log-log fit {fit:.3f})" if fit is not None else ""))
    if reference is not None and gci:
        band = gci * scale
        ck.add("the exact value lies within the grid-convergence band of the finest solution", bool(norm(vals[-1] - np.asarray(reference)) <= 3 * band + floor),
               f"finest error {err[-1]:.2e}, GCI band ±{band:.2e}")
    return res


# ------------------------------------------------------------------------------------------ ODE

ODE_ORDERS = {"euler": 1, "heun": 2, "midpoint": 2, "rk4": 4}


def _ode_step(method: str, f: Callable[[float, np.ndarray], np.ndarray], t: float, y: np.ndarray, h: float) -> np.ndarray:
    if method == "euler":
        return y + h * f(t, y)
    if method == "heun":
        k1 = f(t, y)
        return y + 0.5 * h * (k1 + f(t + h, y + h * k1))
    if method == "midpoint":
        return y + h * f(t + 0.5 * h, y + 0.5 * h * f(t, y))
    if method == "rk4":
        k1 = f(t, y)
        k2 = f(t + 0.5 * h, y + 0.5 * h * k1)
        k3 = f(t + 0.5 * h, y + 0.5 * h * k2)
        k4 = f(t + h, y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    raise CrossCuttingError(f"method must be one of {', '.join(ODE_ORDERS)}.")


def ode_convergence(f: Callable[[float, np.ndarray], np.ndarray], y0: list[float], t_end: float, method: str, steps0: int = 8, n_levels: int = 6,
                    exact: Callable[[float], np.ndarray] | None = None, progress: ProgressFn | None = None) -> ConvergenceResult:
    """y(t_end) by a fixed-step method at step sizes t_end/steps0 / 2^k."""
    if method not in ODE_ORDERS:
        raise CrossCuttingError(f"method must be one of {', '.join(ODE_ORDERS)}.")

    def solve(h: float) -> np.ndarray:
        n = int(round(t_end / h))
        y = np.array(y0, dtype=float)
        for k in range(n):
            y = _ode_step(method, f, k * h, y, h)
        return y
    return convergence_study(solve, t_end / steps0, n_levels, 2.0, None if exact is None else np.asarray(exact(t_end)), ODE_ORDERS[method],
                             name=f"{method} on an ODE system", progress=progress)


def hamiltonian_convergence(flow, q0: np.ndarray, p0: np.ndarray, t_end: float, method: str, steps0: int = 16, n_levels: int = 6,
                            progress: ProgressFn | None = None) -> ConvergenceResult:
    """Fixed-step Hamiltonian integrator against the DOP853 reference."""
    from modules.cc_conservation import FIXED_STEPPERS, reference_solution
    if method not in FIXED_STEPPERS:
        raise CrossCuttingError(f"method must be one of {', '.join(FIXED_STEPPERS)}.")
    stepper, order, _, label = FIXED_STEPPERS[method]
    ref = reference_solution(flow, q0, p0, t_end)

    def solve(h: float) -> np.ndarray:
        q, p = np.array(q0, dtype=float), np.array(p0, dtype=float)
        for _ in range(int(round(t_end / h))):
            q, p = stepper(flow, q, p, h)
        return np.concatenate([q, p])
    return convergence_study(solve, t_end / steps0, n_levels, 2.0, ref, order, name=label, progress=progress)


# ------------------------------------------------------------------------------------------ PDE

def poisson_convergence(exact_expr: str, x_range: tuple[float, float] = (0.0, 1.0), y_range: tuple[float, float] = (0.0, 1.0), n0: int = 9,
                        n_levels: int = 4, progress: ProgressFn | None = None) -> ConvergenceResult:
    """The app's finite-difference Poisson solver (u_xx + u_yy = source) against a MANUFACTURED solution: the source is the Laplacian of
    `exact_expr` and the boundary values are its trace, so the exact answer is known everywhere. Second order on a rectangle."""
    from modules.gr_metrics import parse_in_symbols
    from modules.pde_utils import _build_and_solve_fd_grid
    x, y = sp.symbols("x y", real=True)
    try:
        u = parse_in_symbols(exact_expr, {"x": x, "y": y})
    except ValueError as exc:
        raise CrossCuttingError(str(exc)) from None
    src = sp.diff(u, x, 2) + sp.diff(u, y, 2)
    fu, fs = sp.lambdify((x, y), u, "numpy"), sp.lambdify((x, y), src, "numpy")
    if n0 < 5:
        raise CrossCuttingError("n0 must be at least 5.")

    def solve(h: float) -> np.ndarray:
        n = int(round((x_range[1] - x_range[0]) / h)) + 1
        xs, ys, U, inside = _build_and_solve_fd_grid(n, n, x_range, y_range, lambda a, b: (x_range[0] < a < x_range[1]) and (y_range[0] < b < y_range[1]),
                                                     lambda a, b: float(fu(a, b)), lambda a, b: float(fs(a, b)))
        # the grid nodes on the boundary ring are 'outside' the open domain and hold NaN: they carry the boundary values
        full = np.array([[float(fu(a, b)) for b in ys] for a in xs])
        U = np.where(inside, U, full)
        # sample at the n0-grid points (shared by every level, since the grids are nested)
        stride = (n - 1) // (n0 - 1)
        return np.array([[U[i, j] for j in range(0, n, stride)] for i in range(0, n, stride)])
    xs0 = np.linspace(x_range[0], x_range[1], n0)
    ys0 = np.linspace(y_range[0], y_range[1], n0)
    ref = np.array([[float(fu(a, b)) for b in ys0] for a in xs0])
    res = convergence_study(solve, (x_range[1] - x_range[0]) / (n0 - 1), n_levels, 2.0, ref, 2.0, name="finite-difference Poisson solver", progress=progress)
    return res


# ------------------------------------------------------------------------------------------ wavepacket

def wavepacket_convergence(steps0: int = 40, n_levels: int = 5, omega: float = 1.0, x0: float = 2.0, k0: float = 1.0, sigma: float = 0.7,
                           total_time: float = 6.0, progress: ProgressFn | None = None) -> ConvergenceResult:
    """The app's split-step wavepacket solver in a harmonic trap: <x>(T) against the exact (Ehrenfest = classical) law
    x0 cos(wT) + k0/(m w) sin(wT), as the number of time steps doubles. Strang splitting is second order in dt."""
    from modules.quantum_dynamics import evolve_wavepacket
    exact = x0 * math.cos(omega * total_time) + k0 / omega * math.sin(omega * total_time)

    def solve(h: float) -> float:
        steps = int(round(total_time / h))
        r = evolve_wavepacket("harmonic", {"w": omega, "center": 0.0}, x0=x0, sigma=sigma, k0=k0, domain=(-12.0, 12.0), n_points=512,
                              total_time=total_time, steps=steps, n_frames=2, method="split-step", cross_check=False)
        return float(r.mean_x[-1])
    return convergence_study(solve, total_time / steps0, n_levels, 2.0, exact, 2.0, order_tol=0.3, name="split-step wavepacket (Strang)", progress=progress)


# ------------------------------------------------------------------------------------------ SDE

@dataclass
class SDEStudy:
    result: ConvergenceResult
    scheme: str
    seed: int


def sde_strong_convergence(scheme: str = "euler", mu: float = 1.0, sigma: float = 1.0, x0: float = 1.0, T: float = 1.0, n_paths: int = 20000,
                           steps0: int = 4, n_levels: int = 6, seed: int | None = None, progress: ProgressFn | None = None) -> SDEStudy:
    """Strong error E|X_T - X_T^h| of Euler-Maruyama (order 1/2) or Milstein (order 1) for geometric Brownian motion dX = mu X dt + sigma X dW,
    whose exact solution X_T = x0 exp((mu - sigma^2/2) T + sigma W_T) is evaluated on the SAME Brownian path (the coarse increments are
    sums of the finest ones)."""
    from modules.sm_common import resolve_seed
    if scheme not in ("euler", "milstein"):
        raise CrossCuttingError("scheme must be euler or milstein.")
    seed = resolve_seed(seed)
    n_fine = steps0 * 2 ** (n_levels - 1)
    rng = np.random.default_rng(seed)
    dw = rng.normal(0.0, math.sqrt(T / n_fine), size=(n_paths, n_fine))
    exact = x0 * np.exp((mu - 0.5 * sigma ** 2) * T + sigma * dw.sum(axis=1))

    def solve_scheme(h: float) -> np.ndarray:
        n = int(round(T / h))
        group = n_fine // n
        inc = dw.reshape(n_paths, n, group).sum(axis=2)
        x = np.full(n_paths, x0)
        for k in range(n):
            dwk = inc[:, k]
            x = x + mu * x * h + sigma * x * dwk + (0.5 * sigma ** 2 * x * (dwk ** 2 - h) if scheme == "milstein" else 0.0)
        return x
    # the error at each level is the strong error, the mean over paths of |X_T - X_T^h|: the values are per-path arrays and the norm is a mean
    res = convergence_study(solve_scheme, T / steps0, n_levels, 2.0, exact, 0.5 if scheme == "euler" else 1.0, order_tol=0.2,
                            norm=lambda a: float(np.mean(np.abs(a))), name=f"{scheme} strong convergence (GBM)", progress=progress)
    return SDEStudy(res, scheme, seed)
