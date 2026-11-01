"""
Stochastic dynamics in one dimension: Langevin simulation, Fokker-Planck, and the fluctuation-dissipation relation
checked numerically -- Streamlit-free.

    OVERDAMPED  gamma dx/dt = -U'(x) + F + sqrt(2 gamma kT) xi(t)                 (Euler-Maruyama)
    UNDERDAMPED m dv/dt = -U'(x) + F - gamma v + sqrt(2 gamma kT) xi(t), dx/dt = v   (BAOAB splitting: exact Ornstein-
                Uhlenbeck step for v, so the velocity variance kT/m is preserved at any dt; second-order for x)
    FOKKER-PLANCK  dP/dt = d/dx [ (U' - F) P / gamma + D dP/dx ],   D = kT / gamma
                solved by a finite-volume scheme with Scharfetter-Gummel fluxes (the exact stationary state of the
                DISCRETE operator is the Boltzmann distribution) and Crank-Nicolson stepping (two backward-Euler
                starter steps, Rannacher smoothing, so a narrow initial Gaussian does not ring).

Everything is seeded the project's way: seed=None picks a fresh concrete seed that is stored in the result (see sm_common).

NUMERICAL CHECKS (each pairs a simulation with something it shares no code with):
  * MSD: ballistic (kT/m) t^2 at short times and 2 D t at long times, with the exact crossover
        MSD(t) = 2 (kT/gamma) [ t - (m/gamma) (1 - e^(-gamma t/m)) ]  (underdamped), 2 (kT/gamma) t (overdamped);
  * EQUIPARTITION <v^2> = kT/m, and for a harmonic trap <x^2> = kT/k;
  * the EINSTEIN RELATION / fluctuation-dissipation theorem: D is measured from the spread of a FORCE-FREE ensemble and the
    mobility mu from the mean drift under a small force F, and D / mu must equal kT (the noise strength and the friction
    are tied together; break that tie and the check fails);
  * GREEN-KUBO: D = integral of <v(0)v(t)> dt, with <v(0)v(t)> = (kT/m) e^(-gamma t/m);
  * BOLTZMANN: the long-time positions in a potential pass a Kolmogorov-Smirnov test against exp(-U/kT);
  * FOKKER-PLANCK vs LANGEVIN: the PDE solution and the histogram of the ensemble at the same time, and for the harmonic trap
    both against the exact Gaussian (mean x0 e^(-t/tau), variance (kT/k)(1 - e^(-2t/tau)) + s0^2 e^(-2t/tau), tau = gamma/k);
  * probability is conserved to rounding, the discrete generator annihilates the Boltzmann vector, and P stays >= 0.
"""
import math
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.integrate import trapezoid
from scipy.sparse import csc_matrix, diags, identity
from scipy.sparse.linalg import splu
from scipy.stats import kstest

from modules.gr_metrics import _names_in, parse_in_symbols
from modules.progress import ProgressFn, chunk_bounds, report
from modules.quantum_common import CheckList
from modules.sm_common import StatMechError, resolve_seed

MAX_WORK = 3e8                  # n_steps * n_paths
X = sp.Symbol("x", real=True)


# ------------------------------------------------------------------------------------------ potentials

@dataclass
class Potential:
    name: str
    text: str
    params: dict[str, float]
    expr: sp.Expr
    _U: object = None
    _dU: object = None

    def U(self, x: np.ndarray | float) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        return np.array(np.broadcast_to(np.asarray(self._U(xa), dtype=float), xa.shape))   # type: ignore[operator, type-var]

    def dU(self, x: np.ndarray | float) -> np.ndarray:
        xa = np.asarray(x, dtype=float)
        return np.array(np.broadcast_to(np.asarray(self._dU(xa), dtype=float), xa.shape))  # type: ignore[operator, type-var]


def potential_from_text(text: str, params: dict[str, float] | None = None, name: str = "custom potential") -> Potential:
    """U(x) typed as an expression in x; every other name is a parameter (default 1)."""
    names = sorted(_names_in(text) - {"x"})
    vals = {p: float((params or {}).get(p, 1.0)) for p in names}
    syms: dict[str, sp.Symbol] = {"x": X}
    syms.update({p: sp.Symbol(p, positive=True) for p in names})
    try:
        expr = parse_in_symbols(text, syms)
    except ValueError as exc:
        raise StatMechError(str(exc)) from None
    expr = expr.subs({syms[p]: v for p, v in vals.items()})
    if expr.free_symbols - {X}:
        raise StatMechError("U(x) must depend only on x and named parameters.")
    pot = Potential(name, text, vals, expr)
    pot._U = sp.lambdify(X, expr, modules="numpy")
    pot._dU = sp.lambdify(X, sp.diff(expr, X), modules="numpy")
    return pot


def preset_potential(kind: str, **params: float) -> Potential:
    if kind == "free":
        return potential_from_text("0", name="free particle")
    if kind == "harmonic":
        return potential_from_text("k*x^2/2", {"k": params.get("k", 1.0)}, "harmonic trap")
    if kind == "double_well":
        return potential_from_text("a*x^4/4 - b*x^2/2", {"a": params.get("a", 1.0), "b": params.get("b", 1.0)}, "double well")
    raise StatMechError("Unknown potential preset (free, harmonic, double_well).")


# ------------------------------------------------------------------------------------------ Langevin

@dataclass
class LangevinResult:
    kind: str
    t: np.ndarray
    x: np.ndarray                       # (n_out, n_paths)
    v: np.ndarray | None
    kT: float
    gamma: float
    mass: float
    dt: float
    seed: int
    potential: Potential
    force: float = 0.0


def _check_run(n_steps: int, n_paths: int, kT: float, gamma: float, dt: float) -> None:
    if kT <= 0 or gamma <= 0 or dt <= 0:
        raise StatMechError("kT, gamma and dt must be positive.")
    if n_paths < 10 or n_steps < 10:
        raise StatMechError("Need at least 10 paths and 10 steps.")
    if n_steps * n_paths > MAX_WORK:
        raise StatMechError(f"That is {n_steps * n_paths:.2g} path-steps; the limit is {MAX_WORK:.0g}. Reduce the paths or the duration, or raise dt.")


def simulate_overdamped(pot: Potential, kT: float, gamma: float, x0: float | np.ndarray, dt: float, t_max: float, n_paths: int,
                        n_out: int = 200, force: float = 0.0, seed: int | None = None, progress: ProgressFn | None = None) -> LangevinResult:
    n_steps = int(round(t_max / dt))
    _check_run(n_steps, n_paths, kT, gamma, dt)
    seed = resolve_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.full(n_paths, float(np.asarray(x0, dtype=float))) if np.isscalar(x0) else np.asarray(x0, dtype=float).copy()
    noise = math.sqrt(2 * kT * dt / gamma)
    stride = max(1, n_steps // n_out)
    t_out, x_out = [0.0], [x.copy()]
    for a, b in chunk_bounds(n_steps, 2000):
        report(progress, "Simulating", a / n_steps)
        for k in range(a, b):
            x += (-pot.dU(x) + force) / gamma * dt + noise * rng.standard_normal(n_paths)
            if (k + 1) % stride == 0:
                t_out.append((k + 1) * dt)
                x_out.append(x.copy())
    return LangevinResult("overdamped", np.array(t_out), np.array(x_out), None, kT, gamma, 1.0, dt, seed, pot, force)


def simulate_underdamped(pot: Potential, kT: float, gamma: float, mass: float, x0: float | np.ndarray, dt: float, t_max: float,
                         n_paths: int, n_out: int = 200, force: float = 0.0, seed: int | None = None, v0: float | None = None,
                         progress: ProgressFn | None = None) -> LangevinResult:
    """BAOAB. Velocities start thermal (variance kT/m) unless v0 is given."""
    n_steps = int(round(t_max / dt))
    _check_run(n_steps, n_paths, kT, gamma, dt)
    if mass <= 0:
        raise StatMechError("The mass must be positive.")
    seed = resolve_seed(seed)
    rng = np.random.default_rng(seed)
    x = np.full(n_paths, float(np.asarray(x0, dtype=float))) if np.isscalar(x0) else np.asarray(x0, dtype=float).copy()
    v = rng.normal(0, math.sqrt(kT / mass), n_paths) if v0 is None else np.full(n_paths, float(v0))
    c1 = math.exp(-gamma * dt / mass)
    c2 = math.sqrt((1 - c1 * c1) * kT / mass)
    f = -pot.dU(x) + force
    stride = max(1, n_steps // n_out)
    t_out, x_out, v_out = [0.0], [x.copy()], [v.copy()]
    for a, b in chunk_bounds(n_steps, 2000):
        report(progress, "Simulating", a / n_steps)
        for k in range(a, b):
            v += 0.5 * dt / mass * f
            x += 0.5 * dt * v
            v = c1 * v + c2 * rng.standard_normal(n_paths)
            x += 0.5 * dt * v
            f = -pot.dU(x) + force
            v += 0.5 * dt / mass * f
            if (k + 1) % stride == 0:
                t_out.append((k + 1) * dt)
                x_out.append(x.copy())
                v_out.append(v.copy())
    return LangevinResult("underdamped", np.array(t_out), np.array(x_out), np.array(v_out), kT, gamma, mass, dt, seed, pot, force)


# ------------------------------------------------------------------------------------------ analysis

def msd(res: LangevinResult) -> np.ndarray:
    """Ensemble mean-squared displacement <(x(t) - x(0))^2> at each output time."""
    return ((res.x - res.x[0]) ** 2).mean(axis=1)


def msd_exact_free(t: np.ndarray, kT: float, gamma: float, mass: float | None) -> np.ndarray:
    """Free-particle MSD: underdamped (mass given, equilibrium initial velocity) or overdamped (mass None)."""
    t = np.asarray(t, dtype=float)
    if mass is None:
        return 2 * kT / gamma * t
    return 2 * kT / gamma * (t - mass / gamma * (1 - np.exp(-gamma * t / mass)))


def fit_diffusion(t: np.ndarray, m: np.ndarray, t_min: float) -> tuple[float, float]:
    """(D, standard error) from the slope of the MSD for t >= t_min (1D: MSD = 2 D t), by least squares."""
    sel = t >= t_min
    if sel.sum() < 4:
        raise StatMechError("Too few points after t_min to fit a slope.")
    p, cov = np.polyfit(t[sel], m[sel], 1, cov=True)
    return float(p[0] / 2), float(math.sqrt(cov[0, 0]) / 2)


def velocity_autocorrelation(res: LangevinResult) -> np.ndarray:
    if res.v is None:
        raise StatMechError("Velocities are only recorded by the underdamped integrator.")
    return (res.v[0] * res.v).mean(axis=1)


def boltzmann_ks(samples: np.ndarray, pot: Potential, kT: float, span: float | None = None) -> tuple[float, float]:
    """(KS statistic, p-value) of independent samples against the Boltzmann density exp(-U/kT)."""
    lo, hi = float(samples.min()), float(samples.max())
    pad = 0.5 * (hi - lo) + 1.0 if span is None else span
    grid = np.linspace(lo - pad, hi + pad, 20001)
    w = np.exp(-(pot.U(grid) - pot.U(grid).min()) / kT)
    cdf = np.concatenate([[0.0], np.cumsum(0.5 * (w[1:] + w[:-1]) * np.diff(grid))])
    cdf /= cdf[-1]
    res = kstest(samples, lambda s: np.interp(s, grid, cdf))
    return float(res.statistic), float(res.pvalue)


@dataclass
class EinsteinCheck:
    D: float
    D_err: float
    mobility: float
    mobility_err: float
    kT_estimate: float
    kT_err: float
    kT_expected: float
    checks: CheckList = field(default_factory=CheckList)
    seed: int = 0


def einstein_relation(kT: float, gamma: float, force: float = 1.0, dt: float = 0.01, t_max: float = 20.0, n_paths: int = 20000,
                      mass: float | None = None, seed: int | None = None, progress: ProgressFn | None = None) -> EinsteinCheck:
    """Numerical fluctuation-dissipation test. D from the spread of a force-free ensemble at time t_max, the mobility from the
    drift of a second ensemble under a small force; D/mu must equal kT. (overdamped if mass is None)."""
    seed = resolve_seed(seed)
    free, driven = preset_potential("free"), preset_potential("free")
    if mass is None:
        r0 = simulate_overdamped(free, kT, gamma, 0.0, dt, t_max, n_paths, 50, 0.0, seed, progress)
        r1 = simulate_overdamped(driven, kT, gamma, 0.0, dt, t_max, n_paths, 50, force, seed + 1, progress)
    else:
        r0 = simulate_underdamped(free, kT, gamma, mass, 0.0, dt, t_max, n_paths, 50, 0.0, seed, None, progress)
        r1 = simulate_underdamped(driven, kT, gamma, mass, 0.0, dt, t_max, n_paths, 50, force, seed + 1, None, progress)
    dx0 = r0.x[-1] - r0.x[0]
    t = r0.t[-1]
    # underdamped: the exact free MSD is 2 (kT/gamma)(t - x), x = (m/gamma)(1 - e^(-gamma t/m)), so D = MSD / (2 (t - x)) = MSD/(2t) * t/(t - x)
    offset = 1.0 if mass is None else t / (t - mass / gamma * (1 - math.exp(-gamma * t / mass)))
    D = float((dx0 ** 2).mean() / (2 * t) * offset)
    D_err = D * math.sqrt(2.0 / n_paths)
    dx1 = r1.x[-1] - r1.x[0]
    if mass is None:
        drift = float(dx1.mean() / t)
    else:                                                 # terminal velocity F/gamma is reached after ~m/gamma: use the late-time mean velocity
        drift = float(r1.v[-1].mean()) if r1.v is not None else float(dx1.mean() / t)
    drift_err = float(dx1.std(ddof=1) / math.sqrt(n_paths) / t) if mass is None else float(r1.v[-1].std(ddof=1) / math.sqrt(n_paths))  # type: ignore[index]
    mob, mob_err = drift / force, drift_err / force
    kt_est = D / mob
    kt_err = kt_est * math.sqrt((D_err / D) ** 2 + (mob_err / mob) ** 2)
    out = EinsteinCheck(D, D_err, mob, mob_err, kt_est, kt_err, kT, seed=seed)
    ck = out.checks
    ck.add("D = kT/γ  (diffusion from the force-free spread)", abs(D - kT / gamma) < 4 * D_err, f"D = {D:.5g} ± {D_err:.2g}, kT/γ = {kT / gamma:.5g}")
    ck.add("μ = 1/γ  (mobility from the drift under a force)", abs(mob - 1 / gamma) < 4 * mob_err, f"μ = {mob:.5g} ± {mob_err:.2g}, 1/γ = {1 / gamma:.5g}")
    ck.add("Einstein relation: D/μ = kT", abs(kt_est - kT) < 4 * kt_err, f"D/μ = {kt_est:.5g} ± {kt_err:.2g}, kT = {kT:.5g}")
    return out


def equipartition_checks(res: LangevinResult, k_spring: float | None = None) -> CheckList:
    """<v^2> = kT/m (underdamped) and <x^2> = kT/k for a harmonic trap, from the second half of the run (after relaxation)."""
    ck = CheckList()
    half = len(res.t) // 2
    n = res.x.shape[1]
    if res.v is not None:
        v2 = float((res.v[half:] ** 2).mean())
        err = float((res.v[half:] ** 2).std() / math.sqrt(n * (len(res.t) - half)))
        ck.add("equipartition: ⟨v²⟩ = kT/m", abs(v2 - res.kT / res.mass) < 5 * err * 3, f"⟨v²⟩ = {v2:.5g}, kT/m = {res.kT / res.mass:.5g}")
    if k_spring is not None:
        x2 = float((res.x[half:] ** 2).mean())
        err = float((res.x[half:] ** 2).std() / math.sqrt(n))
        ck.add("harmonic trap: ⟨x²⟩ = kT/k", abs(x2 - res.kT / k_spring) < 6 * err, f"⟨x²⟩ = {x2:.5g}, kT/k = {res.kT / k_spring:.5g}")
    return ck


def green_kubo(res: LangevinResult) -> tuple[float, float]:
    """(D from the integral of the velocity autocorrelation, kT/gamma). Needs a long free underdamped run."""
    c = velocity_autocorrelation(res)
    return float(trapezoid(c, res.t)), res.kT / res.gamma


# ------------------------------------------------------------------------------------------ Fokker-Planck

def _bernoulli(z: np.ndarray) -> np.ndarray:
    """B(z) = z/(e^z - 1), with B(0) = 1."""
    z = np.asarray(z, dtype=float)
    out = np.ones_like(z)
    big = np.abs(z) > 1e-8
    out[big] = z[big] / np.expm1(z[big])
    return out


def fp_generator(pot: Potential, kT: float, gamma: float, x: np.ndarray, force: float = 0.0) -> csc_matrix:
    """The finite-volume Fokker-Planck generator A (dP/dt = A P) with Scharfetter-Gummel fluxes and zero flux at the ends."""
    n, h, D = len(x), float(x[1] - x[0]), kT / gamma
    phi = (pot.U(x) - force * x) / kT
    a = np.diff(phi)                                           # (U_{i+1} - U_i)/kT
    c = D / h ** 2
    diag = np.zeros(n)
    diag[:-1] -= c * _bernoulli(a)
    diag[1:] -= c * _bernoulli(-a)
    upper = c * _bernoulli(-a)                                 # coefficient of P_{i+1} in row i
    lower = c * _bernoulli(a)                                  # coefficient of P_{i-1} in row i+1
    return csc_matrix(diags([lower, diag, upper], [-1, 0, 1], format="csc"))


@dataclass
class FPSolution:
    x: np.ndarray
    t: np.ndarray
    P: np.ndarray                      # (n_out, nx)
    potential: Potential
    kT: float
    gamma: float
    force: float
    generator: csc_matrix
    checks: CheckList = field(default_factory=CheckList)


def boltzmann_density(pot: Potential, kT: float, x: np.ndarray, force: float = 0.0) -> np.ndarray:
    phi = (pot.U(x) - force * x) / kT
    w = np.exp(-(phi - phi.min()))
    return w / trapezoid(w, x)


def solve_fokker_planck(pot: Potential, kT: float, gamma: float, x_range: tuple[float, float], nx: int, x0: float, s0: float,
                        t_max: float, dt: float, n_out: int = 60, force: float = 0.0, progress: ProgressFn | None = None) -> FPSolution:
    if nx < 50 or nx > 20001:
        raise StatMechError("nx must be between 50 and 20001.")
    if x_range[1] <= x_range[0] or s0 <= 0 or kT <= 0 or gamma <= 0 or dt <= 0 or t_max <= 0:
        raise StatMechError("Need x_min < x_max and positive s0, kT, gamma, dt, t_max.")
    x = np.linspace(x_range[0], x_range[1], nx)
    h = x[1] - x[0]
    if s0 < 2 * h:
        raise StatMechError(f"The initial width s0 = {s0:g} is not resolved by the grid (h = {h:.3g}); use s0 >= {2 * h:.3g} or more points.")
    A = fp_generator(pot, kT, gamma, x, force)
    P = np.exp(-0.5 * ((x - x0) / s0) ** 2)
    P /= trapezoid(P, x)
    n_steps = int(round(t_max / dt))
    stride = max(1, n_steps // n_out)
    I = identity(nx, format="csc")
    lu = splu((I - 0.5 * dt * A).tocsc())             # the same matrix serves Crank-Nicolson and a backward-Euler step of size dt/2
    rhs_mat = (I + 0.5 * dt * A).tocsc()
    ts, Ps = [0.0], [P.copy()]
    for a, b in chunk_bounds(n_steps, 500):
        report(progress, "Solving Fokker–Planck", a / n_steps)
        for k in range(a, b):
            if k < 2:                                           # two half-size backward-Euler steps (Rannacher smoothing)
                P = lu.solve(lu.solve(P))
            else:
                P = lu.solve(rhs_mat @ P)
            if (k + 1) % stride == 0:
                ts.append((k + 1) * dt)
                Ps.append(P.copy())
    sol = FPSolution(x, np.array(ts), np.array(Ps), pot, kT, gamma, force, A)
    sol.checks = fp_checks(sol)
    return sol


def fp_checks(sol: FPSolution) -> CheckList:
    ck = CheckList()
    x, P = sol.x, sol.P
    h = x[1] - x[0]
    cell = np.array([p.sum() * h for p in P])
    ck.add("probability is conserved", float(np.max(np.abs(cell - cell[0]))) < 1e-9, f"max drift of ∫P dx = {float(np.max(np.abs(cell - cell[0]))):.1e}")
    pb = boltzmann_density(sol.potential, sol.kT, x, sol.force)
    resid = float(np.max(np.abs(sol.generator @ pb)) / np.max(pb) * h ** 2 / (sol.kT / sol.gamma))
    ck.add("the discrete generator annihilates the Boltzmann distribution", resid < 1e-9, f"max |A P_B| (relative) = {resid:.1e}")
    ck.add("P ≥ 0 (up to rounding)", float(P.min()) > -1e-6 * float(P.max()), f"min P = {float(P.min()):.2e}")
    return ck


def langevin_vs_fp(res: LangevinResult, sol: FPSolution, index: int = -1, bins: int = 60) -> tuple[CheckList, float]:
    """Histogram of the Langevin ensemble at the chosen output time against the PDE solution at the same time."""
    ck = CheckList()
    t_l = res.t[index]
    j = int(np.argmin(np.abs(sol.t - t_l)))
    edge = float(max(sol.P[j][0], sol.P[j][-1]) / sol.P[j].max())
    ck.add("the domain is wide enough: the PDE's reflecting walls are not felt (P at the edges ≪ P_max)", edge < 1e-4,
           f"P(edge)/P_max = {edge:.1e}" + ("" if edge < 1e-4 else " -- the Langevin walkers are unbounded, so the two will differ; widen the x range"))
    samples = res.x[index]
    hist, edges = np.histogram(samples, bins=bins, range=(float(sol.x[0]), float(sol.x[-1])), density=True)
    centres = 0.5 * (edges[1:] + edges[:-1])
    pde = np.interp(centres, sol.x, sol.P[j])
    l1 = float(np.sum(np.abs(hist - pde)) * (edges[1] - edges[0]) / 2)
    mean_l, var_l = float(samples.mean()), float(samples.var())
    mean_p = float(trapezoid(sol.x * sol.P[j], sol.x))
    var_p = float(trapezoid((sol.x - mean_p) ** 2 * sol.P[j], sol.x))
    n = len(samples)
    ck.add("Langevin ensemble mean = Fokker–Planck mean", abs(mean_l - mean_p) < 5 * math.sqrt(var_p / n) + 1e-3 * (abs(mean_p) + math.sqrt(var_p)),
           f"{mean_l:.5g} vs {mean_p:.5g} at t = {sol.t[j]:.3g}")
    ck.add("Langevin ensemble variance = Fokker–Planck variance", abs(var_l - var_p) < 6 * var_p * math.sqrt(2 / n) + 1e-3 * var_p, f"{var_l:.5g} vs {var_p:.5g}")
    ck.add("histogram vs PDE solution: total-variation distance small", l1 < 4 * math.sqrt(bins / n) + 0.01, f"TV distance {l1:.4f} (sampling noise ≈ {math.sqrt(bins / n):.3f})")
    return ck, l1


def ou_exact(x: np.ndarray, t: float, x0: float, s0: float, kT: float, gamma: float, k: float) -> np.ndarray:
    """Exact Gaussian solution for the harmonic trap U = k x^2/2: mean x0 e^(-t/tau), variance s0^2 e^(-2t/tau) + (kT/k)(1 - e^(-2t/tau))."""
    tau = gamma / k
    mean = x0 * math.exp(-t / tau)
    var = s0 ** 2 * math.exp(-2 * t / tau) + kT / k * (1 - math.exp(-2 * t / tau))
    return np.exp(-0.5 * (x - mean) ** 2 / var) / math.sqrt(2 * math.pi * var)
