"""
Lattice Monte Carlo (Metropolis) for the Ising, q-state Potts and XY models on an L x L periodic square lattice --
Streamlit-free.

    Ising   H = -J sum s_i s_j - h sum s_i                      s = +-1
    Potts   H = -J sum delta(s_i, s_j) - h sum delta(s_i, 0)    s in {0..q-1}
    XY      H = -J sum cos(th_i - th_j) - h sum cos(th_i)       th in [0, 2 pi)

UPDATE. Checkerboard Metropolis: the lattice is split into two sublattices and one is updated at a time. A site's
neighbours all sit on the other sublattice, so updating a whole sublattice at once is exactly the same as updating its
sites one after another, and it is a legitimate Metropolis step (detailed balance holds) that numpy can do for
thousands of sites per call. L must therefore be even. A sweep is both sublattices. The XY proposal step is adapted
during thermalisation only (adapting while measuring would break detailed balance).

OBSERVABLES per site: e = E/N and the order parameter m (Ising |mean s|; Potts (q max_k f_k - 1)/(q - 1); XY |mean e^{i th}|).
    chi' = N (<m^2> - <|m|>^2) / T       C = N (<e^2> - <e>^2) / T^2       Binder  U4 = 1 - <m^4> / (k <m^2>^2)
with k = 3 for Ising and Potts and k = 2 for the two-component XY order parameter.

HONEST ERROR BARS. Successive Metropolis sweeps are correlated, so a plain standard error is too small by sqrt(2 tau).
Every mean carries tau_int (Sokal windowing) in its error, and chi', C and U4 are jackknife estimates over blocks
longer than tau. Near the critical point tau grows (critical slowing down, tau ~ L^z with z ~ 2.17 for Metropolis in
2D Ising), which the results show; a run much shorter than ~50 tau is flagged. A thermalisation-drift z-score
(first half vs second half) is reported for each point.

CRITICAL POINT. U4 curves for different L cross at T_c (corrections to scaling shift the crossing of small
lattices); `estimate_critical_point` averages the crossings of adjacent sizes. FINITE-SIZE SCALING then fits
chi'_max ~ L^(gamma/nu) and |m|(T_c) ~ L^(-beta/nu) by regression in log-log.

VERIFICATION against exact results:
  * `exact_ising_enumeration` sums all 2^16 states of the 4 x 4 Ising lattice, so <e>, <|m|>, <m^2>, <m^4>, C, chi'
    and U4 are known EXACTLY at any T and the Metropolis estimates must agree within their (tau-corrected) errors;
  * `torus_exact` is a transfer-matrix solution for the Potts/Ising model on a W x L torus (checked against the
    enumeration at q = 2) giving exact e and C for q = 3, 4 on 4 x 4;
  * the infinite-lattice references are Onsager's T_c = 2J / ln(1 + sqrt 2) and the Potts T_c = J / ln(1 + sqrt q)
    (exact on the square lattice); the XY (Kosterlitz-Thouless) T_c ~ 0.8929 J is a numerical literature value and
    is labelled as such.
"""
import copy
import math
from functools import lru_cache
from dataclasses import dataclass, field

import numpy as np

from modules import gpu
from modules.progress import ProgressFn, chunk_bounds, report
from modules.sm_common import StatMechError, integrated_autocorrelation_time, jackknife, mean_with_error, resolve_seed, spawn_rngs

MODELS = ("ising", "potts", "xy")
XY_TC_LITERATURE = 0.8929


@dataclass(frozen=True)
class LatticeModel:
    model: str
    L: int
    J: float = 1.0
    h: float = 0.0
    q: int = 3

    def __post_init__(self) -> None:
        if self.model not in MODELS:
            raise StatMechError(f"model must be one of {', '.join(MODELS)}.")
        if self.L < 4 or self.L > 256 or self.L % 2:
            raise StatMechError("L must be even and between 4 and 256 (the checkerboard update needs an even lattice).")
        if self.J <= 0:
            raise StatMechError("J must be positive (a ferromagnet).")
        if self.model == "potts" and not 2 <= self.q <= 8:
            raise StatMechError("The Potts model needs 2 <= q <= 8.")

    @property
    def N(self) -> int:
        return self.L * self.L

    @property
    def binder_k(self) -> int:
        return 2 if self.model == "xy" else 3


def critical_temperature(m: LatticeModel) -> tuple[float, str]:
    """(T_c in units of J, where it comes from)."""
    if m.model == "ising":
        return 2 * m.J / math.log(1 + math.sqrt(2)), "exact (Onsager)"
    if m.model == "potts":
        return m.J / math.log(1 + math.sqrt(m.q)), "exact (self-duality)"
    return XY_TC_LITERATURE * m.J, "numerical literature value (Kosterlitz–Thouless)"


# ------------------------------------------------------------------------------------------ state and updates

def initial_state(m: LatticeModel, rng: np.random.Generator, ordered: bool = False) -> np.ndarray:
    shape = (m.L, m.L)
    if m.model == "ising":
        return np.ones(shape, dtype=np.int8) if ordered else rng.choice(np.array([-1, 1], dtype=np.int8), size=shape)
    if m.model == "potts":
        return np.zeros(shape, dtype=np.int8) if ordered else rng.integers(0, m.q, size=shape).astype(np.int8)
    return np.zeros(shape) if ordered else rng.uniform(0, 2 * math.pi, size=shape)


@lru_cache(maxsize=32)
def _masks(L: int) -> tuple[np.ndarray, np.ndarray]:
    i, j = np.indices((L, L))
    even = (i + j) % 2 == 0
    return even, ~even


_DEVICE_MASKS: dict[tuple[str, int], tuple] = {}


def _masks_for(xp, L: int) -> tuple:
    """The two sublattice masks as arrays of `xp` (the cached NumPy ones, or a cached device copy)."""
    if xp is np:
        return _masks(L)
    key = (xp.__name__, L)
    if key not in _DEVICE_MASKS:
        _DEVICE_MASKS[key] = tuple(xp.asarray(m) for m in _masks(L))
    return _DEVICE_MASKS[key]


def _neighbours(a):
    xp = gpu.array_module(a)
    return [xp.roll(a, 1, 0), xp.roll(a, -1, 0), xp.roll(a, 1, 1), xp.roll(a, -1, 1)]


def sweep(m: LatticeModel, s, T: float, rng, step: float = 1.0) -> float:
    """One Metropolis sweep (both sublattices), in place. Returns the acceptance rate. `s` and `rng` are NumPy, or
    both CuPy (see modules/gpu.py): the code is the same for either."""
    xp = gpu.array_module(s)
    even, odd = _masks_for(xp, m.L)
    accepted = 0
    beta = 1.0 / T
    for mask in (even, odd):
        if m.model == "ising":
            nn = sum(n.astype(xp.int16) for n in _neighbours(s))
            dE = 2.0 * s * (m.J * nn + m.h)
            acc = (dE <= 0) | (rng.random(s.shape) < xp.exp(-beta * xp.maximum(dE, 0.0)))
            flip = acc & mask
            s[flip] = -s[flip]
        elif m.model == "potts":
            new = ((s + rng.integers(1, m.q, size=s.shape)) % m.q).astype(xp.int8)
            nbs = _neighbours(s)
            n_old = sum((n == s).astype(xp.int16) for n in nbs)
            n_new = sum((n == new).astype(xp.int16) for n in nbs)
            dE = -m.J * (n_new - n_old) - m.h * ((new == 0).astype(float) - (s == 0).astype(float))
            acc = (dE <= 0) | (rng.random(s.shape) < xp.exp(-beta * xp.maximum(dE, 0.0)))
            flip = acc & mask
            s[flip] = new[flip]
        else:
            trial = s + step * (2 * rng.random(s.shape) - 1)
            nbs = _neighbours(s)
            dE = -m.J * sum(xp.cos(trial - n) - xp.cos(s - n) for n in nbs) - m.h * (xp.cos(trial) - xp.cos(s))
            acc = (dE <= 0) | (rng.random(s.shape) < xp.exp(-beta * xp.maximum(dE, 0.0)))
            flip = acc & mask
            s[flip] = xp.mod(trial[flip], 2 * math.pi)
        accepted += int(flip.sum())
    return accepted / m.N


def energy_per_site(m: LatticeModel, s) -> float:
    xp = gpu.array_module(s)
    right, down = xp.roll(s, -1, 1), xp.roll(s, -1, 0)
    if m.model == "ising":
        return float(-m.J * ((s * right).sum() + (s * down).sum()) / m.N - m.h * s.mean())
    if m.model == "potts":
        return float(-m.J * ((s == right).sum() + (s == down).sum()) / m.N - m.h * (s == 0).mean())
    return float(-m.J * (xp.cos(s - right).sum() + xp.cos(s - down).sum()) / m.N - m.h * xp.cos(s).mean())


def order_parameter(m: LatticeModel, s) -> float:
    xp = gpu.array_module(s)
    if m.model == "ising":
        return float(abs(s.mean()))
    if m.model == "potts":
        frac = xp.bincount(s.ravel(), minlength=m.q) / m.N
        return float((m.q * frac.max() - 1) / (m.q - 1))
    return float(math.hypot(float(xp.cos(s).mean()), float(xp.sin(s).mean())))


# ------------------------------------------------------------------------------------------ runs

@dataclass
class PointResult:
    model: LatticeModel
    T: float
    n_meas: int
    e: float
    e_err: float
    m: float                    # <|m|>
    m_err: float
    m2: float
    m4: float
    chi: float
    chi_err: float
    C: float
    C_err: float
    U4: float
    U4_err: float
    tau_e: float
    tau_m: float
    acceptance: float
    drift_z: float              # (first half mean - second half mean) / error, for the order parameter
    seed: int = 0
    backend: str = "CPU (NumPy)"

    @property
    def run_long_enough(self) -> bool:
        return self.n_meas >= 50 * max(self.tau_e, self.tau_m)


def _adapt_step(accept: float, step: float) -> float:
    return float(min(math.pi, max(0.02, step * (1.2 if accept > 0.45 else 0.8 if accept < 0.35 else 1.0))))


def run_point(model: LatticeModel, T: float, rng: np.random.Generator, n_therm: int, n_meas: int, state: np.ndarray | None = None,
              ordered: bool = False, seed: int = 0, progress: ProgressFn | None = None, label: str = "") -> tuple[PointResult, np.ndarray]:
    """Thermalise for n_therm sweeps and measure for n_meas; returns the observables with errors and the final state
    (so a scan can continue from it). Large lattices run on the GPU when there is one (modules/gpu.py); the state
    that comes back is always a NumPy array."""
    if T <= 0:
        raise StatMechError("T must be positive.")
    if n_meas < 100 or n_therm < 0:
        raise StatMechError("Need at least 100 measurement sweeps.")
    backend = gpu.select(model.N, "lattice_sites")
    args = (model, T, n_therm, n_meas, state, ordered, seed, progress, label)

    def on_gpu():
        probe = copy.deepcopy(rng)                      # the caller's generator is left untouched, so a CPU fallback starts clean
        device_rng = backend.xp.random.default_rng(int(probe.integers(2 ** 62)))
        return _run_point(backend, device_rng, probe, *args)

    return gpu.attempt(backend, on_gpu, lambda: _run_point(gpu.CPU, rng, rng, *args), (StatMechError,))


def _run_point(backend: gpu.Backend, rng, host_rng: np.random.Generator, model: LatticeModel, T: float, n_therm: int, n_meas: int,
               state: np.ndarray | None, ordered: bool, seed: int, progress: ProgressFn | None, label: str) -> tuple[PointResult, np.ndarray]:
    s = gpu.on_device(backend, initial_state(model, host_rng, ordered) if state is None else state)
    step = 1.0 if model.model != "xy" else min(math.pi, 1.0 + 2 * math.sqrt(T))
    for a, b in chunk_bounds(n_therm, 200):
        report(progress, f"{label}thermalising", None)
        acc = 0.0
        for _ in range(b - a):
            acc = sweep(model, s, T, rng, step)
        if model.model == "xy":
            step = _adapt_step(acc, step)
    e_ser = np.empty(n_meas)
    m_ser = np.empty(n_meas)
    acc_sum = 0.0
    for a, b in chunk_bounds(n_meas, 200):
        report(progress, f"{label}measuring", a / n_meas)
        for k in range(a, b):
            acc_sum += sweep(model, s, T, rng, step)
            e_ser[k] = energy_per_site(model, s)
            m_ser[k] = order_parameter(model, s)
    point = _analyse(model, T, e_ser, m_ser, acc_sum / n_meas, seed)
    point.backend = backend.name
    return point, np.array(gpu.to_host(s))


def _analyse(model: LatticeModel, T: float, e: np.ndarray, m: np.ndarray, acc: float, seed: int) -> PointResult:
    N, k = model.N, model.binder_k
    e_mean, e_err, tau_e = mean_with_error(e)
    m_mean, m_err, tau_m = mean_with_error(m)
    series = {"e": e, "e2": e ** 2, "m": m, "m2": m ** 2, "m4": m ** 4}
    n_blocks = int(max(8, min(32, len(e) // max(1.0, 4 * max(tau_e, tau_m)) // 1)))
    n_blocks = min(n_blocks, 32)
    chi, chi_err = jackknife(series, lambda d: N * (d["m2"] - d["m"] ** 2) / T, n_blocks)
    C, C_err = jackknife(series, lambda d: N * (d["e2"] - d["e"] ** 2) / T ** 2, n_blocks)
    U4, U4_err = jackknife(series, lambda d: 1 - d["m4"] / (k * d["m2"] ** 2) if d["m2"] > 0 else 0.0, n_blocks)
    half = len(m) // 2
    m1, m2_ = m[:half], m[half:]
    se = math.sqrt(2 * tau_m * (m1.var(ddof=1) + m2_.var(ddof=1)) / max(half, 1)) or 1e-300
    drift = float((m1.mean() - m2_.mean()) / se)
    return PointResult(model, T, len(e), e_mean, e_err, m_mean, m_err, float((m ** 2).mean()), float((m ** 4).mean()), chi, chi_err,
                       C, C_err, U4, U4_err, tau_e, tau_m, acc, drift, seed)


@dataclass
class ScanResult:
    model: LatticeModel
    temps: np.ndarray
    points: list[PointResult]
    seed: int

    def column(self, name: str) -> np.ndarray:
        return np.array([getattr(p, name) for p in self.points])

    @property
    def backend(self) -> str:
        """The backend the points were computed on (the same for every point of a scan)."""
        return self.points[0].backend if self.points else "CPU (NumPy)"


def scan_temperatures(model: LatticeModel, temps: list[float] | np.ndarray, n_therm: int = 500, n_meas: int = 4000,
                      seed: int | None = None, progress: ProgressFn | None = None) -> ScanResult:
    """Observables over a list of temperatures, annealing from the highest to the lowest (each point starts from the
    previous configuration and then thermalises again). Each temperature has its own generator spawned from the seed,
    so a point is reproducible on its own and independent of the others."""
    t_sorted = np.array(sorted({float(t) for t in temps}, reverse=True))
    if len(t_sorted) < 1 or np.any(t_sorted <= 0):
        raise StatMechError("Give at least one positive temperature.")
    seed = resolve_seed(seed)
    rngs = spawn_rngs(seed, len(t_sorted))
    state: np.ndarray | None = None
    pts: list[PointResult] = []
    for i, (t, rng) in enumerate(zip(t_sorted, rngs)):
        sub: ProgressFn | None = None
        if progress is not None:
            def sub(lbl: str, fr: float | None, i: int = i, t: float = t) -> None:       # noqa: E306
                progress(f"L = {model.L}, T = {t:.4g}: {lbl}", (i + (fr or 0.0)) / len(t_sorted))
        pt, state = run_point(model, float(t), rng, n_therm if i else 3 * n_therm, n_meas, state, False, seed, sub)
        pts.append(pt)
    pts.reverse()
    return ScanResult(model, t_sorted[::-1].copy(), pts, seed)


@dataclass
class SingleRun:
    model: LatticeModel
    T: float
    energy: np.ndarray
    order: np.ndarray
    snapshots: list[tuple[int, np.ndarray]]
    acceptance: float
    seed: int
    backend: str = "CPU (NumPy)"           # which backend produced the run (see modules/gpu.py)


def run_single(model: LatticeModel, T: float, n_sweeps: int, snapshot_every: int, seed: int | None = None, ordered: bool = False,
               progress: ProgressFn | None = None) -> SingleRun:
    """One run at fixed T, recording the lattice every `snapshot_every` sweeps (for the animation)."""
    if n_sweeps < 1 or snapshot_every < 1:
        raise StatMechError("Need at least one sweep and a positive snapshot interval.")
    seed = resolve_seed(seed)
    backend = gpu.select(model.N, "lattice_sites")
    args = (model, T, n_sweeps, snapshot_every, seed, ordered, progress)

    def on_gpu():
        probe = np.random.default_rng(seed)
        device_rng = backend.xp.random.default_rng(int(probe.integers(2 ** 62)))
        return _run_single(backend, device_rng, probe, *args)

    return gpu.attempt(backend, on_gpu, lambda: (lambda r: _run_single(gpu.CPU, r, r, *args))(np.random.default_rng(seed)),
                       (StatMechError,))


def _run_single(backend: gpu.Backend, rng, host_rng: np.random.Generator, model: LatticeModel, T: float, n_sweeps: int,
                snapshot_every: int, seed: int, ordered: bool, progress: ProgressFn | None) -> SingleRun:
    s = gpu.on_device(backend, initial_state(model, host_rng, ordered))
    step = 1.0 if model.model != "xy" else min(math.pi, 1.0 + 2 * math.sqrt(T))
    e, o = np.empty(n_sweeps), np.empty(n_sweeps)
    snaps: list[tuple[int, np.ndarray]] = [(0, np.array(gpu.to_host(s)))]
    acc_sum = 0.0
    for a, b in chunk_bounds(n_sweeps, 100):
        report(progress, "Sweeping", a / n_sweeps)
        for k in range(a, b):
            acc = sweep(model, s, T, rng, step)
            acc_sum += acc
            if model.model == "xy" and k < n_sweeps // 4:
                step = _adapt_step(acc, step)
            e[k], o[k] = energy_per_site(model, s), order_parameter(model, s)
            if (k + 1) % snapshot_every == 0:
                snaps.append((k + 1, np.array(gpu.to_host(s))))
    return SingleRun(model, T, e, o, snaps, acc_sum / n_sweeps, seed, backend.name)


# ------------------------------------------------------------------------------------------ critical point, FSS

def _crossing(temps: np.ndarray, a: np.ndarray, ea: np.ndarray, b: np.ndarray, eb: np.ndarray) -> tuple[float, float] | None:
    d = a - b
    idx = [i for i in range(len(d) - 1) if d[i] * d[i + 1] < 0]
    if not idx:
        return None
    mid = len(d) / 2
    i = min(idx, key=lambda k: abs(k - mid))
    t0, t1 = temps[i], temps[i + 1]
    frac = d[i] / (d[i] - d[i + 1])
    tc = float(t0 + frac * (t1 - t0))
    slope = abs(d[i + 1] - d[i]) / (t1 - t0)
    sig = math.sqrt(ea[i] ** 2 + eb[i] ** 2 + ea[i + 1] ** 2 + eb[i + 1] ** 2) / 2
    return tc, float(sig / slope) if slope > 0 else float("inf")


@dataclass
class CriticalEstimate:
    pairs: list[tuple[int, int, float, float]]       # (L1, L2, T_cross, error)
    tc: float
    tc_err: float


def estimate_critical_point(scans: dict[int, ScanResult]) -> CriticalEstimate | None:
    """Mean Binder-cumulant crossing of adjacent sizes (all scans must share the same temperature grid)."""
    sizes = sorted(scans)
    pairs: list[tuple[int, int, float, float]] = []
    for l1, l2 in zip(sizes, sizes[1:]):
        a, b = scans[l1], scans[l2]
        if len(a.temps) != len(b.temps) or not np.allclose(a.temps, b.temps):
            raise StatMechError("All lattice sizes must be scanned over the same temperatures.")
        cr = _crossing(a.temps, a.column("U4"), a.column("U4_err"), b.column("U4"), b.column("U4_err"))
        if cr:
            pairs.append((l1, l2, cr[0], cr[1]))
    if not pairs:
        return None
    t = np.array([p[2] for p in pairs])
    spread = float(t.std(ddof=1)) if len(t) > 1 else 0.0
    stat = float(np.sqrt(np.mean([p[3] ** 2 for p in pairs])))
    return CriticalEstimate(pairs, float(t.mean()), float(max(spread, stat)))


@dataclass
class ScalingFit:
    gamma_over_nu: float | None
    gamma_err: float | None
    beta_over_nu: float | None
    beta_err: float | None
    chi_max: dict[int, float]
    m_at_tc: dict[int, float]


def _loglog(x: list[float], y: list[float]) -> tuple[float, float | None]:
    lx, ly = np.log(x), np.log(y)
    if len(x) == 2:
        return float((ly[1] - ly[0]) / (lx[1] - lx[0])), None
    p, cov = np.polyfit(lx, ly, 1, cov=True)
    return float(p[0]), float(math.sqrt(cov[0, 0]))


def finite_size_scaling(scans: dict[int, ScanResult], tc: float) -> ScalingFit:
    """gamma/nu from the peak of chi' and beta/nu from |m| at T_c (interpolated), by log-log regression over L."""
    sizes = sorted(scans)
    chi_max = {L: float(np.max(scans[L].column("chi"))) for L in sizes}
    m_tc = {L: float(np.interp(tc, scans[L].temps, scans[L].column("m"))) for L in sizes}
    if len(sizes) < 2:
        return ScalingFit(None, None, None, None, chi_max, m_tc)
    g, ge = _loglog([float(L) for L in sizes], [chi_max[L] for L in sizes])
    b, be = _loglog([float(L) for L in sizes], [m_tc[L] for L in sizes])
    return ScalingFit(g, ge, -b, be, chi_max, m_tc)


# ------------------------------------------------------------------------------------------ exact references

@dataclass
class ExactSmall:
    T: float
    e: float
    m_abs: float
    m2: float
    m4: float
    chi: float
    C: float
    U4: float


_ISING_CACHE: dict[int, tuple[np.ndarray, np.ndarray]] = {}


def _ising_states(L: int) -> tuple[np.ndarray, np.ndarray]:
    """Bond-sum (sum s_i s_j) and magnetisation sum for every state of an L x L periodic Ising lattice (L <= 4)."""
    if L not in _ISING_CACHE:
        if L > 4:
            raise StatMechError("Enumeration is only feasible up to 4 x 4.")
        n = L * L
        bits = ((np.arange(2 ** n)[:, None] >> np.arange(n)) & 1).astype(np.int8) * 2 - 1
        grid = bits.reshape(-1, L, L)
        bond = (grid * np.roll(grid, -1, 2)).sum(axis=(1, 2)) + (grid * np.roll(grid, -1, 1)).sum(axis=(1, 2))
        _ISING_CACHE[L] = (bond.astype(float), grid.sum(axis=(1, 2)).astype(float))
    return _ISING_CACHE[L]


def exact_ising_enumeration(L: int, T: float, J: float = 1.0, h: float = 0.0) -> ExactSmall:
    """Exact thermodynamics of the L x L periodic Ising lattice by summing every state (L = 4: 65536 states)."""
    bond, mag = _ising_states(L)
    N = L * L
    E = -J * bond - h * mag
    w = np.exp(-(E - E.min()) / T)
    p = w / w.sum()
    m = mag / N
    e1, e2 = float((p * E).sum()) / N, float((p * E ** 2).sum()) / N ** 2
    ma, m2, m4 = float((p * np.abs(m)).sum()), float((p * m ** 2).sum()), float((p * m ** 4).sum())
    return ExactSmall(T, e1, ma, m2, m4, N * (m2 - ma ** 2) / T, N * (e2 - e1 ** 2) / T ** 2, 1 - m4 / (3 * m2 ** 2))


def _transfer_lnZ(model: str, W: int, L: int, q: int, beta: float, J: float, h: float) -> float:
    """ln Z of the Potts (or Ising, as q = 2 with J_potts = 2 J_ising) model on a W x L torus by a column transfer matrix."""
    n_states = q ** W
    states = np.array(np.unravel_index(np.arange(n_states), (q,) * W)).T            # (n_states, W)
    ring = (states == np.roll(states, -1, axis=1)).sum(axis=1)
    field = (states == 0).sum(axis=1)
    d = np.exp(beta * (J * ring + h * field) / 2)                                   # half of the column's own weight at each end
    link = np.exp(beta * J * (states[:, None, :] == states[None, :, :]).sum(axis=2))
    t = d[:, None] * link * d[None, :]
    ev = np.linalg.eigvalsh(t)
    top = ev.max()
    return float(L * math.log(top) + math.log(float(((ev / top) ** L).sum())))


def torus_exact(q: int, W: int, L: int, T: float, J: float = 1.0, h: float = 0.0) -> tuple[float, float]:
    """(e, C) per site of the q-state Potts model on a W x L torus (q = 2 is Ising with J_Ising = J_Potts / 2 plus a
    constant), from -d ln Z / d beta and beta^2 d^2 ln Z / d beta^2 by central differences of the exact ln Z."""
    if q < 2 or W < 2 or L < 2 or q ** W > 4000:
        raise StatMechError("Need q >= 2, W, L >= 2 and q^W <= 4000.")
    beta, d = 1.0 / T, 1e-4 / T
    f = lambda b: _transfer_lnZ("potts", W, L, q, b, J, h)  # noqa: E731
    n = W * L
    e = -(f(beta + d) - f(beta - d)) / (2 * d) / n
    c = beta ** 2 * (f(beta + d) - 2 * f(beta) + f(beta - d)) / d ** 2 / n
    return float(e), float(c)


def compare_with_exact_ising(point: PointResult, n_sigma: float = 4.0) -> list[tuple[str, float, float, float, bool]]:
    """(name, Monte Carlo value, error, exact value, within n_sigma?) for a 4 x 4 Ising point."""
    m = point.model
    if m.model != "ising" or m.L != 4:
        raise StatMechError("Exact comparison is available for the 4 x 4 Ising lattice.")
    ex = exact_ising_enumeration(4, point.T, m.J, m.h)
    rows = []
    for name, val, err, ref in (("e", point.e, point.e_err, ex.e), ("<|m|>", point.m, point.m_err, ex.m_abs), ("χ'", point.chi, point.chi_err, ex.chi),
                                ("C", point.C, point.C_err, ex.C), ("U4", point.U4, point.U4_err, ex.U4)):
        rows.append((name, val, err, ref, abs(val - ref) <= n_sigma * max(err, 1e-12)))
    return rows
