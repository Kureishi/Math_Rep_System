"""
Random matrices and spectra -- Streamlit-free.

ENSEMBLES (scaled so the spectrum fills [-2, 2], <lambda^2> = 1):
    GOE   real symmetric:        H = (G + G^T) / sqrt(2N),  G_ij ~ N(0, 1)       off-diagonal variance 1/N, diagonal 2/N
    GUE   complex Hermitian:     H = (G + G^+) / sqrt(2N),  G_ij ~ (N(0,1) + i N(0,1)) / sqrt 2     variance 1/N everywhere
    Poisson   independent eigenvalues drawn from the semicircle law: the "integrable / uncorrelated" benchmark
    Wishart   (1/n) X X^T with X a p x n Gaussian matrix, c = p/n: Marchenko-Pastur density on [(1-sqrt c)^2, (1+sqrt c)^2]

DENSITY. Wigner's semicircle  rho(x) = sqrt(4 - x^2)/(2 pi). Its even moments are the CATALAN numbers 1, 2, 5, 14, ...;
the second moment is checked against its exact finite-N value (GOE: 1 + 1/N, since tr H^2 = N + 1 on average; GUE: exactly 1),
which also tests the normalisation of the ensembles.

LEVEL SPACINGS. The eigenvalues are UNFOLDED -- mapped through the smooth cumulative density N F(lambda), F the semicircle CDF --
so that the mean spacing is 1 everywhere, and only the central 80% of the spectrum is used (the edges are where the density
changes fastest). The nearest-neighbour spacing distribution is then compared with
    GOE surmise     p(s) = (pi/2) s exp(-pi s^2/4)           (level repulsion linear in s,   beta = 1)
    GUE surmise     p(s) = (32/pi^2) s^2 exp(-4 s^2/pi)       (quadratic repulsion,           beta = 2)
    Poisson         p(s) = exp(-s)                            (no repulsion)
by the Kolmogorov-Smirnov distance to each CDF, and the spectrum is CLASSIFIED as the nearest. The surmise is derived from the
2 x 2 problem, where it is EXACT: spacings of 2 x 2 GOE and GUE matrices are tested against it with a proper KS test. For large N
the surmise is accurate to <~0.5% but not exact, which is why the N x N check is a distance ranking, not a p-value.
The ratio statistic r~ = min(s_i, s_i+1)/max(s_i, s_i+1) needs no unfolding; Poisson: 2 ln 2 - 1 = 0.3863 (exact), GOE 0.5307 and
GUE 0.5996 are the high-precision numerical values of Atas et al. (2013) and are labelled as literature values.
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.stats import kstest

from modules.quantum_common import CheckList
from modules import gpu
from modules.sm_common import StatMechError, resolve_seed, spawn_rngs

KINDS = ("GOE", "GUE", "Poisson")
RATIO_LITERATURE = {"GOE": 0.5307, "GUE": 0.5996, "Poisson": 2 * math.log(2) - 1}
CATALAN = {2: 1, 4: 2, 6: 5, 8: 14}


# ------------------------------------------------------------------------------------------ the ensembles

def sample_matrix(kind: str, N: int, rng: np.random.Generator) -> np.ndarray:
    if kind == "GOE":
        g = rng.standard_normal((N, N))
        return (g + g.T) / math.sqrt(2 * N)
    if kind == "GUE":
        g = (rng.standard_normal((N, N)) + 1j * rng.standard_normal((N, N))) / math.sqrt(2)
        return (g + g.conj().T) / math.sqrt(2 * N)
    raise StatMechError("kind must be GOE or GUE for a matrix sample.")


def semicircle_pdf(x: np.ndarray | float) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) < 2, np.sqrt(np.clip(4 - x ** 2, 0, None)) / (2 * math.pi), 0.0)


def semicircle_cdf(x: np.ndarray | float) -> np.ndarray:
    x = np.clip(np.asarray(x, dtype=float), -2, 2)
    return 0.5 + x * np.sqrt(4 - x ** 2) / (4 * math.pi) + np.arcsin(x / 2) / math.pi


def _semicircle_sample(n: int, rng: np.random.Generator) -> np.ndarray:
    """Inverse-CDF sampling of the semicircle law (the CDF is monotone, so interpolate it on a fine grid)."""
    grid = np.linspace(-2, 2, 40001)
    return np.interp(rng.random(n), semicircle_cdf(grid), grid)


_BATCH_BYTES = 256 * 2 ** 20       # how much matrix data one GPU batch may hold


def _batched_eigvalsh(backend: gpu.Backend, make, n_samples: int, bytes_each: int) -> np.ndarray:
    """Eigenvalues of `n_samples` matrices produced by make(i) on the CPU (so the random numbers are the CPU's, whatever
    the backend), solved in batches on the device. Each row comes back sorted ascending."""
    per = max(1, _BATCH_BYTES // max(1, bytes_each))
    rows = []
    for lo in range(0, n_samples, per):
        stack = backend.xp.asarray(np.stack([make(i) for i in range(lo, min(lo + per, n_samples))]))
        rows.append(gpu.to_host(backend.xp.linalg.eigvalsh(stack)))
    return np.concatenate(rows)


def spectra(kind: str, N: int, n_samples: int, seed: int | None = None) -> tuple[np.ndarray, int]:
    """(n_samples, N) sorted eigenvalues of independent matrices, and the seed used. Large matrices have their
    eigenvalues found on the GPU when there is one (the matrices themselves are the CPU's, so the spectra do not depend
    on the backend)."""
    if kind not in KINDS:
        raise StatMechError(f"kind must be one of {', '.join(KINDS)}.")
    if N < 4 or n_samples < 1 or N * N * n_samples > 4e8:
        raise StatMechError("Need N >= 4 and N^2 x samples <= 4e8.")
    seed = resolve_seed(seed)
    rngs = spawn_rngs(seed, n_samples)
    out = np.empty((n_samples, N))
    if kind != "Poisson":
        backend = gpu.select(N, "matrix_dim")
        bytes_each = N * N * (16 if kind == "GUE" else 8)

        def cpu() -> np.ndarray:
            for i, rng in enumerate(rngs):
                out[i] = np.linalg.eigvalsh(sample_matrix(kind, N, rng))
            return out

        return gpu.attempt(backend, lambda: _batched_eigvalsh(backend, lambda i: sample_matrix(kind, N, rngs[i]), n_samples, bytes_each),
                           cpu), seed
    for i, rng in enumerate(rngs):
        out[i] = np.sort(_semicircle_sample(N, rng))
    return out, seed


def wishart_spectra(p: int, n: int, n_samples: int, seed: int | None = None) -> tuple[np.ndarray, int]:
    if p < 4 or n < p or n_samples < 1:
        raise StatMechError("Need p >= 4 and n >= p (so that c = p/n <= 1).")
    seed = resolve_seed(seed)
    rngs = spawn_rngs(seed, n_samples)
    backend = gpu.select(p, "matrix_dim")

    def on_gpu() -> np.ndarray:
        xp = backend.xp
        per = max(1, _BATCH_BYTES // max(1, p * n * 8))
        rows = []
        for lo in range(0, n_samples, per):
            x = xp.asarray(np.stack([rngs[i].standard_normal((p, n)) for i in range(lo, min(lo + per, n_samples))]))
            rows.append(gpu.to_host(xp.linalg.eigvalsh(xp.matmul(x, x.transpose(0, 2, 1)) / n)))
        return np.concatenate(rows)

    def cpu() -> np.ndarray:
        out = np.empty((n_samples, p))
        for i, rng in enumerate(rngs):
            x = rng.standard_normal((p, n))
            out[i] = np.linalg.eigvalsh(x @ x.T / n)
        return out

    return gpu.attempt(backend, on_gpu, cpu), seed


def marchenko_pastur_pdf(x: np.ndarray, c: float) -> np.ndarray:
    lo, hi = (1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2
    x = np.asarray(x, dtype=float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where((x > lo) & (x < hi), np.sqrt(np.clip((hi - x) * (x - lo), 0, None)) / (2 * math.pi * c * x), 0.0)


def marchenko_pastur_cdf(x: np.ndarray, c: float) -> np.ndarray:
    lo, hi = (1 - math.sqrt(c)) ** 2, (1 + math.sqrt(c)) ** 2
    grid = np.linspace(lo, hi, 20001)
    cum = np.concatenate([[0.0], cumulative_trapezoid(marchenko_pastur_pdf(grid, c)[1:], grid[1:], initial=0.0)])
    cum /= cum[-1]
    return np.interp(np.asarray(x, dtype=float), grid, cum, left=0.0, right=1.0)


# ------------------------------------------------------------------------------------------ spacings

def unfolded_spacings(eigs: np.ndarray, trim: float = 0.1) -> np.ndarray:
    """Nearest-neighbour spacings of the unfolded central eigenvalues of every sample, normalised to mean 1."""
    n = eigs.shape[1]
    lo, hi = int(trim * n), int((1 - trim) * n)
    x = n * semicircle_cdf(eigs)
    s = np.diff(x[:, lo:hi], axis=1).ravel()
    return s / s.mean()


def surmise_pdf(kind: str, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if kind == "GOE":
        return math.pi / 2 * s * np.exp(-math.pi * s ** 2 / 4)
    if kind == "GUE":
        return 32 / math.pi ** 2 * s ** 2 * np.exp(-4 * s ** 2 / math.pi)
    if kind == "Poisson":
        return np.exp(-s)
    raise StatMechError(f"kind must be one of {', '.join(KINDS)}.")


def surmise_cdf(kind: str, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    if kind == "GOE":
        return 1 - np.exp(-math.pi * s ** 2 / 4)
    if kind == "Poisson":
        return 1 - np.exp(-s)
    grid = np.linspace(0, 12, 24001)                            # GUE: integrate the density numerically
    cum = cumulative_trapezoid(surmise_pdf("GUE", grid), grid, initial=0.0)
    return np.interp(s, grid, cum / cum[-1])


def spacing_distances(spacings: np.ndarray) -> dict[str, float]:
    """Kolmogorov-Smirnov distance from the sample to each reference law."""
    return {k: float(kstest(spacings, lambda s, k=k: surmise_cdf(k, s)).statistic) for k in KINDS}


def classify(spacings: np.ndarray) -> tuple[str, dict[str, float]]:
    d = spacing_distances(spacings)
    return min(d, key=lambda k: d[k]), d


def ratio_statistic(eigs: np.ndarray) -> float:
    """<r~>, r~_i = min(s_i, s_{i+1}) / max(s_i, s_{i+1}), from the raw (not unfolded) central spacings."""
    n = eigs.shape[1]
    lo, hi = int(0.1 * n), int(0.9 * n)
    s = np.diff(eigs[:, lo:hi], axis=1)
    a, b = s[:, :-1], s[:, 1:]
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.minimum(a, b) / np.maximum(a, b)
    return float(np.nanmean(r))


def two_by_two_spacings(kind: str, n: int, seed: int | None = None) -> tuple[np.ndarray, int]:
    """Spacings of n random 2x2 GOE / GUE matrices, scaled to mean 1: exactly Wigner's surmise."""
    if kind not in ("GOE", "GUE"):
        raise StatMechError("kind must be GOE or GUE.")
    seed = resolve_seed(seed)
    rng = np.random.default_rng(seed)
    a, c = rng.standard_normal(n), rng.standard_normal(n)
    b2 = 0.5 * rng.standard_normal(n) ** 2 if kind == "GOE" else 0.5 * (rng.standard_normal(n) ** 2 + rng.standard_normal(n) ** 2)
    s = np.sqrt((a - c) ** 2 + 4 * b2)
    return s / s.mean(), seed


def empirical_moments(eigs: np.ndarray, orders: tuple[int, ...] = (2, 4, 6, 8)) -> dict[int, tuple[float, float]]:
    """(mean, standard error over matrices) of (1/N) tr H^k."""
    out = {}
    for k in orders:
        per = (eigs ** k).mean(axis=1)
        out[k] = (float(per.mean()), float(per.std(ddof=1) / math.sqrt(len(per))) if len(per) > 1 else float("nan"))
    return out


# ------------------------------------------------------------------------------------------ the full verification

@dataclass
class RandomMatrixReport:
    kind: str
    N: int
    eigenvalues: np.ndarray
    spacings: np.ndarray
    distances: dict[str, float]
    nearest: str
    ratio: float
    moments: dict[int, tuple[float, float]]
    seed: int
    checks: CheckList = field(default_factory=CheckList)
    backend: str = "CPU (NumPy)"           # where the eigenvalues were computed


def analyse(kind: str, N: int, n_samples: int, seed: int | None = None) -> RandomMatrixReport:
    gpu.reset_last()
    eigs, seed = spectra(kind, N, n_samples, seed)
    used = gpu.last_used() or gpu.CPU.name
    sp = unfolded_spacings(eigs)
    nearest, dist = classify(sp)
    ratio = ratio_statistic(eigs)
    mom = empirical_moments(eigs)
    rep = RandomMatrixReport(kind, N, eigs, sp, dist, nearest, ratio, mom, seed, backend=used)
    ck = rep.checks
    pooled = np.sort(eigs.ravel())
    ks = kstest(pooled, lambda x: semicircle_cdf(x))
    ck.add("eigenvalue density = Wigner semicircle (KS distance of the pooled eigenvalues)", ks.statistic < 0.02 + 2.0 / N ** (2 / 3) / 10 + 1.0 / N,
           f"KS distance {ks.statistic:.4f} (finite-N edge effects ~ N^(-2/3))")
    want2 = {"GOE": 1 + 1 / N, "GUE": 1.0, "Poisson": 1.0}[kind]
    m2, e2 = mom[2]
    ck.add(f"⟨λ²⟩ = {'1 + 1/N' if kind == 'GOE' else '1'} (exact finite-N value)", abs(m2 - want2) < 5 * e2 + 1e-3 * want2 + 2.0 / N ** 2, f"{m2:.5f} ± {e2:.1e} vs {want2:.5f}")
    for k in (4, 6):
        mk, ek = mom[k]
        tol = 5 * ek + 6.0 * CATALAN[k] / N     # the leading finite-N correction is O(1/N)
        ck.add(f"⟨λ^{k}⟩ = Catalan number {CATALAN[k]} (semicircle moment)", abs(mk - CATALAN[k]) < tol, f"{mk:.4f} ± {ek:.1e} vs {CATALAN[k]}")
    ck.add(f"level spacings are classified as {kind}", nearest == kind, "KS distances: " + ", ".join(f"{k} {v:.4f}" for k, v in dist.items()))
    ref = RATIO_LITERATURE[kind]
    ck.add(f"⟨r~⟩ = {ref:.4f}" + (" (exact)" if kind == "Poisson" else " (literature)"), abs(ratio - ref) < 0.012, f"measured {ratio:.4f}")
    return rep


def verify_surmise_exact(n: int = 200_000, seed: int | None = None) -> tuple[CheckList, int]:
    """2x2 spacings are exactly Wigner's surmise: a proper KS test (p-value) applies."""
    seed = resolve_seed(seed)
    ck = CheckList()
    for i, kind in enumerate(("GOE", "GUE")):
        s, _ = two_by_two_spacings(kind, n, seed + i)
        res = kstest(s, lambda x, kind=kind: surmise_cdf(kind, x))
        ck.add(f"2×2 {kind}: spacings are exactly the Wigner surmise (KS test)", res.pvalue > 1e-3, f"KS statistic {res.statistic:.2e}, p = {res.pvalue:.2g}")
    return ck, seed


def verify_wishart(p: int = 200, n: int = 400, n_samples: int = 20, seed: int | None = None) -> tuple[CheckList, np.ndarray, int]:
    eigs, seed = wishart_spectra(p, n, n_samples, seed)
    c = p / n
    ck = CheckList()
    ck.add("Marchenko–Pastur: mean eigenvalue = 1", abs(float(eigs.mean()) - 1) < 5 * float(eigs.mean(axis=1).std(ddof=1)) / math.sqrt(n_samples) + 1e-3,
           f"{float(eigs.mean()):.5f}")
    var = float(eigs.var())
    ck.add("Marchenko–Pastur: variance = c", abs(var - c) < 0.02, f"{var:.4f} vs c = {c:.4f}")
    ks = kstest(np.sort(eigs.ravel()), lambda x: marchenko_pastur_cdf(x, c))
    ck.add("eigenvalue density = Marchenko–Pastur law (KS distance)", ks.statistic < 0.03, f"KS distance {ks.statistic:.4f}")
    return ck, eigs, seed
