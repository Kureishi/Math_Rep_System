"""
Shared pieces of the statistical-mechanics tools (modules/sm_*.py) -- Streamlit-free.

  * StatMechError          -- the one exception type for unusable input (messages say what to change);
  * resolve_seed           -- the project's seed convention: seed=None means "pick a fresh seed", never "no seed";
                              the concrete seed actually used is returned and stored in every result, so any run
                              can be reproduced by passing it back in (see modules/monte_carlo.py);
  * integrated_autocorrelation_time, mean_with_error, jackknife
                           -- HONEST error bars for correlated Monte Carlo series. A Markov chain's successive
                              samples are not independent; the naive standard error sqrt(var/n) is too small by a
                              factor sqrt(2 tau), tau being the integrated autocorrelation time. tau is estimated
                              with Sokal's automatic windowing, and derived quantities (susceptibility, specific
                              heat, Binder cumulant) get jackknife errors over blocks longer than tau.

Units everywhere: k_B = 1 and energies in units of the coupling unless a function says otherwise (the partition-
function engine takes k_B explicitly; the quantum-gas tools use hbar = m = k_B = 1).
"""
from collections.abc import Callable, Sequence

import numpy as np


class StatMechError(ValueError):
    """The input is not usable (the message says why, in terms the user can act on)."""


def resolve_seed(seed: int | None) -> int:
    """A concrete 32-bit seed: the given one, or a fresh one from OS entropy."""
    if seed is None:
        return int(np.random.SeedSequence().generate_state(1, dtype=np.uint32)[0])
    if seed < 0:
        raise StatMechError("The seed must be a non-negative integer.")
    return int(seed)


def spawn_rngs(seed: int, n: int) -> list[np.random.Generator]:
    """n statistically independent generators derived from one seed (so a scan over temperatures is reproducible
    point by point and does not depend on the order or number of the other points)."""
    return [np.random.default_rng(s) for s in np.random.SeedSequence(seed).spawn(n)]


def integrated_autocorrelation_time(x: Sequence[float] | np.ndarray, c: float = 6.0) -> float:
    """tau_int = 1/2 + sum_{t=1}^{W} rho(t), the window W chosen automatically (Sokal): the smallest W with
    W >= c * tau_int(W). Returns 0.5 for an uncorrelated series. A series shorter than ~50 samples, or constant,
    gives 0.5 (nothing can be said)."""
    a = np.asarray(x, dtype=float)
    n = len(a)
    if n < 50 or np.ptp(a) == 0:
        return 0.5
    a = a - a.mean()
    f = np.fft.rfft(a, 2 * n)
    acf = np.fft.irfft(f * np.conj(f))[:n]
    acf = acf / acf[0]
    tau = np.cumsum(acf) - 0.5                       # tau_int(W) for W = 0, 1, 2, ...  (acf[0] = 1)
    windows = np.arange(n)
    ok = np.nonzero(windows >= c * tau)[0]
    w = int(ok[0]) if len(ok) else n - 1
    return float(max(tau[w], 0.5))


def mean_with_error(x: Sequence[float] | np.ndarray) -> tuple[float, float, float]:
    """(mean, standard error, tau_int) of a correlated series: error = sqrt(2 tau var / n)."""
    a = np.asarray(x, dtype=float)
    tau = integrated_autocorrelation_time(a)
    n = len(a)
    err = float(np.sqrt(2 * tau * a.var(ddof=1) / n)) if n > 1 else float("nan")
    return float(a.mean()), err, tau


def jackknife(series: dict[str, np.ndarray], func: Callable[[dict[str, float]], float], n_blocks: int = 32) -> tuple[float, float]:
    """(estimate, jackknife error) of func(means of the series). The series are cut into n_blocks equal blocks; the
    caller should choose runs long enough that a block is much longer than tau (the lattice code does)."""
    names = list(series)
    n = min(len(v) for v in series.values())
    nb = max(2, min(n_blocks, n))
    size = n // nb
    if size == 0:
        raise StatMechError("Too few samples for an error estimate.")
    sums = {k: np.asarray(series[k][: size * nb], dtype=float).reshape(nb, size).sum(axis=1) for k in names}
    total = {k: sums[k].sum() for k in names}
    full = func({k: total[k] / (size * nb) for k in names})
    leave = np.array([func({k: (total[k] - sums[k][b]) / (size * (nb - 1)) for k in names}) for b in range(nb)])
    err = float(np.sqrt((nb - 1) / nb * np.sum((leave - leave.mean()) ** 2)))
    return float(full), err
