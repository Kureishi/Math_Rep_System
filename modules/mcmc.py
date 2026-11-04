"""
Posterior sampling for nonlinear fits (MCMC) -- Streamlit-free, no LLM.

Least squares gives one best-fit curve and, at best, a Gaussian "+/-" around it. For a nonlinear model the real uncertainty
is often lopsided or curved (two parameters that trade off against each other; a rate that is well-determined only from
above), and a +/- hides that. Here the whole POSTERIOR is sampled: every parameter combination that is compatible with the
data, weighted by how compatible. Its marginals give honest credible intervals and its pairwise plots (the "corner plot")
show the correlations and the shape.

THE MODEL. y_i = f(x_i; theta) + Gaussian noise with standard deviation sigma. sigma is either given (a number, or one per
point) or, by default, SAMPLED as one more unknown (flat prior on log sigma). Priors on theta are flat (the default, which
is improper but fine when the data constrain the parameters), bounded (uniform), normal, or positive.

THE SAMPLER. The affine-invariant ensemble sampler of Goodman and Weare (the "stretch move", the algorithm behind emcee):
many walkers, each proposing a move along the line to another walker, so it is unaffected by how stretched or correlated the
posterior is and needs no step-size tuning. It is vectorised: all walkers of a half-ensemble are evaluated in one NumPy call.
It starts from a nonlinear least-squares fit, with the walkers in a small ball around it.

VERIFICATION-FIRST: a sample is not an answer until it is known to have converged, so every run reports
  * the acceptance fraction (healthy: roughly 0.2 - 0.5);
  * the integrated autocorrelation time tau of each parameter and the effective sample size it implies. The run is
    "converged" only if it is at least 50 tau long (the usual rule of thumb) AND split R-hat < 1.1 AND tau itself could be
    estimated (a chain too short to estimate tau is not converged, however good it looks);
  * two checks that the MODEL describes the data (the sampler will happily give tight intervals for a wrong model):
      - residual structure: the lag-1 correlation of the residuals taken in order of x. Independent noise leaves none; a
        wrong model leaves a systematic pattern (a line through a sine: 0.97; the well-specified examples: |z| < 1).
        This works whether sigma is sampled or given;
      - a posterior-predictive check: the share of the data inside the 95% predictive band (about 95% is right). This one
        only has teeth when sigma is GIVEN: when sigma is sampled it simply grows to absorb the misfit and the band covers
        the data by construction, which is exactly why the residual check exists;
  * the comparison with the least-squares answer, and which parameters are visibly non-Gaussian.
The sampler's correctness is tested against a case with a known exact answer (a linear model, whose posterior is Gaussian).

LIMITS, STATED. Gaussian independent noise is assumed (correlated or heavy-tailed noise needs a different likelihood);
multimodal posteriors can defeat a single ensemble started from one fit (the diagnostics may not notice a missed mode);
split R-hat on ensemble walkers is a mixing check, not a proof.
"""
import keyword
import math
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.optimize import least_squares

MAX_PARAMS = 8
MAX_STEPS = 20_000
MAX_WALKERS = 256
MIN_POINTS = 4
TAU_C = 5.0              # window constant of the automated autocorrelation windowing (Sokal)
MIN_TAU_RATIO = 10       # a tau estimate needs a chain at least this many taus long to be trusted at all
RESIDUAL_Z_LIMIT = 3.0
CONVERGED_STEPS_PER_TAU = 50
RHAT_LIMIT = 1.1
SIGMA = "sigma"


class McmcError(ValueError):
    """Bad input, or a start that cannot be found."""


# ------------------------------------------------------------------------------------------ priors

@dataclass(frozen=True)
class Prior:
    kind: str = "flat"                 # flat | uniform | normal | positive
    a: float = 0.0
    b: float = 0.0

    def validate(self, name: str) -> None:
        if self.kind not in ("flat", "uniform", "normal", "positive"):
            raise McmcError(f"Unknown prior '{self.kind}' for {name}.")
        if self.kind == "uniform" and not self.a < self.b:
            raise McmcError(f"The uniform prior for {name} needs lower < upper.")
        if self.kind == "normal" and not self.b > 0:
            raise McmcError(f"The normal prior for {name} needs a positive standard deviation.")

    def logpdf(self, v: np.ndarray) -> np.ndarray:
        if self.kind == "flat":
            return np.zeros_like(v)
        if self.kind == "uniform":
            return np.where((v >= self.a) & (v <= self.b), 0.0, -np.inf)
        if self.kind == "positive":
            return np.where(v > 0, 0.0, -np.inf)
        return -0.5 * ((v - self.a) / self.b) ** 2


# ------------------------------------------------------------------------------------------ the model

@dataclass
class Model:
    expr: sp.Expr
    x: sp.Symbol
    params: list[sp.Symbol]
    fn: Callable

    def __call__(self, xs: np.ndarray, theta: np.ndarray) -> np.ndarray:
        """Model values: xs (n,) with theta (D,) -> (n,); or theta (W, D) -> (W, n)."""
        theta = np.asarray(theta, dtype=float)
        if theta.ndim == 1:
            out = self.fn(xs, *theta)
            return np.broadcast_to(np.asarray(out, dtype=float), xs.shape)
        out = self.fn(xs[None, :], *[theta[:, k][:, None] for k in range(theta.shape[1])])
        return np.broadcast_to(np.asarray(out, dtype=float), (theta.shape[0], xs.size))


def compile_model(expr_str: str, param_names: list[str]) -> Model:
    if not 1 <= len(param_names) <= MAX_PARAMS:
        raise McmcError(f"Between 1 and {MAX_PARAMS} parameters are supported.")
    for n in param_names:
        if not n.isidentifier() or keyword.iskeyword(n) or n == "x":
            raise McmcError(f"'{n}' cannot be a parameter name (it must be an identifier and not 'x').")
        if n == SIGMA:
            raise McmcError("'sigma' is reserved for the noise level; choose another name.")
    if len(set(param_names)) != len(param_names):
        raise McmcError("Parameter names must be unique.")
    x = sp.Symbol("x")
    params = [sp.Symbol(n) for n in param_names]
    try:
        expr = sp.sympify(expr_str, locals={**{p.name: p for p in params}, "x": x})
    except (sp.SympifyError, TypeError, SyntaxError) as exc:
        raise McmcError(f"Could not parse the model: {exc}") from exc
    extra = expr.free_symbols - {x} - set(params)
    if extra:
        raise McmcError("The model uses symbol(s) not listed as parameters: " + ", ".join(sorted(s.name for s in extra)))
    missing = [p.name for p in params if p not in expr.free_symbols]
    if missing:
        raise McmcError("Parameter(s) that do not appear in the model: " + ", ".join(missing))
    return Model(expr, x, params, sp.lambdify([x, *params], expr, "numpy"))


# ------------------------------------------------------------------------------------------ diagnostics

def _autocorr_function(series: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation of a 1-D series by FFT."""
    n = len(series)
    x = series - series.mean()
    f = np.fft.rfft(x, n=2 * n)
    acf = np.fft.irfft(f * np.conjugate(f))[:n].real
    return acf / acf[0] if acf[0] > 0 else np.zeros(n)


def integrated_time(chain: np.ndarray, c: float = TAU_C) -> tuple[np.ndarray, bool]:
    """(tau per parameter, reliable) for a chain of shape (steps, walkers, dim): the integrated autocorrelation time from the
    walker-averaged autocorrelation function with Sokal's automated window. `reliable` is False when the chain is too short
    to find a window, or shorter than MIN_TAU_RATIO times the tau it finds (tau is then only a lower bound)."""
    steps, walkers, dim = chain.shape
    taus = np.empty(dim)
    reliable = True
    for d in range(dim):
        acf = np.mean([_autocorr_function(chain[:, w, d]) for w in range(walkers)], axis=0)
        cum = 2.0 * np.cumsum(acf) - 1.0
        window = np.arange(len(cum)) < c * cum
        taus[d] = cum[np.argmin(window) if not window.all() else len(cum) - 1]
        if window.all():
            reliable = False
    taus = np.maximum(taus, 1.0)
    if steps < MIN_TAU_RATIO * float(taus.max()):
        reliable = False
    return taus, reliable


def split_rhat(chain: np.ndarray) -> np.ndarray:
    """Split R-hat per parameter, treating each walker's series split in two halves as a chain."""
    steps, walkers, dim = chain.shape
    half = steps // 2
    if half < 4:
        return np.full(dim, np.inf)
    parts = np.concatenate([chain[:half], chain[half:2 * half]], axis=1)         # (half, 2W, D)
    n = parts.shape[0]
    means = parts.mean(axis=0)
    within = parts.var(axis=0, ddof=1).mean(axis=0)
    between = n * means.var(axis=0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        var_hat = (n - 1) / n * within + between / n
        return np.sqrt(var_hat / within)


# ------------------------------------------------------------------------------------------ the sampler

def stretch_sampler(logp: Callable[[np.ndarray], np.ndarray], p0: np.ndarray, steps: int, rng: np.random.Generator,
                    a: float = 2.0, progress: Callable[[float], None] | None = None) -> tuple[np.ndarray, np.ndarray, float]:
    """Goodman-Weare stretch move. `logp` maps (W, D) -> (W,). Returns (chain (steps, W, D), logp (steps, W), acceptance)."""
    walkers, dim = p0.shape
    half = walkers // 2
    p = p0.copy()
    lnp = logp(p)
    chain = np.empty((steps, walkers, dim))
    lps = np.empty((steps, walkers))
    accepted = 0
    for t in range(steps):
        for first, second in ((slice(0, half), slice(half, walkers)), (slice(half, walkers), slice(0, half))):
            S, C = p[first], p[second]
            partner = C[rng.integers(0, C.shape[0], size=S.shape[0])]
            z: np.ndarray = np.asarray(((a - 1.0) * rng.random(S.shape[0]) + 1.0) ** 2 / a, dtype=float)
            proposal = partner + z[:, None] * (S - partner)
            lnp_new = logp(proposal)
            with np.errstate(invalid="ignore"):
                accept = np.log(rng.random(S.shape[0])) < (dim - 1) * np.log(z) + lnp_new - lnp[first]
            accept &= np.isfinite(lnp_new)
            S[accept] = proposal[accept]                    # S is a view of p
            lnp[first] = np.where(accept, lnp_new, lnp[first])
            accepted += int(accept.sum())
        chain[t], lps[t] = p, lnp
        if progress and t % max(1, steps // 50) == 0:
            progress(t / steps)
    return chain, lps, accepted / (steps * walkers)


# ------------------------------------------------------------------------------------------ the posterior

@dataclass
class ParameterSummary:
    name: str
    mean: float
    sd: float
    median: float
    lo68: float
    hi68: float
    lo95: float
    hi95: float
    skew: float
    ls_value: float | None = None
    ls_se: float | None = None

    @property
    def non_gaussian(self) -> bool:
        """Visibly lopsided: strong skew, or a 68% interval whose two sides differ by more than half again."""
        up, down = self.hi68 - self.median, self.median - self.lo68
        return abs(self.skew) > 0.5 or (min(up, down) > 0 and max(up, down) / min(up, down) > 1.5)


@dataclass
class PosteriorResult:
    names: list[str]                       # includes 'sigma' last when it was sampled
    model: Model
    xs: np.ndarray
    ys: np.ndarray
    chain: np.ndarray                      # (steps, walkers, dim), all steps
    log_prob: np.ndarray                   # (steps, walkers)
    burn: int
    samples: np.ndarray                    # (N, dim) after burn-in, all walkers
    summaries: list[ParameterSummary]
    sigma_sampled: bool
    sigma_known: float | None
    acceptance: float
    tau: np.ndarray
    tau_reliable: bool
    ess: np.ndarray
    rhat: np.ndarray
    predictive_coverage: float
    ls_theta: np.ndarray
    seed: int
    warnings: list[str] = field(default_factory=list)
    unbounded: list[str] = field(default_factory=list)      # parameters the data do not bound (an infinite tail)
    residual_lag1: float = 0.0                              # lag-1 correlation of the residuals, in order of x
    residual_z: float = 0.0                                 # that correlation in units of its null standard error

    @property
    def converged(self) -> bool:
        post = self.chain.shape[0] - self.burn
        return bool(not self.unbounded and self.tau_reliable and post >= CONVERGED_STEPS_PER_TAU * float(self.tau.max())
                    and float(np.max(self.rhat)) < RHAT_LIMIT)

    @property
    def n_params(self) -> int:
        return len(self.names) - (1 if self.sigma_sampled else 0)

    def correlation(self) -> np.ndarray:
        return np.corrcoef(self.samples, rowvar=False) if self.samples.shape[1] > 1 else np.array([[1.0]])

    def summary_table(self) -> list[dict]:
        return [{"parameter": s.name, "median": s.median, "−1σ (16%)": s.lo68 - s.median, "+1σ (84%)": s.hi68 - s.median,
                 "95% interval": f"[{s.lo95:.4g}, {s.hi95:.4g}]", "mean": s.mean, "sd": s.sd,
                 "least squares": s.ls_value, "non-Gaussian": s.non_gaussian} for s in self.summaries]

    def predictive(self, xs: np.ndarray, n_draws: int = 300, include_noise: bool = True,
                   seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """(central, bands): the median model curve and its 2.5/16/84/97.5 percentile curves (4, len(xs)) over posterior draws,
        with the noise added when `include_noise`."""
        rng = np.random.default_rng(seed)
        draw = self.samples[rng.integers(0, len(self.samples), size=min(n_draws, len(self.samples)))]
        curves = self.model(np.asarray(xs, dtype=float), draw[:, :self.n_params])
        if include_noise:
            sig = draw[:, -1] if self.sigma_sampled else np.full(len(draw), self.sigma_known)
            curves = curves + rng.normal(size=curves.shape) * np.atleast_1d(sig)[:, None]
        return np.median(curves, axis=0), np.percentile(curves, [2.5, 16, 84, 97.5], axis=0)


def _least_squares_start(model: Model, xs: np.ndarray, ys: np.ndarray, p0: np.ndarray, sigma: np.ndarray | None):
    w = 1.0 / sigma if sigma is not None else np.ones_like(ys)

    def resid(theta):
        return (model(xs, theta) - ys) * w

    try:
        with np.errstate(all="ignore"):
            sol = least_squares(resid, p0, method="lm" if len(xs) >= len(p0) else "trf", max_nfev=2000)
    except Exception as exc:  # noqa: BLE001
        raise McmcError(f"Least squares could not start from the initial guess ({type(exc).__name__}: {exc}); try a different guess.") from exc
    if not np.all(np.isfinite(sol.x)):
        raise McmcError("Least squares did not find a finite starting point; try a different initial guess.")
    rss = float(np.sum(sol.fun ** 2))
    dof = max(len(xs) - len(p0), 1)
    scale = 1.0 if sigma is not None else rss / dof
    try:
        cov = np.linalg.pinv(sol.jac.T @ sol.jac) * scale
    except np.linalg.LinAlgError:
        cov = np.eye(len(p0))
    se = np.sqrt(np.maximum(np.diag(cov), 1e-300))
    return sol.x, se, math.sqrt(rss / dof) if sigma is None else None


def sample_posterior(xs, ys, expr: str, param_names: list[str], p0: list[float] | None = None,
                     priors: dict[str, Prior] | None = None, sigma: float | list[float] | None = None, steps: int = 8000,
                     walkers: int | None = None, burn_fraction: float = 0.3, seed: int = 0,
                     progress: Callable[[float], None] | None = None) -> PosteriorResult:
    """Sample the posterior of `expr` (a function of x with the named parameters) given the data."""
    xs, ys = np.asarray(xs, dtype=float).ravel(), np.asarray(ys, dtype=float).ravel()
    if xs.shape != ys.shape:
        raise McmcError("x and y must have the same length.")
    if len(xs) < MIN_POINTS:
        raise McmcError(f"Need at least {MIN_POINTS} data points.")
    if not (np.all(np.isfinite(xs)) and np.all(np.isfinite(ys))):
        raise McmcError("The data contains missing or non-finite values.")
    if not 100 <= steps <= MAX_STEPS:
        raise McmcError(f"Steps must be between 100 and {MAX_STEPS}.")
    if not 0.05 <= burn_fraction <= 0.8:
        raise McmcError("The burn-in fraction must be between 5% and 80%.")
    model = compile_model(expr, param_names)
    dim = len(param_names)
    if len(xs) < dim + 1:
        raise McmcError(f"{len(xs)} points cannot determine {dim} parameters and the noise level.")
    sigma_arr: np.ndarray | None = None
    if sigma is not None:
        sigma_arr = np.broadcast_to(np.asarray(sigma, dtype=float), xs.shape).copy()
        if not np.all(sigma_arr > 0) or not np.all(np.isfinite(sigma_arr)):
            raise McmcError("sigma must be positive.")
    prior_list = [(priors or {}).get(n, Prior()) for n in param_names]
    for n, pr in zip(param_names, prior_list):
        pr.validate(n)
    start = np.ones(dim) if p0 is None else np.asarray(p0, dtype=float)
    if start.shape != (dim,) or not np.all(np.isfinite(start)):
        raise McmcError(f"The initial guess needs {dim} finite numbers.")

    theta_hat, se, sigma_hat = _least_squares_start(model, xs, ys, start, sigma_arr)
    sampled = sigma_arr is None
    full_dim = dim + (1 if sampled else 0)
    walkers = walkers or max(4 * full_dim, 32)
    if walkers % 2 or not 2 * full_dim + 2 <= walkers <= MAX_WALKERS:
        raise McmcError(f"The number of walkers must be even and between {2 * full_dim + 2} and {MAX_WALKERS}.")
    n_obs = len(xs)

    def logp(P: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            lp = np.zeros(P.shape[0])
            for k, pr in enumerate(prior_list):
                lp = lp + pr.logpdf(P[:, k])
            pred = model(xs, P[:, :dim])
            if sampled:
                log_s = P[:, dim]
                resid2 = np.sum((pred - ys[None, :]) ** 2, axis=1)
                ll = -n_obs * log_s - 0.5 * resid2 * np.exp(-2.0 * log_s)
            else:
                assert sigma_arr is not None
                ll = -0.5 * np.sum(((pred - ys[None, :]) / sigma_arr[None, :]) ** 2, axis=1)
            out = lp + ll
        return np.where(np.isfinite(out), out, -np.inf)

    rng = np.random.default_rng(seed)
    centre = np.append(theta_hat, math.log(max(sigma_hat or 1e-12, 1e-12))) if sampled else theta_hat
    spread = np.append(se, 0.05) if sampled else se
    p_init = centre + 0.1 * spread * rng.normal(size=(walkers, full_dim))
    for _ in range(50):
        bad = ~np.isfinite(logp(p_init))
        if not bad.any():
            break
        p_init[bad] = centre + 0.01 * spread * rng.normal(size=(int(bad.sum()), full_dim))
    if not np.all(np.isfinite(logp(p_init))):
        raise McmcError("The walkers could not be started at finite probability: the priors may exclude the least-squares "
                        "solution; widen them or change the initial guess.")

    chain, lps, acc = stretch_sampler(logp, p_init, steps, rng, progress=progress)
    burn = int(burn_fraction * steps)
    post = chain[burn:]
    unbounded = _unbounded_parameters(post, theta_hat, se, param_names)
    if unbounded:                                          # statistics of an infinite tail are meaningless (and overflow)
        tau, reliable, rhat = np.full(full_dim, np.inf), False, np.full(full_dim, np.inf)
    else:
        with np.errstate(all="ignore"):
            tau, reliable = integrated_time(post)
            rhat = split_rhat(post)
    samples = post.reshape(-1, full_dim)
    ess = samples.shape[0] / tau
    names = list(param_names) + ([SIGMA] if sampled else [])
    if sampled:
        samples = samples.copy()
        samples[:, -1] = np.exp(samples[:, -1])               # report sigma, not log sigma
        chain = chain.copy()
        chain[:, :, -1] = np.exp(chain[:, :, -1])

    summaries = []
    for k, name in enumerate(names):
        col = samples[:, k]
        q = np.percentile(col, [2.5, 16, 50, 84, 97.5])
        with np.errstate(all="ignore"):
            sd = float(col.std())
            skew = float(np.mean(((col - col.mean()) / sd) ** 3)) if sd > 0 and np.isfinite(sd) else math.inf
        summaries.append(ParameterSummary(name, float(col.mean()), sd, float(q[2]), float(q[1]), float(q[3]), float(q[0]), float(q[4]), skew,
                                          float(theta_hat[k]) if k < dim else sigma_hat, float(se[k]) if k < dim else None))

    result = PosteriorResult(names, model, xs, ys, chain, lps, burn, samples, summaries, sampled,
                             None if sampled else float(np.mean(sigma_arr)),        # type: ignore[arg-type]
                             acc, tau, reliable, ess, rhat, 0.0, theta_hat, seed, unbounded=unbounded)
    _, bands = result.predictive(xs, include_noise=True, seed=seed + 1)
    result.predictive_coverage = float(np.mean((ys >= bands[0]) & (ys <= bands[3])))
    if len(xs) >= 8:
        with np.errstate(all="ignore"):
            res = (ys - model(xs, np.array([sm.median for sm in summaries[:dim]])))[np.argsort(xs, kind="stable")]
            r1 = float(np.corrcoef(res[:-1], res[1:])[0, 1])
        if np.isfinite(r1):
            result.residual_lag1, result.residual_z = r1, r1 * math.sqrt(len(xs))
    _diagnose(result)
    return result


UNBOUNDED_SE = 1e3        # a tail reaching this many standard errors from the least-squares value is "unbounded"


def _unbounded_parameters(post: np.ndarray, theta_hat: np.ndarray, se: np.ndarray, names: list[str]) -> list[str]:
    """Parameters whose sampled values run off to enormous distances from the least-squares fit. That happens when the data
    fix only a COMBINATION of parameters (e.g. V/K in a saturating curve observed far from saturation) and the prior does
    not bound the rest: the posterior is then not even normalisable, and its "intervals" are artefacts of how long the
    chain ran."""
    out = []
    for k, name in enumerate(names):
        col = post[:, :, k]
        scale = max(float(se[k]), 1e-9 * abs(float(theta_hat[k])), 1e-12)
        with np.errstate(all="ignore"):
            far = np.abs(col - theta_hat[k])
            if not np.all(np.isfinite(col)) or float(np.percentile(far, 99.9)) > UNBOUNDED_SE * scale:
                out.append(name)
    return out


def _diagnose(r: PosteriorResult) -> None:
    if r.unbounded:
        r.warnings.append("The data do not bound " + ", ".join(r.unbounded) + ": " + ("its posterior has" if len(r.unbounded) == 1 else
                          "their posteriors have") + " an unbounded tail (typical when "
                          "parameters trade off, as V and K do in a saturating curve measured far from saturation). The "
                          "intervals are not meaningful. Give " + ("it" if len(r.unbounded) == 1 else "them") +
                          " a prior bound (uniform) or a normal prior, or collect data that constrain it.")
        return
    post = r.chain.shape[0] - r.burn
    if not r.tau_reliable:
        r.warnings.append("The run is too short to estimate the autocorrelation time reliably (fewer than "
                          f"{MIN_TAU_RATIO} of them long), so convergence cannot be confirmed: run longer.")
    elif post < CONVERGED_STEPS_PER_TAU * float(r.tau.max()):
        r.warnings.append(f"The run is only {post / float(r.tau.max()):.0f} autocorrelation times long (aim for at least "
                          f"{CONVERGED_STEPS_PER_TAU}): the intervals are rough; run longer.")
    if float(np.max(r.rhat)) >= RHAT_LIMIT:
        r.warnings.append(f"Split R-hat is {float(np.max(r.rhat)):.2f} (should be below {RHAT_LIMIT}): the walkers have not mixed.")
    if not 0.1 <= r.acceptance <= 0.8:
        r.warnings.append(f"The acceptance fraction is {r.acceptance:.2f}, outside the healthy 0.1 - 0.8: the posterior may be "
                          "awkwardly shaped or the model badly scaled.")
    if abs(r.residual_z) > RESIDUAL_Z_LIMIT:
        r.warnings.append(f"The residuals are correlated along x (lag-1 correlation {r.residual_lag1:+.2f}): the model leaves systematic "
                          "structure unexplained." + (" The sampled noise level has grown to absorb that misfit, so the intervals "
                                                      "understate the real uncertainty." if r.sigma_sampled else ""))
    if r.predictive_coverage < 0.8:
        r.warnings.append(f"Only {r.predictive_coverage:.0%} of the data lies inside the 95% predictive band: the model (or the "
                          "noise level) does not describe these data, however tight the intervals look.")
    if r.sigma_known is not None and r.predictive_coverage > 0.995 and len(r.ys) > 10:
        r.warnings.append("Nearly all the data lies inside the predictive band: the given sigma may be too large.")
    odd = [s.name for s in r.summaries if s.non_gaussian]
    if odd:
        r.warnings.append("Visibly non-Gaussian posterior for " + ", ".join(odd) + ": a symmetric +/- would misreport it; use the "
                          "credible intervals.")


# ------------------------------------------------------------------------------------------ demo data

DEMOS: dict[str, dict] = {
    "Exponential decay with an offset": dict(expr="a*exp(-b*x) + c", params=["a", "b", "c"], truth=[2.0, 0.8, 0.5], p0=[1.0, 1.0, 0.0],
                                              xs=(0.0, 6.0), noise=0.08, n=30),
    "Michaelis-Menten (saturating)": dict(expr="V*x/(K + x)", params=["V", "K"], truth=[10.0, 2.0], p0=[8.0, 1.0],
                                           xs=(0.3, 6.0), noise=0.5, n=10,
                                           priors={"V": ("uniform", 0.0, 60.0), "K": ("uniform", 0.0, 30.0)}),
    "Gaussian peak": dict(expr="A*exp(-(x - m)**2/(2*s**2))", params=["A", "m", "s"], truth=[3.0, 1.5, 0.6], p0=[2.5, 1.4, 0.7],
                          xs=(-1.0, 4.0), noise=0.15, n=40),
    "Logistic growth": dict(expr="K/(1 + exp(-r*(x - x0)))", params=["K", "r", "x0"], truth=[8.0, 1.2, 3.0], p0=[7.0, 1.0, 2.5],
                            xs=(0.0, 8.0), noise=0.3, n=35),
}


def demo_dataset(name: str, seed: int = 0) -> dict:
    """{xs, ys, expr, params, truth, p0, noise} for a built-in nonlinear example."""
    if name not in DEMOS:
        raise McmcError(f"Unknown demo '{name}'.")
    d = DEMOS[name]
    model = compile_model(d["expr"], d["params"])
    xs = np.linspace(d["xs"][0], d["xs"][1], d["n"])
    ys = model(xs, np.asarray(d["truth"])) + np.random.default_rng(seed).normal(0.0, d["noise"], xs.shape)
    priors = {k: Prior(*v) for k, v in d.get("priors", {}).items()}
    return dict(xs=xs, ys=ys, expr=d["expr"], params=list(d["params"]), truth=list(d["truth"]), p0=list(d["p0"]), noise=d["noise"],
                priors=priors)
