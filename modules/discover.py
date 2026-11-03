"""
Equation discovery from data: find the differential equations a time series obeys -- Streamlit-free, no LLM.

THE METHOD (sparse identification of nonlinear dynamics, "SINDy"). Given measurements X(t) of one or more state
variables, estimate their time derivatives, build a library of candidate terms (1, x, y, x*y, x**2, sin(x), ...) and
find, for each state, the SPARSEST combination of terms that reproduces dx/dt. Sparsity is what turns curve-fitting
into discovery: out of dozens of candidates only the few that matter survive, and what is left reads as an equation.
The regression is "sequentially thresholded least squares" (STLSQ): solve, zero every coefficient smaller than a
threshold, re-solve on what is left, repeat.

VERIFICATION-FIRST. A discovered equation is a hypothesis, so each one comes with the evidence for it:
  * derivative fit: R^2 of dx/dt against the library terms that survived (how much of the motion they explain);
  * simulation: integrate the discovered system from the data's own initial state and compare with the measurements.
    This is the check that matters: a model that fits derivatives but drifts away when run forward is not a law;
  * stability: refit on resampled rows (a bootstrap) and report how often each term is selected. A term chosen
    only half the time is noise the threshold happened to keep. (Resampling rows ignores the correlation between
    neighbouring time points, so the inclusion rates are optimistic, and the report says so);
  * the threshold itself is chosen by an information criterion (BIC) over a grid, not by eye, and the whole
    path (how error trades against number of terms) is returned so it can be inspected. BIC rather than AICc because
    derivative residuals are correlated in time, so the effective number of observations is smaller than the number of
    rows and the lighter AICc penalty keeps noise terms (checked: on noisy oscillator and predator-prey data AICc kept
    6 and 9 terms where the truth has 3 and 4; BIC kept the true ones).

LIMITS, STATED. The true law must be (close to) expressible in the library: if it contains a term that is not a
candidate (a ratio, an exponential of a state), no threshold will find it, and the simulation check is what shows
it. Noise is amplified by differentiation, so noisy data wants the Savitzky-Golay smoother (the default) and
enough points. Chaotic systems (Lorenz) fit the structure well but cannot be forecast far by ANY model, including the
true one, so validate those over a short horizon.
"""
import keyword
import math
from dataclasses import dataclass, field
from itertools import combinations_with_replacement

import numpy as np
import sympy as sp
from scipy.integrate import solve_ivp
from scipy.signal import savgol_filter

MAX_TERMS = 200
MAX_STATES = 6
MIN_ROWS_PER_TERM = 3


class DiscoverError(ValueError):
    """Bad input (names, shapes, a library larger than the data can support)."""


# ------------------------------------------------------------------------------------------ data in

def check_names(names: list[str]) -> None:
    if not 1 <= len(names) <= MAX_STATES:
        raise DiscoverError(f"Between 1 and {MAX_STATES} state variables are supported.")
    for n in names:
        if not n.isidentifier() or keyword.iskeyword(n):
            raise DiscoverError(f"'{n}' cannot be used as a variable name (use letters, digits and underscores, "
                                "starting with a letter).")
    if len(set(names)) != len(names):
        raise DiscoverError("Variable names must be unique.")


def parse_table(text: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Parse pasted or uploaded CSV/whitespace text: first column time, the rest states. A header row is optional.
    Returns (t, X, names)."""
    rows = [ln.strip() for ln in text.strip().splitlines() if ln.strip() and not ln.strip().startswith("#")]
    if len(rows) < 3:
        raise DiscoverError("Need a header (optional) and at least a few rows of numbers.")
    sep = "," if "," in rows[0] else (";" if ";" in rows[0] else None)
    cells = [r.split(sep) if sep else r.split() for r in rows]
    header = None
    try:
        [float(c) for c in cells[0]]
    except ValueError:
        header, cells = [c.strip() for c in cells[0]], cells[1:]
    try:
        data = np.array([[float(c) for c in r] for r in cells], dtype=float)
    except ValueError as exc:
        raise DiscoverError(f"A cell is not a number: {exc}") from exc
    if data.ndim != 2 or data.shape[1] < 2:
        raise DiscoverError("Need a time column and at least one state column.")
    if header is not None and len(header) != data.shape[1]:
        raise DiscoverError("The header has a different number of columns than the data.")
    names = header[1:] if header else (["x"] if data.shape[1] == 2 else [f"x{i}" for i in range(1, data.shape[1])])
    check_names(names)
    if not np.all(np.isfinite(data)):
        raise DiscoverError("The data contains missing or non-finite values.")
    return data[:, 0], data[:, 1:], names


def to_uniform(t: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, bool]:
    """(t, X, resampled): the data on an evenly spaced time grid, interpolating (cubic) only if it was not already."""
    t = np.asarray(t, float)
    X = np.asarray(X, float)
    if X.ndim == 1:
        X = X[:, None]
    if t.ndim != 1 or len(t) != len(X):
        raise DiscoverError("Time and data must have the same number of rows.")
    if len(t) < 10:
        raise DiscoverError("Need at least 10 time points.")
    if np.any(np.diff(t) <= 0):
        raise DiscoverError("Time must be strictly increasing.")
    dt = np.diff(t)
    if np.ptp(dt) <= 1e-6 * np.mean(dt):
        return t, X, False
    from scipy.interpolate import interp1d
    grid = np.linspace(t[0], t[-1], len(t))
    return grid, interp1d(t, X, kind="cubic", axis=0)(grid), True


def estimate_derivatives(t: np.ndarray, X: np.ndarray, method: str = "savgol", window: int | None = None,
                         polyorder: int = 3) -> tuple[np.ndarray, int]:
    """(dX/dt, window used). `savgol` smooths while differentiating (use it for noisy data); `gradient` is plain
    central differences (exact enough for clean data, and it has no window)."""
    if method == "gradient":
        return np.gradient(X, t, axis=0, edge_order=2), 1
    n = len(t)
    if window is None:
        window = int(min(51, max(7, round(0.05 * n))))
    window = int(window) | 1                                  # odd
    window = min(window, n - 1 if (n - 1) % 2 else n - 2)
    if window <= polyorder:
        raise DiscoverError("Not enough points for the smoothing window; use more data or the 'gradient' method.")
    return savgol_filter(X, window, polyorder, deriv=1, delta=float(t[1] - t[0]), axis=0, mode="interp"), window


# ------------------------------------------------------------------------------------------ library

def build_library(names: list[str], degree: int = 2, trig: bool = False, constant: bool = True) -> tuple[list[sp.Symbol], list[sp.Expr]]:
    check_names(names)
    if not 1 <= degree <= 4:
        raise DiscoverError("The polynomial degree must be between 1 and 4.")
    syms = [sp.Symbol(n) for n in names]
    terms: list[sp.Expr] = [sp.Integer(1)] if constant else []
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(len(syms)), d):
            terms.append(sp.Mul(*[syms[i] for i in combo]))
    if trig:
        for s in syms:
            terms += [sp.sin(s), sp.cos(s)]
    if len(terms) > MAX_TERMS:
        raise DiscoverError(f"That library has {len(terms)} terms (the limit is {MAX_TERMS}); lower the degree or drop trig.")
    return syms, terms


def evaluate_library(terms: list[sp.Expr], syms: list[sp.Symbol], X: np.ndarray) -> np.ndarray:
    f = sp.lambdify(syms, terms, "numpy")
    cols = f(*[X[:, i] for i in range(X.shape[1])])
    return np.column_stack([np.broadcast_to(np.asarray(c, float), (len(X),)) for c in cols])


# ------------------------------------------------------------------------------------------ the regression

def stlsq(Theta: np.ndarray, dX: np.ndarray, threshold: float, ridge: float = 1e-8, max_iter: int = 20) -> np.ndarray:
    """Sequentially thresholded least squares. Columns are scaled to unit length for conditioning, but the threshold
    applies to the coefficients in their ORIGINAL units (what a person reads off the equation)."""
    norms = np.linalg.norm(Theta, axis=0)
    norms[norms == 0] = 1.0
    A = Theta / norms
    n_terms = A.shape[1]
    coef = np.linalg.solve(A.T @ A + ridge * np.eye(n_terms), A.T @ dX)            # (terms, states), scaled
    for _ in range(max_iter):
        small = np.abs(coef / norms[:, None]) < threshold
        coef[small] = 0.0
        changed = False
        for j in range(dX.shape[1]):
            keep = ~small[:, j]
            if not keep.any():
                continue
            sol = np.linalg.lstsq(A[:, keep], dX[:, j], rcond=None)[0]
            if not np.allclose(coef[keep, j], sol):
                changed = True
            coef[keep, j] = sol
        if not changed:
            break
    coef = coef / norms[:, None]
    coef[np.abs(coef) < threshold] = 0.0
    return coef


def _rss(Theta: np.ndarray, dX: np.ndarray, coef: np.ndarray) -> float:
    return float(np.sum((dX - Theta @ coef) ** 2))


def bic(rss: float, n: int, k: int) -> float:
    """Bayesian information criterion for a regression with n observations and k free coefficients (lower is better)."""
    n = max(n, 1)
    return n * math.log(max(rss, 1e-300) / n) + k * math.log(n)


@dataclass
class PathPoint:
    threshold: float
    n_terms: int
    rss: float
    bic: float


def sparsity_path(Theta: np.ndarray, dX: np.ndarray, n_grid: int = 30) -> list[PathPoint]:
    """The trade-off between error and number of terms across a grid of thresholds (geometric, relative to the largest
    coefficient of the unregularised fit)."""
    dense = np.linalg.lstsq(Theta, dX, rcond=None)[0]
    top = float(np.max(np.abs(dense))) or 1.0
    out = []
    for thr in np.geomspace(1e-4 * top, 0.5 * top, n_grid):
        coef = stlsq(Theta, dX, float(thr))
        k = int(np.count_nonzero(coef))
        rss = _rss(Theta, dX, coef)
        out.append(PathPoint(float(thr), k, rss, bic(rss, dX.size, k)))
    return out


# ------------------------------------------------------------------------------------------ result and fit

@dataclass
class DiscoverResult:
    names: list[str]
    syms: list[sp.Symbol]
    terms: list[sp.Expr]
    coef: np.ndarray                          # (terms, states)
    threshold: float
    threshold_mode: str                       # "auto (BIC)" | "manual"
    r2: list[float]                           # derivative R^2 per state
    path: list[PathPoint]
    t: np.ndarray                             # the (uniform) time grid used
    X: np.ndarray
    derivative_method: str
    window: int
    resampled: bool
    inclusion: np.ndarray | None = None       # (terms, states) bootstrap selection frequency
    notes: list[str] = field(default_factory=list)

    @property
    def n_terms(self) -> int:
        return int(np.count_nonzero(self.coef))

    def rhs(self, j: int) -> sp.Expr:
        return sp.Add(*[sp.Float(f"{c:.6g}") * term for c, term in zip(self.coef[:, j], self.terms) if c != 0])

    def equations(self) -> list[sp.Eq]:
        return [sp.Eq(sp.Symbol(f"d{n}/dt"), self.rhs(j)) for j, n in enumerate(self.names)]

    def latex(self) -> list[str]:
        return [rf"\dot{{{n}}} = {sp.latex(self.rhs(j))}" for j, n in enumerate(self.names)]

    def term_table(self) -> list[dict]:
        rows = []
        for j, n in enumerate(self.names):
            for i, term in enumerate(self.terms):
                if self.coef[i, j] != 0:
                    rows.append({"equation": f"d{n}/dt", "term": str(term), "coefficient": float(self.coef[i, j]),
                                 "selected in resamples": None if self.inclusion is None else float(self.inclusion[i, j])})
        return rows


def fit(t, X, names: list[str], degree: int = 2, trig: bool = False, constant: bool = True, threshold: float | None = None,
        derivative_method: str = "savgol", window: int | None = None, bootstrap: int = 0, seed: int = 0) -> DiscoverResult:
    """Discover dX/dt = f(X) from the time series. `threshold=None` picks it by BIC. `bootstrap` > 0 adds term-selection
    frequencies over that many row-resamples (seeded, so a run is reproducible)."""
    t, X, resampled = to_uniform(t, X)
    if X.shape[1] != len(names):
        raise DiscoverError(f"{len(names)} name(s) given for {X.shape[1]} data column(s).")
    syms, terms = build_library(names, degree, trig, constant)
    dX, w = estimate_derivatives(t, X, derivative_method, window)
    trim = w // 2 if derivative_method == "savgol" else 2          # the edges are the least reliable derivatives
    sl = slice(trim, len(t) - trim)
    Theta, D = evaluate_library(terms, syms, X)[sl], dX[sl]
    if len(Theta) < MIN_ROWS_PER_TERM * len(terms):
        raise DiscoverError(f"{len(Theta)} usable rows is too few for {len(terms)} candidate terms; give more data or a "
                            "smaller library.")
    path = sparsity_path(Theta, D)
    if threshold is None:
        best = min(path, key=lambda p: (p.bic, p.n_terms))
        threshold, mode = best.threshold, "auto (BIC)"
    else:
        if threshold <= 0:
            raise DiscoverError("The threshold must be positive.")
        mode = "manual"
    coef = stlsq(Theta, D, threshold)
    tss = np.sum((D - D.mean(axis=0)) ** 2, axis=0)
    rss = np.sum((D - Theta @ coef) ** 2, axis=0)
    r2 = [float(1 - r / s) if s > 0 else 1.0 for r, s in zip(rss, tss)]
    result = DiscoverResult(names, syms, terms, coef, float(threshold), mode, r2, path, t, X, derivative_method, w, resampled)
    if resampled:
        result.notes.append("The time points were not evenly spaced, so the data was interpolated onto an even grid first.")
    if bootstrap > 0:
        rng = np.random.default_rng(seed)
        counts = np.zeros_like(coef)
        for _ in range(bootstrap):
            idx = rng.integers(0, len(Theta), len(Theta))
            counts += stlsq(Theta[idx], D[idx], threshold) != 0
        result.inclusion = counts / bootstrap
        result.notes.append("Selection frequencies resample rows independently, ignoring the correlation between "
                            "neighbouring time points, so they are optimistic.")
    return result


# ------------------------------------------------------------------------------------------ verification

def simulate(result: DiscoverResult, x0: np.ndarray, t_eval: np.ndarray) -> np.ndarray:
    """Integrate the discovered system. Rows are time points; a run that blows up or fails comes back as NaN rows
    after the failure (never an exception: a bad model is a finding, not a crash)."""
    f = sp.lambdify(result.syms, [result.rhs(j) for j in range(len(result.names))], "numpy")

    def rhs(_t, y):
        return np.asarray(f(*y), float)

    out = np.full((len(t_eval), len(x0)), np.nan)
    try:
        sol = solve_ivp(rhs, (float(t_eval[0]), float(t_eval[-1])), np.asarray(x0, float), t_eval=t_eval, method="LSODA",
                        rtol=1e-8, atol=1e-10)
        if sol.y.shape[1]:
            out[:sol.y.shape[1]] = sol.y.T
    except Exception:  # noqa: BLE001 -- stiff blow-up, overflow, ...
        pass
    return out


def simulation_error(result: DiscoverResult, horizon: float = 1.0) -> float:
    """Normalised RMS error between the data and the discovered system integrated from the data's first point, over the
    first `horizon` fraction of the record (0 = perfect; 1 = no better than predicting each state's mean; inf if the
    model blows up)."""
    n = max(3, int(round(horizon * len(result.t))))
    sim = simulate(result, result.X[0], result.t[:n])
    data = result.X[:n]
    if not np.all(np.isfinite(sim)):
        return math.inf
    scale = np.std(data, axis=0)
    scale[scale == 0] = 1.0
    return float(np.sqrt(np.mean(((sim - data) / scale) ** 2)))


def verdict(result: DiscoverResult, sim_error: float) -> tuple[str, str]:
    """('verified' | 'caution' | 'failed', why) -- the discovered system as a hypothesis with its evidence."""
    weak = [n for n, r in zip(result.names, result.r2) if r < 0.9]
    unstable = (result.inclusion is not None and np.any((result.coef != 0) & (result.inclusion < 0.8)))
    if not math.isfinite(sim_error) or sim_error > 0.5:
        return "failed", "Run forward, the discovered system does not reproduce the data."
    if weak or sim_error > 0.15:
        why = (f"The derivative fit is weak for {', '.join(weak)}. " if weak else "") + f"The simulation error is {sim_error:.2f}."
        return "caution", why.strip()
    if unstable:
        return "caution", "Some selected terms appear in fewer than 80% of the resamples: they may be noise."
    return "verified", f"Derivatives are explained (R² ≥ 0.9) and the simulation tracks the data (error {sim_error:.2f})."


# ------------------------------------------------------------------------------------------ naming known laws

def _support(result: DiscoverResult) -> dict[str, set[str]]:
    return {n: {str(result.terms[i]) for i in np.flatnonzero(result.coef[:, j])} for j, n in enumerate(result.names)}


def recognise(result: DiscoverResult) -> str | None:
    """A named law when the discovered structure is exactly one (a hint, not a claim: the coefficients carry the meaning)."""
    names = result.names
    sup = _support(result)
    c = {n: {str(result.terms[i]): result.coef[i, j] for i in np.flatnonzero(result.coef[:, j])} for j, n in enumerate(names)}

    def prod(*factors: str) -> str:
        return str(sp.Mul(*[sp.Symbol(f) for f in factors]))

    if len(names) == 1:
        x = names[0]
        x2 = prod(x, x)
        if sup[x] == {x}:
            return "exponential growth" if c[x][x] > 0 else "exponential decay"
        if sup[x] == {x, x2} and c[x][x] > 0 > c[x][x2]:
            return f"logistic growth (carrying capacity ≈ {-c[x][x] / c[x][x2]:.3g})"
    if len(names) == 2:
        x, y = names
        xy = prod(x, y)
        if sup[x] == {y} and x in sup[y] and sup[y] <= {x, y} and c[x][y] * c[y][x] < 0:
            return "harmonic oscillator" + (" (damped)" if y in sup[y] and c[y][y] < 0 else "")
        if sup[x] == {x, xy} and sup[y] == {y, xy} and c[x][x] > 0 > c[x][xy] and c[y][y] < 0 < c[y][xy]:
            return "predator-prey (Lotka-Volterra)"
    if len(names) == 3:
        x, y, z = names
        if sup[x] == {x, y} and sup[y] == {x, y, prod(x, z)} and sup[z] == {z, prod(x, y)}:
            return "Lorenz system"
    return None


# ------------------------------------------------------------------------------------------ export

def to_python(result: DiscoverResult) -> str:
    """A self-contained function for the discovered system, for use with scipy.integrate.solve_ivp."""
    from sympy.printing.numpy import NumPyPrinter
    printer = NumPyPrinter()
    lines = ["import numpy", "", "", "def rhs(t, state):", f"    {', '.join(result.names)} = state" +
             ("," if len(result.names) == 1 else ""), "    return ["]
    for j, n in enumerate(result.names):
        lines.append(f"        {printer.doprint(result.rhs(j))},   # d{n}/dt")
    lines.append("    ]")
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------ demo systems

DEMOS: dict[str, dict] = {
    "Damped oscillator": dict(names=["x", "v"], x0=[1.0, 0.0], span=(0.0, 15.0), n=1500,
                              truth=["dx/dt = v", "dv/dt = -4*x - 0.4*v"],
                              rhs=lambda t, s: [s[1], -4.0 * s[0] - 0.4 * s[1]]),
    "Logistic growth": dict(names=["x"], x0=[0.2], span=(0.0, 12.0), n=1200,
                            truth=["dx/dt = 0.8*x - 0.08*x**2"],
                            rhs=lambda t, s: [0.8 * s[0] - 0.08 * s[0] ** 2]),
    "Predator-prey (Lotka-Volterra)": dict(names=["x", "y"], x0=[10.0, 5.0], span=(0.0, 20.0), n=2000,
                                           truth=["dx/dt = 1.0*x - 0.1*x*y", "dy/dt = -1.5*y + 0.075*x*y"],
                                           rhs=lambda t, s: [1.0 * s[0] - 0.1 * s[0] * s[1], -1.5 * s[1] + 0.075 * s[0] * s[1]]),
    "Lorenz attractor": dict(names=["x", "y", "z"], x0=[-8.0, 7.0, 27.0], span=(0.0, 12.0), n=4800,
                             truth=["dx/dt = -10*x + 10*y", "dy/dt = 28*x - y - x*z", "dz/dt = -2.667*z + x*y"],
                             rhs=lambda t, s: [10 * (s[1] - s[0]), s[0] * (28 - s[2]) - s[1], s[0] * s[1] - (8 / 3) * s[2]]),
}


def demo_data(name: str, noise: float = 0.0, seed: int = 0) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """(t, X, names, truth) for a built-in system, with Gaussian noise of `noise` x each state's standard deviation."""
    if name not in DEMOS:
        raise DiscoverError(f"Unknown demo '{name}'.")
    d = DEMOS[name]
    t = np.linspace(d["span"][0], d["span"][1], d["n"])
    sol = solve_ivp(d["rhs"], d["span"], d["x0"], t_eval=t, method="DOP853", rtol=1e-11, atol=1e-12)
    X = sol.y.T
    if noise > 0:
        X = X + np.random.default_rng(seed).normal(0.0, noise, X.shape) * np.std(X, axis=0)
    return t, X, list(d["names"]), list(d["truth"])
