"""
The time-independent Schrödinger equation in one dimension, solved and CHECKED -- Streamlit-free.

    -(hbar^2 / 2m) psi'' + V(x) psi = E psi        on a finite interval, psi = 0 at both ends

The interval's ends are hard walls. For an infinite square well (and for the radial problem, where psi = u(r)
must vanish at r = 0) that is exactly the model; for everything else the interval is a numerical box and
the solver checks that the states it reports do not actually feel it.

METHOD. Three-point finite differences on a uniform grid give a symmetric tridiagonal matrix, whose lowest
eigenpairs `scipy.linalg.eigh_tridiagonal` finds in O(N) per state. The solve is repeated on grids of N, 2N
and 4N points. From the three energies of each level the convergence order is MEASURED
(p = log2 of the ratio of successive differences), and when it is close to the textbook 2 the energies are
Richardson-extrapolated. A level whose measured order is not near 2 (a singular potential such as the
Coulomb 1/r converges more slowly on a uniform grid) is reported as it is, with no extrapolation and a
note, rather than quietly corrected by a formula that does not apply.

CHECKS, all computed on the result:
  * the states are orthonormal;
  * state n has exactly n nodes (the 1D oscillation theorem -- a skipped or doubled level shows up here);
  * the energy matches the known analytic spectrum, where the potential has one (box, oscillator, Morse,
    Pöschl–Teller, finite well, hydrogen radial), within a tolerance stated on screen;
  * the states are contained in the box (probability near the ends is negligible);
  * Δx Δp >= hbar/2 for every state;
  * the measured convergence order.

POTENTIALS are expressions in x with named parameters, built-in or typed (`custom`); hbar and m are
available inside every expression.
"""
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import sympy as sp
from scipy.linalg import eigh_tridiagonal
from scipy.optimize import brentq

from modules.progress import ProgressFn, report
from modules.quantum_common import (
    CheckList, ExpressionError, safe_expression, trapezoid_norm, vectorized,
)

MIN_POINTS = 50
MAX_POINTS = 6000
MAX_LEVELS = 20


@dataclass(frozen=True)
class PotentialSpec:
    key: str
    label: str
    description: str
    formula: str                          # an expression in x, hbar, m and the parameters
    params: dict[str, float]              # parameter name -> default value
    domain: tuple[float, float]           # suggested interval
    analytic: Callable[[int, dict[str, float], float, float, tuple[float, float]], np.ndarray] | None = None
    hard_walls: bool = False              # the interval ends ARE part of the model
    atomic_units: bool = False            # hbar = m = 1 are not adjustable (hydrogen)
    n_levels: int = 6


def _box_energies(n_levels, params, hbar, m, domain):
    length = domain[1] - domain[0]
    return (np.arange(1, n_levels + 1) ** 2) * np.pi ** 2 * hbar ** 2 / (2 * m * length ** 2)


def _oscillator_energies(n_levels, params, hbar, m, domain):
    return hbar * params["w"] * (np.arange(n_levels) + 0.5)


def _morse_energies(n_levels, params, hbar, m, domain):
    d, a = params["D"], params["a"]
    w0 = a * np.sqrt(2 * d / m)
    n_max = int(np.floor(np.sqrt(2 * m * d) / (a * hbar) - 0.5))        # the bound levels
    n = np.arange(min(n_levels, n_max + 1))
    e = hbar * w0 * (n + 0.5)
    return e - e ** 2 / (4 * d)


def _poschl_teller_energies(n_levels, params, hbar, m, domain):
    lam, a = params["lam"], params["a"]
    n = np.arange(min(n_levels, int(np.ceil(lam))))                       # bound states have n < lambda
    return -hbar ** 2 * a ** 2 * (lam - n) ** 2 / (2 * m)


def _finite_well_energies(n_levels, params, hbar, m, domain):
    """Bound states of V = -V0 for |x| < a/2, 0 outside, from the matching conditions (found by bracketing
    each root of the even / odd transcendental equation): independent of any grid."""
    v0, a = params["V0"], params["a"]
    z0 = (a / 2) * np.sqrt(2 * m * v0) / hbar
    energies = []
    # with z = k a/2 and z0 as above, kappa a/2 = sqrt(z0^2 - z^2):
    #   even states:  z tan z = sqrt(z0^2 - z^2);   odd states:  -z cot z = sqrt(z0^2 - z^2)
    n_branches = int(np.ceil(2 * z0 / np.pi))
    for branch in range(n_branches):
        lo = branch * np.pi / 2 + 1e-9
        hi = min((branch + 1) * np.pi / 2 - 1e-9, z0 - 1e-12)
        if lo >= hi:
            continue
        g = ((lambda z: z * np.tan(z) - np.sqrt(max(z0 ** 2 - z ** 2, 0.0))) if branch % 2 == 0
             else (lambda z: -z / np.tan(z) - np.sqrt(max(z0 ** 2 - z ** 2, 0.0))))
        try:
            z = brentq(g, lo, hi)
        except ValueError:
            continue
        energies.append(-v0 + 2 * hbar ** 2 * z ** 2 / (m * a ** 2))     # E = -V0 + hbar^2 k^2 / 2m
    return np.array(sorted(energies)[:n_levels])


def _hydrogen_energies(n_levels, params, hbar, m, domain):
    l = int(round(params["l"]))
    return -0.5 / (np.arange(n_levels) + l + 1.0) ** 2


POTENTIALS: dict[str, PotentialSpec] = {s.key: s for s in [
    PotentialSpec("box", "Infinite square well", "Hard walls at both ends of the interval; V = 0 inside.",
                  "0", {}, (0.0, 1.0), _box_energies, hard_walls=True, n_levels=6),
    PotentialSpec("harmonic", "Harmonic oscillator", "V = ½ m w² x²; evenly spaced levels ħw(n + ½).",
                  "0.5*m*w**2*x**2", {"w": 1.0}, (-8.0, 8.0), _oscillator_energies, n_levels=6),
    PotentialSpec("finite_well", "Finite square well", "V = -V0 for |x| < a/2, else 0; a finite number of bound states.",
                  "-V0*Heaviside(a/2 - abs(x))", {"V0": 10.0, "a": 2.0}, (-10.0, 10.0), _finite_well_energies,
                  n_levels=4),
    PotentialSpec("morse", "Morse potential", "V = D (1 - e^{-a x})²: a molecular bond; finitely many bound levels.",
                  "D*(1 - exp(-a*x))**2", {"D": 12.0, "a": 0.5}, (-6.0, 40.0), _morse_energies, n_levels=5),
    PotentialSpec("poschl_teller", "Pöschl–Teller well", "V = -ħ²a²λ(λ+1)/(2m) sech²(a x): reflectionless for integer λ.",
                  "-hbar**2*a**2*lam*(lam+1)/(2*m)*sech(a*x)**2", {"lam": 3.0, "a": 1.0}, (-14.0, 14.0),
                  _poschl_teller_energies, n_levels=3),
    PotentialSpec("double_well", "Double well", "V = λ (x² - x0²)²: two minima with a tunnelling splitting (no closed form).",
                  "lam*(x**2 - x0**2)**2", {"lam": 1.0, "x0": 1.5}, (-4.0, 4.0), None, n_levels=6),
    PotentialSpec("hydrogen", "Hydrogen, radial (atomic units)",
                  "u(r) for V = -1/r + l(l+1)/2r²; E = -1/(2n²). Atomic units, so ħ = m = 1 are fixed.",
                  "-1/x + l*(l+1)/(2*x**2)", {"l": 0.0}, (0.0, 60.0), _hydrogen_energies, hard_walls=True,
                  atomic_units=True, n_levels=4),
    PotentialSpec("custom", "Custom V(x)", "Type any expression in x (and hbar, m, and your own parameters).",
                  "0.5*x**2 + 0.1*x**4", {}, (-6.0, 6.0), None, n_levels=5),
]}


# ------------------------------------------------------------------------------------------ result types

@dataclass
class StateObservables:
    energy: float
    mean_x: float
    std_x: float
    std_p: float
    mean_v: float
    mean_t: float
    nodes: int
    parity: str | None                    # "even" | "odd" for a symmetric potential on a symmetric box
    edge_probability: float


@dataclass
class EigenResult:
    key: str
    formula: str
    params: dict[str, float]
    hbar: float
    mass: float
    x: np.ndarray                         # the grid (interior points)
    potential: np.ndarray
    energies: np.ndarray                  # best estimates (extrapolated where the order was measured sensibly)
    raw_energies: np.ndarray              # on the user's grid, no extrapolation
    states: np.ndarray                    # shape (len(x), n_levels), each column normalised
    observables: list[StateObservables]
    orders: np.ndarray                    # measured convergence order per level (nan if undefined)
    extrapolated: np.ndarray              # bool per level
    analytic: np.ndarray | None
    checks: CheckList
    n_points: int
    domain: tuple[float, float]
    notes: list[str] = field(default_factory=list)

    @property
    def n_levels(self) -> int:
        return len(self.energies)


# ------------------------------------------------------------------------------------------ the solver

def _grid(domain: tuple[float, float], n: int) -> tuple[np.ndarray, float]:
    """n interior points of a uniform grid whose end points (excluded) are the walls."""
    h = (domain[1] - domain[0]) / (n + 1)
    return domain[0] + h * np.arange(1, n + 1), h


def _solve_levels(v_fn, domain, n, n_levels, hbar, mass, want_vectors: bool):
    x, h = _grid(domain, n)
    coupling = hbar ** 2 / (2 * mass * h ** 2)
    diag = 2 * coupling + v_fn(x)
    off = -coupling * np.ones(n - 1)
    k = min(n_levels, n)
    if want_vectors:
        vals, vecs = eigh_tridiagonal(diag, off, select="i", select_range=(0, k - 1))
        return x, h, vals, vecs
    vals = eigh_tridiagonal(diag, off, eigvals_only=True, select="i", select_range=(0, k - 1))
    return x, h, vals, None


def build_potential(key: str, params: dict[str, float], hbar: float, mass: float,
                    custom_formula: str | None = None) -> tuple[str, Callable[[np.ndarray], np.ndarray], dict[str, float]]:
    """(formula text, V(x) as a numpy function, the parameter values used) for a library key or `custom`."""
    spec = POTENTIALS.get(key)
    if spec is None:
        raise ValueError(f"Unknown potential {key!r}; choose from {', '.join(POTENTIALS)}.")
    # for the custom potential the formula is whatever was typed -- an empty box is an error, not a silent
    # fallback to the default potential
    formula = custom_formula if key == "custom" and custom_formula is not None else spec.formula
    values = {**spec.params, **{k: float(v) for k, v in (params or {}).items()}}
    names = ["x", "hbar", "m", *values]
    expr = safe_expression(formula, names)
    fn = vectorized(expr, "x", {**values, "hbar": hbar, "m": mass})
    return formula, fn, values


def _count_nodes(psi: np.ndarray) -> int:
    scale = np.max(np.abs(psi))
    significant = psi[np.abs(psi) > 1e-6 * scale]
    return int(np.sum(significant[:-1] * significant[1:] < 0))


def solve_eigenstates(key: str, params: dict[str, float] | None = None, domain: tuple[float, float] | None = None,
                      n_points: int = 800, n_levels: int | None = None, hbar: float = 1.0, mass: float = 1.0,
                      custom_formula: str | None = None, tolerance: float = 1e-3,
                      progress: ProgressFn | None = None) -> EigenResult:
    """Lowest levels of the chosen potential, with convergence and verification (see the module docstring).

    `tolerance` is the relative error allowed against an analytic spectrum; it applies to the best (extrapolated
    where possible) energies. ValueError for unusable settings; ExpressionError for a bad custom formula."""
    spec = POTENTIALS.get(key)
    if spec is None:
        raise ValueError(f"Unknown potential {key!r}; choose from {', '.join(POTENTIALS)}.")
    if spec.atomic_units:
        hbar, mass = 1.0, 1.0
    if not (np.isfinite(hbar) and hbar > 0 and np.isfinite(mass) and mass > 0):
        raise ValueError("hbar and the mass must be positive numbers.")
    chosen = domain or spec.domain
    domain = (float(chosen[0]), float(chosen[1]))
    if not domain[0] < domain[1]:
        raise ValueError("The interval must have its left end below its right end.")
    n_points = int(n_points)
    if not MIN_POINTS <= n_points <= MAX_POINTS:
        raise ValueError(f"Use between {MIN_POINTS} and {MAX_POINTS} grid points.")
    n_levels = spec.n_levels if n_levels is None else int(n_levels)          # 0 is an error, not "use the default"
    if not 1 <= n_levels <= MAX_LEVELS:
        raise ValueError(f"Ask for between 1 and {MAX_LEVELS} levels.")
    n_levels = min(n_levels, n_points - 1)

    formula, v_fn, values = build_potential(key, params or {}, hbar, mass, custom_formula)

    report(progress, f"Solving on {n_points} points", 0.1)
    x, h, e1, vecs = _solve_levels(v_fn, domain, n_points, n_levels, hbar, mass, want_vectors=True)
    report(progress, f"Solving on {2 * n_points} points", 0.4)
    _, _, e2, _ = _solve_levels(v_fn, domain, 2 * n_points, n_levels, hbar, mass, want_vectors=False)
    report(progress, f"Solving on {4 * n_points} points", 0.7)
    _, _, e3, _ = _solve_levels(v_fn, domain, 4 * n_points, n_levels, hbar, mass, want_vectors=False)
    report(progress, "Checking the result", 0.95)

    # measured convergence order and, where it is sensible, Richardson extrapolation
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = (e1 - e2) / (e2 - e3)
        orders = np.where(ratio > 1.5, np.log2(ratio), np.nan)
    sensible = np.isfinite(orders) & (orders > 1.6) & (orders < 2.4)
    best = e3 + np.where(sensible, (e3 - e2) / (2.0 ** 2 - 1.0), 0.0)        # extrapolate the finest pair with p = 2
    states = vecs / np.sqrt(h * np.sum(vecs ** 2, axis=0))
    for j in range(states.shape[1]):                                          # a fixed sign convention
        states[:, j] *= 1.0 if states[np.argmax(np.abs(states[:, j])), j] > 0 else -1.0
    v = v_fn(x)

    observables = _observables(x, h, v, states, e1, hbar, mass, domain, v_fn)

    analytic = None
    notes: list[str] = []
    if spec.analytic is not None:
        try:
            analytic = np.asarray(spec.analytic(n_levels, values, hbar, mass, domain), dtype=float)
        except Exception as e:  # noqa: BLE001 -- a failure in the reference formula must not hide the numerics
            notes.append(f"Could not evaluate the analytic spectrum: {e}")

    checks = _verify(spec, x, h, states, observables, e1, best, orders, sensible, analytic, tolerance, hbar, n_points, v)
    if analytic is not None and len(analytic) < n_levels:
        notes.append(f"Only {len(analytic)} level(s) are bound for these parameters; the rest are box states, "
                     "not physical bound states, and are not compared.")
    if not np.all(sensible):
        notes.append("Levels whose measured convergence order is not close to 2 are shown on the grid you chose, "
                     "without extrapolation (typical for a singular potential such as 1/r on a uniform grid).")
    return EigenResult(key=key, formula=formula, params=values, hbar=hbar, mass=mass, x=x, potential=v,
                       energies=best, raw_energies=e1, states=states, observables=observables, orders=orders,
                       extrapolated=sensible, analytic=analytic, checks=checks, n_points=n_points,
                       domain=domain, notes=notes)


def _observables(x, h, v, states, energies, hbar, mass, domain, v_fn) -> list[StateObservables]:
    out = []
    mid = 0.5 * (domain[0] + domain[1])
    symmetric = bool(np.allclose(v_fn(2 * mid - x), v, rtol=1e-9, atol=1e-12))
    for j in range(states.shape[1]):
        psi = states[:, j]
        prob = psi ** 2 * h
        mean_x = float(np.sum(prob * x))
        var_x = float(np.sum(prob * x ** 2) - mean_x ** 2)
        mean_v = float(np.sum(prob * v))
        mean_t = float(energies[j] - mean_v)
        # <p> = 0 for a real wavefunction, so (Δp)^2 = <p²> = 2m<T>
        std_p = float(np.sqrt(max(2 * mass * mean_t, 0.0)))
        parity = None
        if symmetric:
            reflected = psi[::-1]
            parity = "even" if np.sum(psi * reflected) > 0 else "odd"
        edge = float(np.sum(prob[: max(1, len(x) // 100)]) + np.sum(prob[-max(1, len(x) // 100):]))
        out.append(StateObservables(float(energies[j]), mean_x, float(np.sqrt(max(var_x, 0.0))), std_p, mean_v,
                                    mean_t, _count_nodes(psi), parity, edge))
    return out


def _verify(spec, x, h, states, observables, raw, best, orders, sensible, analytic, tolerance, hbar, n_points,
            v) -> CheckList:
    checks = CheckList()
    gram = states.T @ states * h
    err = float(np.max(np.abs(gram - np.eye(states.shape[1]))))
    checks.add("Orthonormal states", err < 1e-9, f"largest |<ψi|ψj> - δij| = {err:.2e}")

    ascending = bool(np.all(np.diff(raw) > 0))
    checks.add("Energies ascending and distinct", ascending, "E0 < E1 < ..." if ascending else "levels out of order")

    wrong = [(j, o.nodes) for j, o in enumerate(observables) if o.nodes != j]
    checks.add("Node counts follow the oscillation theorem", not wrong,
               "state n has n nodes for every level" if not wrong else
               "state(s) with the wrong number of nodes: " + ", ".join(f"n={j} has {nd}" for j, nd in wrong))

    if not spec.hard_walls:
        # only states whose energy is below the potential at both ends are BOUND in the box's interior; a
        # level above that is a standing wave of the box itself and is expected to reach the walls
        wall_height = min(float(v[0]), float(v[-1]))
        bound = [o for o in observables if o.energy < wall_height]
        if bound:
            leak = max(o.edge_probability for o in bound)
            checks.add("Bound states are contained in the box", leak < 1e-6,
                       f"largest probability within 1% of an end = {leak:.1e} over {len(bound)} bound level(s)" +
                       ("" if leak < 1e-6 else " -- widen the interval, or the walls are changing the answer"))
        else:
            checks.add("Bound states are contained in the box", False,
                       "no level lies below the potential at the ends of the interval: nothing is bound here")

    # the grid represents Δx and <p²> only to O(h²), so a state that saturates the bound (the oscillator's
    # ground state) may land a hair under 1: allow that much, and no more
    slack = max(1e-6, 20.0 / n_points ** 2)
    worst = min(o.std_x * o.std_p / (hbar / 2) for o in observables)
    checks.add("Uncertainty relation Δx Δp ≥ ħ/2", worst >= 1 - slack,
               f"smallest Δx Δp / (ħ/2) = {worst:.6f} (grid allowance {slack:.0e})")

    finite = np.isfinite(orders)
    if finite.any():
        checks.add("Grid convergence is second order", bool(np.all(sensible[finite])),
                   "measured order per level: " + ", ".join("n/a" if not np.isfinite(o) else f"{o:.2f}" for o in orders))

    if analytic is not None and len(analytic):
        k = len(analytic)
        scale = np.maximum(np.abs(analytic), 1e-12)
        rel = np.abs(best[:k] - analytic) / scale
        checks.add(f"Matches the analytic spectrum (tolerance {tolerance:g})", bool(np.all(rel <= tolerance)),
                   "relative error per level: " + ", ".join(f"{r:.1e}" for r in rel))
    return checks


def analytic_table(result: EigenResult) -> list[dict]:
    """Rows comparing the computed and analytic energies, for display and export."""
    rows = []
    for j in range(result.n_levels):
        a = float(result.analytic[j]) if result.analytic is not None and j < len(result.analytic) else None
        row = {"n": j, "E (computed)": float(result.energies[j]), "E (your grid)": float(result.raw_energies[j]),
               "order": None if not np.isfinite(result.orders[j]) else float(result.orders[j])}
        if a is not None:
            row["E (analytic)"] = a
            row["relative error"] = abs(result.energies[j] - a) / max(abs(a), 1e-12)
        rows.append(row)
    return rows


def level_spacings(energies: np.ndarray) -> np.ndarray:
    return np.diff(energies)


def tunnelling_splitting(result: EigenResult) -> float | None:
    """E1 - E0 for the lowest pair of a symmetric double well -- the tunnelling splitting -- or None if the
    potential is not symmetric or has fewer than two levels."""
    if result.n_levels < 2 or result.observables[0].parity is None:
        return None
    return float(result.energies[1] - result.energies[0])
