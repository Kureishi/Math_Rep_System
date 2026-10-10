"""
A quantum wavepacket moving in one dimension -- evolved, and checked against exact results -- Streamlit-free.

    i hbar dψ/dt = [ -(hbar²/2m) d²/dx² + V(x) ] ψ

TWO INDEPENDENT METHODS, both unitary:
  * split-step Fourier (Strang splitting): half a step of V, a full step of the kinetic term in momentum
    space, half a step of V. Periodic box.
  * Crank–Nicolson: (1 + iΔt H/2ħ) ψ' = (1 - iΔt H/2ħ) ψ with a finite-difference H, solved with a
    sparse LU factorisation computed once. Hard walls at the box ends.
Running both and comparing the final density is the cross-check: they share nothing but the potential, so
agreement is evidence that neither the splitting nor the discretisation is producing the result.

WHAT IS CHECKED:
  * the norm stays 1 (both methods are unitary, so drift is rounding only);
  * the energy <H> is conserved (the potential is time-independent);
  * Ehrenfest's theorem, d<x>/dt = <p>/m, using <x> and <p> recorded at every few steps;
  * the packet never reaches the box ends (which would be wrap-around, or a wall, not physics);
  * for a potential made of flat pieces (a barrier, a step, a double barrier, a well), the TRANSMISSION
    measured from the simulation against the exact stationary-scattering result.

THE SCATTERING PREDICTION. For a piecewise-constant potential the stationary problem has an exact solution
by transfer matrices: across each flat piece (ψ, ψ') is multiplied by [[cos kd, sin kd / k], [-k sin kd, cos kd]]
(cosh/sinh when E < V), and matching to an incoming plane wave plus reflected wave on the left and a
transmitted wave on the right gives T(E) and R(E) with T + R = 1 (checked). A wavepacket is a superposition of
plane waves with weights |φ(k)|² ∝ exp(-2σ²(k - k0)²), so the transmission it should show is the
weighted average of T(E(k)). That average, not T at the central energy, is what the simulation is compared
to: a packet narrow in space is broad in energy and straddles a barrier's threshold.

Probabilities are only compared once the packet has left the scattering region (probability still inside
it is reported and checked).
"""
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import scipy.sparse as sp_sparse
from scipy.sparse.linalg import splu

from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList, safe_expression, vectorized

MIN_POINTS = 128
MAX_POINTS = 8192
MAX_STEPS = 60_000
MAX_FRAMES = 240


# ------------------------------------------------------------------------------------------ potentials

@dataclass(frozen=True)
class DynPotentialSpec:
    key: str
    label: str
    description: str
    params: dict[str, float]


DYNAMIC_POTENTIALS: dict[str, DynPotentialSpec] = {s.key: s for s in [
    DynPotentialSpec("free", "Free particle", "V = 0: the packet spreads and nothing else.", {}),
    DynPotentialSpec("barrier", "Rectangular barrier", "V = V0 for |x - c| < w/2: tunnelling and reflection.",
                     {"V0": 2.0, "width": 1.0, "center": 0.0}),
    DynPotentialSpec("step", "Potential step", "V = V0 for x > c.", {"V0": 1.0, "center": 0.0}),
    DynPotentialSpec("double_barrier", "Double barrier",
                     "Two barriers of height V0 and width w, a gap apart: resonant tunnelling.",
                     {"V0": 3.0, "width": 0.5, "gap": 1.5, "center": 0.0}),
    DynPotentialSpec("well", "Finite well", "V = -V0 for |x - c| < w/2: scattering and quasi-bound resonances.",
                     {"V0": 2.0, "width": 2.0, "center": 0.0}),
    DynPotentialSpec("harmonic", "Harmonic trap", "V = ½ m w² (x - c)²: a coherent packet oscillates without spreading.",
                     {"w": 1.0, "center": 0.0}),
    DynPotentialSpec("custom", "Custom V(x)", "Any expression in x (and hbar, m, your own parameters).", {}),
]}


@dataclass
class PotentialModel:
    values: np.ndarray                       # V on the grid
    boundaries: list[float] | None           # edges between flat pieces (None if V is not piecewise constant)
    levels: list[float] | None               # V on each piece: len(boundaries) + 1 values
    description: str
    params: dict[str, float]
    region: tuple[float, float] | None       # the scattering region (first boundary, last boundary)


def build_dynamic_potential(key: str, params: dict[str, float], x: np.ndarray, hbar: float, mass: float,
                            custom_formula: str | None = None) -> PotentialModel:
    spec = DYNAMIC_POTENTIALS.get(key)
    if spec is None:
        raise ValueError(f"Unknown potential {key!r}; choose from {', '.join(DYNAMIC_POTENTIALS)}.")
    p = {**spec.params, **{k: float(v) for k, v in (params or {}).items()}}
    c = p.get("center", 0.0)
    if key == "free":
        return PotentialModel(np.zeros_like(x), None, None, spec.description, p, None)
    if key == "barrier":
        w = _positive(p, "width")
        b = [c - w / 2, c + w / 2]
        return PotentialModel(np.where(np.abs(x - c) < w / 2, p["V0"], 0.0), b, [0.0, p["V0"], 0.0],
                              spec.description, p, (b[0], b[1]))
    if key == "step":
        return PotentialModel(np.where(x > c, p["V0"], 0.0), [c], [0.0, p["V0"]], spec.description, p, (c, c))
    if key == "double_barrier":
        w, g = _positive(p, "width"), _positive(p, "gap")
        b = [c - g / 2 - w, c - g / 2, c + g / 2, c + g / 2 + w]
        v = np.where(((x > b[0]) & (x < b[1])) | ((x > b[2]) & (x < b[3])), p["V0"], 0.0)
        return PotentialModel(v, b, [0.0, p["V0"], 0.0, p["V0"], 0.0], spec.description, p, (b[0], b[3]))
    if key == "well":
        w = _positive(p, "width")
        b = [c - w / 2, c + w / 2]
        return PotentialModel(np.where(np.abs(x - c) < w / 2, -p["V0"], 0.0), b, [0.0, -p["V0"], 0.0],
                              spec.description, p, (b[0], b[1]))
    if key == "harmonic":
        return PotentialModel(0.5 * mass * p["w"] ** 2 * (x - c) ** 2, None, None, spec.description, p, None)
    formula = custom_formula or "0"
    names = ["x", "hbar", "m", *p]
    fn = vectorized(safe_expression(formula, names), "x", {**p, "hbar": hbar, "m": mass})
    return PotentialModel(fn(x), None, None, f"V(x) = {formula}", p, None)


def _snap_to_grid(model: PotentialModel, x: np.ndarray) -> PotentialModel:
    """Replace a flat-piece potential's nominal boundaries by those the grid actually has."""
    if model.boundaries is None:
        return model
    eff = effective_piecewise(x, model.values)
    if eff is None or not eff[0]:
        return model
    boundaries, levels = eff
    return PotentialModel(model.values, boundaries, levels, model.description, model.params,
                          (boundaries[0], boundaries[-1]))


def _positive(p: dict[str, float], name: str) -> float:
    if not p[name] > 0:
        raise ValueError(f"{name} must be positive.")
    return p[name]


# ------------------------------------------------------------------------------------------ scattering

def _wavenumber(energy: float, v: float, hbar: float, mass: float) -> complex:
    return np.sqrt(complex(2 * mass * (energy - v))) / hbar


def transmission_reflection(energy: float, boundaries: Sequence[float], levels: Sequence[float],
                            hbar: float = 1.0, mass: float = 1.0) -> tuple[float, float]:
    """(T, R) for a plane wave of the given energy incident from the left on a piecewise-constant potential
    (see the module docstring). `levels` has one more entry than `boundaries`. If the energy is below the
    potential far to the left nothing can arrive (0, 0); below it far to the right nothing can leave (0, 1)."""
    if len(levels) != len(boundaries) + 1:
        raise ValueError("levels must have exactly one more entry than boundaries.")
    k_left = _wavenumber(energy, levels[0], hbar, mass)
    k_right = _wavenumber(energy, levels[-1], hbar, mass)
    if k_left.real <= 0:
        return 0.0, 0.0
    if k_right.real <= 0:
        return 0.0, 1.0
    if not boundaries:
        return 1.0, 0.0
    # propagate (psi, psi') from the first boundary to the last through each interior flat piece
    m_total = np.eye(2, dtype=complex)
    for j in range(1, len(boundaries)):
        d = boundaries[j] - boundaries[j - 1]
        k = _wavenumber(energy, levels[j], hbar, mass)
        if abs(k) < 1e-12:
            piece = np.array([[1.0, d], [0.0, 1.0]], dtype=complex)
        else:
            piece = np.array([[np.cos(k * d), np.sin(k * d) / k], [-k * np.sin(k * d), np.cos(k * d)]], dtype=complex)
        m_total = piece @ m_total
    # Matching at the two ends: on the left psi = 1 + r and psi' = i k_L (1 - r); on the right psi = t and
    # psi' = i k_R t. Propagating the left pair through M must give the right pair:
    #     M [1 + r, i k_L (1 - r)] = [t, i k_R t]
    # The left pair is (1, i k_L) + r (1, -i k_L), so with c0 = M (1, i k_L) and c1 = M (1, -i k_L):
    #     c0 + r c1 = t (1, i k_R)      ->     r c1 - t (1, i k_R) = -c0     (two equations, unknowns r and t)
    c0 = m_total @ np.array([1.0, 1j * k_left])
    c1 = m_total @ np.array([1.0, -1j * k_left])
    system = np.array([[c1[0], -1.0], [c1[1], -1j * k_right]], dtype=complex)
    r, t = np.linalg.solve(system, -c0)
    return float((k_right.real / k_left.real) * abs(t) ** 2), float(abs(r) ** 2)


def transmission_curve(energies: np.ndarray, boundaries: Sequence[float], levels: Sequence[float],
                       hbar: float = 1.0, mass: float = 1.0) -> np.ndarray:
    return np.array([transmission_reflection(float(e), boundaries, levels, hbar, mass)[0] for e in energies])


def packet_transmission(k0: float, sigma: float, boundaries: Sequence[float], levels: Sequence[float],
                        hbar: float = 1.0, mass: float = 1.0) -> tuple[float, float, float]:
    """(transmitted, reflected, other) probabilities a Gaussian packet should show after scattering, from
    the exact T(E) averaged over the packet's momentum distribution. `other` is the weight of components
    moving AWAY from the potential (k < 0), which never arrive."""
    width_k = 1.0 / (2.0 * sigma)
    ks = np.linspace(k0 - 8 * width_k, k0 + 8 * width_k, 2001)
    weight = np.exp(-2.0 * sigma ** 2 * (ks - k0) ** 2)
    weight /= np.trapezoid(weight, ks)
    forward = ks > 0
    ts, rs = [], []
    for k in ks:
        if k <= 0:
            ts.append(0.0)
            rs.append(0.0)
            continue
        t, r = transmission_reflection(levels[0] + hbar ** 2 * k ** 2 / (2 * mass), boundaries, levels, hbar, mass)
        ts.append(t)
        rs.append(r)
    t_sum = float(np.trapezoid(weight * np.array(ts), ks))
    r_sum = float(np.trapezoid(weight * np.array(rs), ks))
    other = float(np.trapezoid(weight * (~forward), ks))
    return t_sum, r_sum, other


# ------------------------------------------------------------------------------------------ evolution

@dataclass
class DynamicsResult:
    method: str
    x: np.ndarray
    potential: np.ndarray
    frame_times: np.ndarray
    density: np.ndarray                      # (n_frames, N): |ψ|²
    real_part: np.ndarray                    # (n_frames, N): Re ψ
    series_times: np.ndarray
    norm: np.ndarray
    energy: np.ndarray
    mean_x: np.ndarray
    mean_p: np.ndarray
    std_x: np.ndarray
    region_probability: np.ndarray | None    # (n_frames, 3): left / inside / right of the scattering region
    final_probabilities: dict[str, float]
    predicted: dict[str, float] | None
    checks: CheckList
    spec: dict
    hbar: float
    mass: float
    dt: float
    steps: int
    cross_check_distance: float | None = None
    notes: list[str] = field(default_factory=list)


def gaussian_packet(x: np.ndarray, x0: float, sigma: float, k0: float) -> np.ndarray:
    psi = np.exp(-((x - x0) ** 2) / (4 * sigma ** 2) + 1j * k0 * x)
    dx = x[1] - x[0]
    return psi / np.sqrt(np.sum(np.abs(psi) ** 2) * dx)


class _SplitStep:
    name = "split-step Fourier"

    def __init__(self, x, v, dt, hbar, mass):
        n, dx = len(x), x[1] - x[0]
        self.k = 2 * np.pi * np.fft.fftfreq(n, d=dx)
        self.half_v = np.exp(-0.5j * v * dt / hbar)
        self.kinetic = np.exp(-0.5j * hbar * self.k ** 2 * dt / mass)
        self.hbar, self.mass, self.dx, self.v = hbar, mass, dx, v

    def step(self, psi):
        psi = self.half_v * psi
        psi = np.fft.ifft(self.kinetic * np.fft.fft(psi))
        return self.half_v * psi

    def observables(self, psi):
        phi = np.fft.fft(psi) / np.sqrt(len(psi))                 # unitary normalisation: sum|phi|² = sum|psi|²
        weight = np.abs(phi) ** 2
        t = float(np.sum(self.hbar ** 2 * self.k ** 2 / (2 * self.mass) * weight) * self.dx)
        p = float(np.sum(self.hbar * self.k * weight) * self.dx)
        return t, p


class _CrankNicolson:
    name = "Crank–Nicolson"

    def __init__(self, x, v, dt, hbar, mass):
        n, dx = len(x), x[1] - x[0]
        c = hbar ** 2 / (2 * mass * dx ** 2)
        self.diag = 2 * c + v
        self.off = -c
        alpha = 1j * dt / (2 * hbar)
        h = sp_sparse.diags([np.full(n - 1, self.off), self.diag, np.full(n - 1, self.off)], [-1, 0, 1], format="csc")
        ident = sp_sparse.identity(n, format="csc", dtype=complex)
        self.lu = splu((ident + alpha * h).tocsc())
        self.rhs = (ident - alpha * h).tocsr()
        self.hbar, self.mass, self.dx, self.v, self.c = hbar, mass, dx, v, c

    def step(self, psi):
        return self.lu.solve(self.rhs @ psi)

    def observables(self, psi):
        left, right = np.roll(psi, 1), np.roll(psi, -1)
        left[0] = 0.0
        right[-1] = 0.0
        t_psi = 2 * self.c * psi - self.c * (left + right)
        t = float(np.real(np.vdot(psi, t_psi)) * self.dx)
        grad = (right - left) / (2 * self.dx)
        p = float(np.real(np.vdot(psi, -1j * self.hbar * grad)) * self.dx)
        return t, p


METHODS: dict[str, Any] = {"split-step": _SplitStep, "crank-nicolson": _CrankNicolson}


def suggest_total_time(x0: float, k0: float, region: tuple[float, float] | None, hbar: float, mass: float,
                       domain: tuple[float, float], sigma: float, transmitted: float = 1.0,
                       reflected: float = 1.0) -> float:
    """How long to run.

    With a scattering region: the first time at which every packet that matters has CLEARED it, so that T and R
    can be read -- its centre is 3 packet-widths (at that time, spreading included) beyond the region plus the
    2σ margin used to call something "inside". The transmitted packet matters if `transmitted` is non-negligible,
    the reflected one if `reflected` is. Without a region: half a crossing of the box.

    A packet cannot always be separated cleanly: if its momentum spread is a large fraction of its mean
    momentum, the front of the spreading tail travels as fast as its centre and never clears. Then the answer
    is the longest time at which the packet still fits in the box, and the "left the scattering region" check
    will say so; a wider packet (larger σ) is the remedy."""
    v = abs(hbar * k0 / mass)
    if v < 1e-9:
        return float(2 * np.pi * mass / hbar)                              # no drift: one period of a unit trap
    direction = 1.0 if k0 > 0 else -1.0
    horizon = 0.5 * (domain[1] - domain[0]) / v
    wanted = horizon
    if region is not None:
        near, far = (region[0], region[1]) if direction > 0 else (region[1], region[0])
        for t in np.linspace(0.05, 6 * horizon, 6000):
            width = 3.0 * free_packet_width(sigma, t, hbar, mass)
            ok_t = transmitted < 1e-3 or direction * (x0 + direction * v * t - far) - width > 2 * sigma
            ok_r = reflected < 1e-3 or v * t - abs(near - x0) - width > 2 * sigma
            if ok_t and ok_r:
                wanted = float(t) * 1.05
                break
    for t in np.linspace(wanted, wanted * 0.1, 120):                          # the longest time at which it still fits
        reach = 4.5 * free_packet_width(sigma, t, hbar, mass)
        centre = x0 + direction * v * t
        if domain[0] + reach < min(centre, x0) and max(centre, x0) < domain[1] - reach:
            return float(t)
    return float(wanted * 0.1)


def effective_piecewise(x: np.ndarray, v: np.ndarray) -> tuple[list[float], list[float]] | None:
    """The potential AS THE GRID REPRESENTS IT, as (boundaries, levels): each change of V between neighbouring
    grid points is a boundary at the midpoint between them. A barrier "of width 1" occupies a whole number
    of grid cells, so on a coarse grid it is a little wider or narrower than asked for; the exact scattering
    result must be computed for the barrier that is actually simulated, not the one that was typed.
    None if V takes more than 40 distinct pieces (not a flat-piece potential)."""
    dx = x[1] - x[0]
    change = np.nonzero(~np.isclose(v[1:], v[:-1], rtol=0, atol=1e-12))[0]
    if len(change) > 40:
        return None
    boundaries = [float(x[j] + dx / 2) for j in change]
    levels = [float(v[0])] + [float(v[j + 1]) for j in change]
    return boundaries, levels


def recommended_steps(total_time: float, domain: tuple[float, float], n_points: int, hbar: float, mass: float) -> int:
    """Time steps for the split-step method to be accurate on this grid.

    Its kinetic factor rotates the highest-momentum component (k = π/Δx) by ħk²Δt/2m per step. Measured
    energy drift is ~1e-6 when the half-step phase there is at most about 3 rad and grows to order 1 when it
    is ~10 (a sharp-edged potential feeds a little probability into those components, and a wrongly rotated
    high-k tail carries a lot of energy). Δt is chosen to keep that phase at 3."""
    dx = (domain[1] - domain[0]) / n_points
    k_max = np.pi / dx
    dt_max = 12.0 * mass / (hbar * k_max ** 2)
    return int(min(max(np.ceil(total_time / dt_max), 400), MAX_STEPS))


def evolve_wavepacket(potential: str = "barrier", params: dict[str, float] | None = None,
                      x0: float = -10.0, sigma: float = 1.5, k0: float = 2.0,
                      domain: tuple[float, float] = (-30.0, 30.0), n_points: int = 2048,
                      total_time: float | None = None, steps: int | None = None, n_frames: int = 60,
                      method: str = "split-step", hbar: float = 1.0, mass: float = 1.0,
                      custom_formula: str | None = None, cross_check: bool = True,
                      transmission_tolerance: float = 0.02, progress: ProgressFn | None = None) -> DynamicsResult:
    """Evolve a Gaussian packet and verify the run (see the module docstring)."""
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; choose from {', '.join(METHODS)}.")
    if not (hbar > 0 and mass > 0):
        raise ValueError("hbar and the mass must be positive.")
    if not sigma > 0:
        raise ValueError("The packet width σ must be positive.")
    if not domain[0] < domain[1]:
        raise ValueError("The box must have its left end below its right end.")
    if not MIN_POINTS <= n_points <= MAX_POINTS:
        raise ValueError(f"Use between {MIN_POINTS} and {MAX_POINTS} grid points.")
    if not 2 <= n_frames <= MAX_FRAMES:
        raise ValueError(f"Use between 2 and {MAX_FRAMES} animation frames.")
    if not domain[0] < x0 < domain[1]:
        raise ValueError("The packet must start inside the box.")

    x = np.linspace(domain[0], domain[1], n_points, endpoint=False)
    dx = x[1] - x[0]
    pot = _snap_to_grid(build_dynamic_potential(potential, params or {}, x, hbar, mass, custom_formula), x)
    if total_time is None:
        t_w, r_w = 1.0, 1.0
        if pot.boundaries is not None and pot.levels is not None and k0 > 0:
            t_w, r_w, _ = packet_transmission(k0, sigma, pot.boundaries, pot.levels, hbar, mass)
        total_time = suggest_total_time(x0, k0, pot.region, hbar, mass, domain, sigma, t_w, r_w)
    if not total_time > 0:
        raise ValueError("The total time must be positive.")
    if steps is None:
        steps = recommended_steps(total_time, domain, n_points, hbar, mass)
    if not 10 <= steps <= MAX_STEPS:
        raise ValueError(f"Use between 10 and {MAX_STEPS:,} time steps.")
    dt = total_time / steps

    k_nyquist = np.pi / dx
    notes: list[str] = []
    if abs(k0) + 6 / (2 * sigma) > 0.5 * k_nyquist:
        notes.append("The packet's momentum is a large fraction of the grid's limit (π/Δx); use more grid points "
                     "or a smaller k0.")

    psi0 = gaussian_packet(x, x0, sigma, k0)
    main = _run(METHODS[method], x, pot, psi0, dt, steps, n_frames, hbar, mass, progress, 0.0,
                0.5 if cross_check else 1.0, sigma)
    cross_distance = None
    if cross_check:
        other_name = "crank-nicolson" if method == "split-step" else "split-step"
        other = _run(METHODS[other_name], x, pot, psi0, dt, steps, n_frames, hbar, mass, progress, 0.5, 1.0, sigma)
        cross_distance = float(0.5 * np.sum(np.abs(main["density"][-1] - other["density"][-1])) * dx)

    result_checks = _verify(main, x, dx, pot, hbar, mass, sigma, k0, x0, transmission_tolerance, cross_distance)
    final = main["final_probabilities"]
    predicted = None
    if pot.boundaries is not None and pot.levels is not None and k0 > 0:
        t, r, other_w = packet_transmission(k0, sigma, pot.boundaries, pot.levels, hbar, mass)
        predicted = {"transmitted": t, "reflected": r, "moving_away": other_w}
        _check_transmission(result_checks, final, predicted, transmission_tolerance, x0=x0, k0=k0, hbar=hbar, mass=mass,
                            far_end=pot.region[1] if pot.region else x0, run_time=total_time)
    return DynamicsResult(
        method=METHODS[method].name, x=x, potential=pot.values, frame_times=main["frame_times"],
        density=main["density"], real_part=main["real"], series_times=main["series_t"], norm=main["norm"],
        energy=main["energy"], mean_x=main["mean_x"], mean_p=main["mean_p"], std_x=main["std_x"],
        region_probability=main["region"], final_probabilities=final, predicted=predicted, checks=result_checks,
        spec={"potential": potential, "params": pot.params, "x0": x0, "sigma": sigma, "k0": k0, "domain": domain,
              "n_points": n_points, "total_time": total_time, "steps": steps, "description": pot.description,
              "boundaries": pot.boundaries, "levels": pot.levels},
        hbar=hbar, mass=mass, dt=dt, steps=steps, cross_check_distance=cross_distance, notes=notes)


def _run(solver_cls: Any, x, pot, psi0, dt, steps, n_frames, hbar, mass, progress, p_lo, p_hi, margin: float) -> dict:
    dx = x[1] - x[0]
    solver = solver_cls(x, pot.values, dt, hbar, mass)
    frame_steps = np.unique(np.round(np.linspace(0, steps, n_frames)).astype(int))
    record_every = max(1, steps // 2000)
    psi = psi0.astype(complex)
    frames, reals, frame_t, region_rows = [], [], [], []
    series: dict[str, list[float]] = {"t": [], "norm": [], "energy": [], "x": [], "p": [], "sx": []}
    boundaries = None if pot.region is None else (pot.region[0] - 2 * margin, pot.region[1] + 2 * margin)
    next_frame = 0

    def record(step_index: int) -> None:
        prob = np.abs(psi) ** 2
        t_kin, p_mean = solver.observables(psi)
        mean_x = float(np.sum(prob * x) * dx)
        series["t"].append(step_index * dt)
        series["norm"].append(float(np.sum(prob) * dx))
        series["energy"].append(t_kin + float(np.sum(prob * pot.values) * dx))
        series["x"].append(mean_x)
        series["p"].append(p_mean)
        series["sx"].append(float(np.sqrt(max(np.sum(prob * x ** 2) * dx - mean_x ** 2, 0.0))))

    for i in range(steps + 1):
        if i % record_every == 0 or i == steps:
            record(i)
        if next_frame < len(frame_steps) and i == frame_steps[next_frame]:
            prob = np.abs(psi) ** 2
            frames.append(prob.copy())
            reals.append(psi.real.copy())
            frame_t.append(i * dt)
            if boundaries is not None:
                lo, hi = boundaries
                region_rows.append([float(np.sum(prob[x < lo]) * dx), float(np.sum(prob[(x >= lo) & (x <= hi)]) * dx),
                                    float(np.sum(prob[x > hi]) * dx)])
            report(progress, f"{solver.name}: step {i:,} of {steps:,}", p_lo + (p_hi - p_lo) * i / steps)
            next_frame += 1
        if i < steps:
            psi = solver.step(psi)

    final_prob = np.abs(psi) ** 2
    final = {"norm": float(np.sum(final_prob) * dx)}
    if boundaries is not None:
        lo, hi = boundaries
        final.update({"left": float(np.sum(final_prob[x < lo]) * dx),
                      "inside": float(np.sum(final_prob[(x >= lo) & (x <= hi)]) * dx),
                      "right": float(np.sum(final_prob[x > hi]) * dx)})
    return {"density": np.array(frames), "real": np.array(reals), "frame_times": np.array(frame_t),
            "series_t": np.array(series["t"]), "norm": np.array(series["norm"]), "energy": np.array(series["energy"]),
            "mean_x": np.array(series["x"]), "mean_p": np.array(series["p"]), "std_x": np.array(series["sx"]),
            "region": np.array(region_rows) if region_rows else None, "final_probabilities": final,
            "solver": solver.name}


def _verify(run: dict, x, dx, pot, hbar, mass, sigma, k0, x0, tol, cross_distance) -> CheckList:
    checks = CheckList()
    drift = float(np.max(np.abs(run["norm"] - 1.0)))
    checks.add(f"Norm conserved ({run['solver']})", drift < 1e-8, f"largest |∫|ψ|² - 1| = {drift:.1e}")

    e = run["energy"]
    scale = max(abs(e[0]), abs(run["mean_p"][0]) ** 2 / (2 * mass), 1e-12)
    e_drift = float(np.max(np.abs(e - e[0])) / scale)
    checks.add("Energy ⟨H⟩ conserved", e_drift < 2e-3, f"largest relative drift = {e_drift:.1e}")

    t, xm, pm = run["series_t"], run["mean_x"], run["mean_p"]
    if len(t) > 5:
        velocity = np.gradient(xm, t)
        speed = max(float(np.max(np.abs(pm / mass))), 1e-12)
        ehrenfest = float(np.sqrt(np.mean((velocity[2:-2] - pm[2:-2] / mass) ** 2)) / speed)
        checks.add("Ehrenfest: d⟨x⟩/dt = ⟨p⟩/m", ehrenfest < 2e-2, f"rms mismatch / peak speed = {ehrenfest:.1e}")

    dens = run["density"]
    edge_width = max(1, len(x) // 50)
    edge = float(max(np.max(np.sum(dens[:, :edge_width], axis=1)), np.max(np.sum(dens[:, -edge_width:], axis=1))) * dx)
    checks.add("The packet never reaches the box ends", edge < 1e-4,
               f"largest probability within 2% of an end = {edge:.1e}" +
               ("" if edge < 1e-4 else " -- enlarge the box or shorten the run: the ends (a wrap-around or a wall) "
                                      "are affecting the result"))

    if cross_distance is not None:
        # The methods are not expected to agree EXACTLY: the finite-difference kinetic operator of Crank–Nicolson
        # has dispersion error ω_FD ≈ ω (1 - k²Δx²/12), which accumulates as a phase error
        #     δφ ≈ (ħk²/2m) · T · (kΔx)²/12     at a typical wavenumber k = |k0| + 1/(2σ)
        # over the run time T. The allowance is 2% plus that predicted phase error, so a finer grid is held to
        # a tighter standard and a coarse one is not failed for a difference the discretisation explains.
        k_typ = abs(k0) + 1.0 / (2.0 * sigma)
        run_time = float(run["series_t"][-1])
        phase_error = (hbar * k_typ ** 2 / (2 * mass)) * run_time * (k_typ * dx) ** 2 / 12.0
        allowance = 2e-2 + phase_error
        checks.add("The two methods agree", cross_distance < allowance,
                   f"total-variation distance between the final densities = {cross_distance:.1e} "
                   f"(allowed {allowance:.1e}: 2% plus the {phase_error:.1e} rad dispersion error the finite-difference "
                   "method is expected to accumulate)")
    return checks


def _check_transmission(checks: CheckList, final: dict, predicted: dict, tol: float, *, x0: float, k0: float,
                        hbar: float, mass: float, far_end: float, run_time: float) -> None:
    """Compare the simulated transmission and reflection with the exact prediction -- but only if the
    comparison means something: the run must be long enough for the packet to have reached and crossed the
    region, and the probability near the region must have settled. (A run that stops while the packet is still
    approaching has "no probability near the region" too, trivially; that is not scattering.)"""
    velocity = hbar * k0 / mass
    arrival = (far_end - x0) / velocity if velocity > 0 else float("inf")
    crossed = run_time > arrival
    checks.add("The run lasts long enough for the packet to cross the region", crossed,
               f"the packet's centre reaches the far side at t ≈ {arrival:.3g}; the run lasts {run_time:.3g}" +
               ("" if crossed else " -- run longer"))
    inside = final.get("inside", 0.0)
    # whatever is still near the region is probability that is neither "transmitted" nor "reflected" yet, so
    # it bounds how well T and R can be read: it must be small next to the tolerance they are compared with
    settled = inside < tol / 2
    checks.add("The packet has left the scattering region", settled,
               f"probability within 2σ of the region = {inside:.1e} (must be below {tol / 2:.0e})" +
               ("" if settled else " -- run longer, widen the box, or use a wider packet; a resonance can hold "
                                  "probability for a long time"))
    if crossed and settled:
        err_t = abs(final["right"] - predicted["transmitted"])
        checks.add(f"Transmission matches the exact scattering result (±{tol:g})", err_t <= tol,
                   f"simulated {final['right']:.4f}, predicted {predicted['transmitted']:.4f}")
        err_r = abs(final["left"] - (predicted["reflected"] + predicted["moving_away"]))
        checks.add(f"Reflection matches the exact scattering result (±{tol:g})", err_r <= tol,
                   f"simulated {final['left']:.4f}, predicted {predicted['reflected'] + predicted['moving_away']:.4f}")


def barrier_tunnelling_formula(energy: float, v0: float, width: float, hbar: float = 1.0, mass: float = 1.0) -> float:
    """The textbook transmission through a rectangular barrier (closed form), used to test the transfer matrices:
    T = [1 + V0² sinh²(κa) / (4E(V0 - E))]^-1 for E < V0, and with sin²(qa) / (4E(E - V0)) for E > V0."""
    if energy <= 0:
        return 0.0
    if abs(energy - v0) < 1e-12:
        return 1.0 / (1.0 + mass * v0 * width ** 2 / (2 * hbar ** 2))
    if energy < v0:
        kappa = np.sqrt(2 * mass * (v0 - energy)) / hbar
        return float(1.0 / (1.0 + v0 ** 2 * np.sinh(kappa * width) ** 2 / (4 * energy * (v0 - energy))))
    q = np.sqrt(2 * mass * (energy - v0)) / hbar
    return float(1.0 / (1.0 + v0 ** 2 * np.sin(q * width) ** 2 / (4 * energy * (energy - v0))))


def free_packet_width(sigma: float, t: float, hbar: float = 1.0, mass: float = 1.0) -> float:
    """The exact width of a free Gaussian packet at time t: σ(t) = σ sqrt(1 + (ħt / 2mσ²)²)."""
    return float(sigma * np.sqrt(1.0 + (hbar * t / (2 * mass * sigma ** 2)) ** 2))

