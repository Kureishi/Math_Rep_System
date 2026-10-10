"""
A single qubit on the Bloch sphere: gate sequences, noise, purity and fidelity -- Streamlit-free.

A qubit state is its density matrix ρ = (I + r·σ)/2, where r is the Bloch vector (|r| ≤ 1; |r| = 1 for a pure
state). |0> is the north pole (r = +z), |1> the south pole, |+> is +x, |+i> is +y.

GATES act on r as rotations. Any single-qubit unitary is, up to a global phase that no measurement can see,
U = exp(-i θ n·σ/2): a rotation of the Bloch sphere by θ about the axis n. `axis_angle` extracts (n, θ) from
the matrix, which is what lets the animation turn a gate into a smooth arc instead of a jump, and is checked
by rebuilding U from it.

NOISE is the Lindblad master equation
        dρ/dt = -i[H, ρ] + Σ_k ( L_k ρ L_k† - ½{L_k†L_k, ρ} )
with three standard channels, each optional (rate 0 = off):
    amplitude damping (T1)   L = √γ1 σ-         relaxes toward |0>, γ1 = 1/T1
    pure dephasing (Tφ)      L = √(γφ/2) σz     destroys phase only, γφ = 1/Tφ
    depolarising (rate γd)   L = √(γd/4) σx,y,z shrinks the whole Bloch vector
and the resulting coherence time is 1/T2 = 1/(2 T1) + 1/Tφ (+ γd), so T2 ≤ 2 T1 always.

THE PROPAGATION IS EXACT, not stepped. During a gate the Lindblad equation reduces to a linear equation for
the Bloch vector, dr/dt = M r + b, with M = Ω [n]× - Γ (a rotation at angular speed Ω = θ/τ about n, minus
the decay matrix Γ) and b = (0, 0, γ1). Its solution over any time is one 4x4 matrix exponential, so there is
no time-step error whatever the noise rates.

HOW THAT IS CHECKED, since the Bloch-vector equations are a derivation and could be wrong:
  * the same evolution is computed independently from the density-matrix superoperator (the Lindblad
    equation itself, vectorised and exponentiated) and the final states are compared;
  * idle decay from the start state is compared with the closed forms
        r_z(t) = 1 + (r_z0 - 1) e^{-t/T1},    r_xy(t) = r_xy0 e^{-t/T2};
  * every gate must be unitary, and rebuilt from its axis and angle;
  * the state must stay physical: |r| ≤ 1, purity ≤ 1, ρ positive semidefinite;
  * with no noise, purity and fidelity with the ideal path stay exactly 1.
"""
import re
from collections.abc import Sequence
from typing import Union
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, sqrtm

from modules.progress import ProgressFn, report
from modules.quantum_common import CheckList, ExpressionError, safe_expression

BlochLike = Union[Sequence[float], np.ndarray]          # a Bloch vector: any sequence of three numbers, or an array

PAULI = (np.array([[0, 1], [1, 0]], dtype=complex),
         np.array([[0, -1j], [1j, 0]], dtype=complex),
         np.array([[1, 0], [0, -1]], dtype=complex))
IDENTITY = np.eye(2, dtype=complex)
SIGMA_MINUS = np.array([[0, 1], [0, 0]], dtype=complex)         # |0><1|: takes |1> to |0>
MAX_GATES = 40
SAMPLES_PER_GATE = 24


# ------------------------------------------------------------------------------------------ states

def bloch_to_density(r: BlochLike) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    return 0.5 * (IDENTITY + r[0] * PAULI[0] + r[1] * PAULI[1] + r[2] * PAULI[2])


def density_to_bloch(rho: np.ndarray) -> np.ndarray:
    return np.array([np.real(np.trace(rho @ s)) for s in PAULI])


def purity(r: BlochLike) -> float:
    """Tr ρ² = (1 + |r|²)/2: 1 for a pure state, ½ for the maximally mixed one."""
    return float((1.0 + np.dot(r, r)) / 2.0)


def uhlmann_fidelity(r: BlochLike, s: BlochLike) -> float:
    """The fidelity (Tr √(√ρ σ √ρ))² of two qubit states from their Bloch vectors:
    F = ½(1 + r·s) + ½ √((1 - |r|²)(1 - |s|²)). It reduces to ⟨ψ|ρ|ψ⟩ when σ = |ψ⟩⟨ψ|."""
    r, s = np.asarray(r, dtype=float), np.asarray(s, dtype=float)
    return float(0.5 * (1 + np.dot(r, s)) + 0.5 * np.sqrt(max((1 - np.dot(r, r)) * (1 - np.dot(s, s)), 0.0)))


STATE_PRESETS: dict[str, tuple[float, float, float]] = {
    "|0⟩ (north pole)": (0.0, 0.0, 1.0), "|1⟩ (south pole)": (0.0, 0.0, -1.0),
    "|+⟩ = (|0⟩+|1⟩)/√2": (1.0, 0.0, 0.0), "|−⟩ = (|0⟩−|1⟩)/√2": (-1.0, 0.0, 0.0),
    "|+i⟩ = (|0⟩+i|1⟩)/√2": (0.0, 1.0, 0.0), "|−i⟩ = (|0⟩−i|1⟩)/√2": (0.0, -1.0, 0.0),
    "maximally mixed": (0.0, 0.0, 0.0),
}


def bloch_from_angles(theta: float, phi: float, radius: float = 1.0) -> np.ndarray:
    """The Bloch vector of polar angle θ (from +z), azimuth φ and length `radius` (< 1 is a mixed state)."""
    if not 0.0 <= radius <= 1.0 + 1e-12:
        raise ValueError("The Bloch vector's length must be between 0 and 1.")
    return radius * np.array([np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi), np.cos(theta)])


# ------------------------------------------------------------------------------------------ gates

@dataclass
class Gate:
    label: str
    unitary: np.ndarray
    axis: np.ndarray
    angle: float
    phase: float


def rotation_unitary(axis: Sequence[float], angle: float) -> np.ndarray:
    n = np.asarray(axis, dtype=float)
    n = n / np.linalg.norm(n)
    return np.cos(angle / 2) * IDENTITY - 1j * np.sin(angle / 2) * (n[0] * PAULI[0] + n[1] * PAULI[1] + n[2] * PAULI[2])


def axis_angle(u: np.ndarray) -> tuple[np.ndarray, float, float]:
    """(axis, angle, global phase) with U = e^{iα} exp(-i θ n·σ/2), 0 ≤ θ ≤ 2π. For the identity (up to
    phase) the axis is +z and the angle 0."""
    u = np.asarray(u, dtype=complex)
    det = np.linalg.det(u)
    alpha = float(np.angle(det) / 2)
    v = u * np.exp(-1j * alpha)
    c = float(np.real(np.trace(v)) / 2)
    w = np.array([np.real(0.5j * np.trace(s @ v)) for s in PAULI])
    s_mag = float(np.linalg.norm(w))
    theta = float(2 * np.arctan2(s_mag, c))
    axis = w / s_mag if s_mag > 1e-12 else np.array([0.0, 0.0, 1.0])
    return axis, theta, alpha


def _gate_from_unitary(label: str, u: np.ndarray) -> Gate:
    if not np.allclose(u.conj().T @ u, IDENTITY, atol=1e-10):
        raise ValueError(f"The gate {label} is not unitary.")
    axis, theta, alpha = axis_angle(u)
    return Gate(label, u, axis, theta, alpha)


_FIXED = {
    "I": IDENTITY, "X": PAULI[0], "Y": PAULI[1], "Z": PAULI[2],
    "H": (PAULI[0] + PAULI[2]) / np.sqrt(2),
    "S": np.diag([1, 1j]).astype(complex), "SDG": np.diag([1, -1j]).astype(complex),
    "T": np.diag([1, np.exp(1j * np.pi / 4)]).astype(complex), "TDG": np.diag([1, np.exp(-1j * np.pi / 4)]).astype(complex),
    "SX": 0.5 * np.array([[1 + 1j, 1 - 1j], [1 - 1j, 1 + 1j]]),
}
_PARAMETRIC = ("RX", "RY", "RZ", "P")
GATE_HELP = ("I, X, Y, Z, H, S, SDG, T, TDG, SX, and Rx(θ), Ry(θ), Rz(θ), P(φ) with θ in radians "
             "(write pi, e.g. Rx(pi/2))")


def split_sequence(text: str) -> list[str]:
    """Split "H, Rz(pi/4), X" at the commas that are not inside parentheses."""
    parts, depth, current = [], 0, ""
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch in ",;\n" and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    parts.append(current.strip())
    return [p for p in parts if p]


def parse_gate(token: str) -> Gate:
    m = re.fullmatch(r"\s*([A-Za-z]+)\s*(?:\((.*)\))?\s*", token)
    if not m:
        raise ExpressionError(f"Could not read the gate '{token}'. Gates: {GATE_HELP}.")
    name, arg = m.group(1).upper(), m.group(2)
    if name in _FIXED:
        if arg is not None:
            raise ExpressionError(f"{m.group(1)} takes no angle.")
        return _gate_from_unitary(m.group(1).upper() if len(m.group(1)) > 1 else m.group(1).upper(), _FIXED[name])
    if name in _PARAMETRIC:
        if arg is None:
            raise ExpressionError(f"{m.group(1)} needs an angle, e.g. {m.group(1)}(pi/2).")
        try:
            angle = float(safe_expression(arg, []).evalf())
        except (TypeError, ValueError):
            raise ExpressionError(f"The angle '{arg}' is not a number.") from None
        if not np.isfinite(angle):
            raise ExpressionError("The angle must be finite.")
        label = f"{m.group(1).capitalize() if name != 'P' else 'P'}({arg.strip()})"
        if name == "P":
            return _gate_from_unitary(label, np.diag([1, np.exp(1j * angle)]).astype(complex))
        axis = {"RX": (1, 0, 0), "RY": (0, 1, 0), "RZ": (0, 0, 1)}[name]
        return _gate_from_unitary(label, rotation_unitary(axis, angle))
    raise ExpressionError(f"Unknown gate '{m.group(1)}'. Gates: {GATE_HELP}.")


def parse_sequence(text: str) -> list[Gate]:
    tokens = split_sequence(text)
    if len(tokens) > MAX_GATES:
        raise ExpressionError(f"At most {MAX_GATES} gates.")
    return [parse_gate(t) for t in tokens]


# ------------------------------------------------------------------------------------------ noise and propagation

@dataclass(frozen=True)
class Noise:
    t1: float | None = None          # amplitude-damping time (None = no relaxation)
    t_phi: float | None = None       # pure-dephasing time
    depolarizing: float = 0.0        # rate γd

    def __post_init__(self):
        for name in ("t1", "t_phi"):
            v = getattr(self, name)
            if v is not None and not (np.isfinite(v) and v > 0):
                raise ValueError(f"{name} must be a positive number (or left off).")
        if not (np.isfinite(self.depolarizing) and self.depolarizing >= 0):
            raise ValueError("The depolarising rate must be zero or positive.")

    @property
    def gamma1(self) -> float:
        return 0.0 if self.t1 is None else 1.0 / self.t1

    @property
    def gamma_phi(self) -> float:
        return 0.0 if self.t_phi is None else 1.0 / self.t_phi

    @property
    def rate_xy(self) -> float:
        """1/T2: the decay rate of the transverse components."""
        return self.gamma1 / 2 + self.gamma_phi + self.depolarizing

    @property
    def rate_z(self) -> float:
        return self.gamma1 + self.depolarizing

    @property
    def t2(self) -> float | None:
        return None if self.rate_xy == 0 else 1.0 / self.rate_xy

    @property
    def noiseless(self) -> bool:
        return self.rate_xy == 0 and self.rate_z == 0


def _cross_matrix(n: np.ndarray) -> np.ndarray:
    return np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])


def bloch_generator(axis: np.ndarray, omega: float, noise: Noise) -> np.ndarray:
    """The 4x4 matrix A with d/dt [r, 1] = A [r, 1]: rotation about `axis` at angular speed `omega`, minus
    the decay, plus the drive toward +z from amplitude damping."""
    a = np.zeros((4, 4))
    a[:3, :3] = omega * _cross_matrix(axis) - np.diag([noise.rate_xy, noise.rate_xy, noise.rate_z])
    a[2, 3] = noise.gamma1
    return a


def propagate(r: np.ndarray, axis: np.ndarray, omega: float, duration: float, noise: Noise) -> np.ndarray:
    """The Bloch vector after `duration` of rotation at speed `omega` about `axis` with the given noise."""
    out = expm(bloch_generator(axis, omega, noise) * duration) @ np.append(r, 1.0)
    return out[:3]


def lindblad_superoperator(h: np.ndarray, noise: Noise) -> np.ndarray:
    """The 4x4 matrix of ρ -> -i[H,ρ] + dissipator, acting on column-stacked vec(ρ)."""
    i2 = IDENTITY

    def dissipator(l_op: np.ndarray) -> np.ndarray:
        ldl = l_op.conj().T @ l_op
        return (np.kron(l_op.conj(), l_op) - 0.5 * np.kron(i2, ldl) - 0.5 * np.kron(ldl.T, i2))

    sup = -1j * (np.kron(i2, h) - np.kron(h.T, i2))
    if noise.gamma1 > 0:
        sup = sup + dissipator(np.sqrt(noise.gamma1) * SIGMA_MINUS)
    if noise.gamma_phi > 0:
        sup = sup + dissipator(np.sqrt(noise.gamma_phi / 2) * PAULI[2])
    if noise.depolarizing > 0:
        for s in PAULI:
            sup = sup + dissipator(np.sqrt(noise.depolarizing / 4) * s)
    return sup


def propagate_lindblad(rho: np.ndarray, axis: np.ndarray, omega: float, duration: float, noise: Noise) -> np.ndarray:
    """The same evolution computed from the master equation on ρ itself (independent of the Bloch-vector
    derivation, hence the cross-check)."""
    h = 0.5 * omega * (axis[0] * PAULI[0] + axis[1] * PAULI[1] + axis[2] * PAULI[2])
    vec = rho.reshape(-1, order="F")
    out = expm(lindblad_superoperator(h, noise) * duration) @ vec
    return out.reshape(2, 2, order="F")


# ------------------------------------------------------------------------------------------ running a sequence

@dataclass
class QubitRun:
    times: np.ndarray
    bloch: np.ndarray                 # (n, 3): the noisy state
    ideal: np.ndarray                 # (n, 3): the same circuit without noise
    purity: np.ndarray
    fidelity: np.ndarray              # with the ideal state at the same time
    gate_labels: list[str]
    gate_edges: list[float]           # the time at which each gate starts, plus the end
    segment_of_sample: np.ndarray     # index of the gate (or idle) each sample belongs to
    initial: np.ndarray
    noise: Noise
    gate_time: float
    idle_time: float
    checks: CheckList
    gates: list[Gate] = field(default_factory=list)

    @property
    def final_density(self) -> np.ndarray:
        return bloch_to_density(self.bloch[-1])

    @property
    def final_ideal_density(self) -> np.ndarray:
        return bloch_to_density(self.ideal[-1])


def simulate_sequence(sequence: str | Sequence[Gate], initial: BlochLike, noise: Noise | None = None,
                      gate_time: float = 1.0, idle_time: float = 0.0, progress: ProgressFn | None = None) -> QubitRun:
    """Apply the gates in order, each taking `gate_time`, with `idle_time` of pure noise after each, and
    verify the run (see the module docstring)."""
    gates = parse_sequence(sequence) if isinstance(sequence, str) else list(sequence)
    if not gates:
        raise ValueError("Give at least one gate.")
    if not (np.isfinite(gate_time) and gate_time > 0):
        raise ValueError("The gate time must be positive.")
    if not (np.isfinite(idle_time) and idle_time >= 0):
        raise ValueError("The idle time must be zero or positive.")
    noise = noise or Noise()
    r0 = np.asarray(initial, dtype=float)
    if r0.shape != (3,) or np.linalg.norm(r0) > 1 + 1e-9:
        raise ValueError("The initial Bloch vector must have three components and length at most 1.")

    times, noisy, ideal, seg = [0.0], [r0.copy()], [r0.copy()], [0]
    edges = [0.0]
    labels: list[str] = []
    r, r_ideal, clock = r0.copy(), r0.copy(), 0.0
    cross_rho = bloch_to_density(r0)
    noiseless = Noise()
    max_rho_gap = 0.0
    segment = 0
    for g_index, gate in enumerate(gates):
        report(progress, f"Gate {g_index + 1} of {len(gates)}: {gate.label}", g_index / len(gates))
        labels.append(gate.label)
        omega = gate.angle / gate_time
        for k in range(1, SAMPLES_PER_GATE + 1):
            dt = gate_time * k / SAMPLES_PER_GATE
            noisy.append(propagate(r, gate.axis, omega, dt, noise))
            ideal.append(propagate(r_ideal, gate.axis, omega, dt, noiseless))
            times.append(clock + dt)
            seg.append(segment)
        r, r_ideal = noisy[-1].copy(), ideal[-1].copy()
        cross_rho = propagate_lindblad(cross_rho, gate.axis, omega, gate_time, noise)
        max_rho_gap = max(max_rho_gap, float(np.max(np.abs(cross_rho - bloch_to_density(r)))))
        clock += gate_time
        segment += 1
        if idle_time > 0:
            labels.append("idle")
            for k in range(1, 7):
                dt = idle_time * k / 6
                noisy.append(propagate(r, np.array([0, 0, 1.0]), 0.0, dt, noise))
                ideal.append(r_ideal.copy())
                times.append(clock + dt)
                seg.append(segment)
            r = noisy[-1].copy()
            cross_rho = propagate_lindblad(cross_rho, np.array([0, 0, 1.0]), 0.0, idle_time, noise)
            max_rho_gap = max(max_rho_gap, float(np.max(np.abs(cross_rho - bloch_to_density(r)))))
            clock += idle_time
            segment += 1
        edges.append(clock)
    report(progress, "Checking the run", 0.95)

    noisy_arr, ideal_arr = np.array(noisy), np.array(ideal)
    purity_arr = (1.0 + np.sum(noisy_arr ** 2, axis=1)) / 2.0
    fid = np.array([uhlmann_fidelity(a, b) for a, b in zip(noisy_arr, ideal_arr)])
    checks = _verify_run(gates, noisy_arr, ideal_arr, purity_arr, fid, noise, r0, max_rho_gap)
    return QubitRun(np.array(times), noisy_arr, ideal_arr, purity_arr, fid, labels, edges, np.array(seg), r0, noise,
                    gate_time, idle_time, checks, gates)


def _verify_run(gates, noisy, ideal, purity_arr, fid, noise: Noise, r0, rho_gap) -> CheckList:
    checks = CheckList()
    worst_unitary = max(float(np.max(np.abs(g.unitary.conj().T @ g.unitary - IDENTITY))) for g in gates)
    checks.add("Every gate is unitary", worst_unitary < 1e-10, f"largest |U†U - 1| = {worst_unitary:.1e}")
    rebuilt = max(float(np.max(np.abs(np.exp(1j * g.phase) * rotation_unitary(g.axis, g.angle) - g.unitary))) for g in gates)
    checks.add("Each gate equals its axis-angle rotation", rebuilt < 1e-9, f"largest rebuild error = {rebuilt:.1e}")

    length = np.linalg.norm(noisy, axis=1)
    checks.add("The state stays physical (|r| ≤ 1, ρ ⪰ 0)", float(np.max(length)) <= 1 + 1e-9,
               f"largest |r| = {np.max(length):.9f}")
    # (Not "never increases": amplitude damping is non-unital and can RAISE the purity of a mixed state by
    # pulling it toward |0>. What can never happen is a purity above 1.)
    checks.add("Purity never exceeds 1", bool(np.all(purity_arr <= 1 + 1e-9)), f"largest purity = {np.max(purity_arr):.9f}")
    checks.add("Bloch-vector and Lindblad propagation agree", rho_gap < 1e-9,
               f"largest entry difference between the two final states = {rho_gap:.1e}")
    if noise.noiseless:
        pure_start = abs(np.linalg.norm(r0) - 1) < 1e-9
        if pure_start:
            checks.add("No noise: the state stays pure", float(np.max(np.abs(length - 1))) < 1e-9,
                       f"largest ||r| - 1| = {np.max(np.abs(length - 1)):.1e}")
        checks.add("No noise: the path equals the ideal path", float(np.max(np.abs(noisy - ideal))) < 1e-9,
                   f"largest deviation = {np.max(np.abs(noisy - ideal)):.1e}")
    idle = idle_decay_check(r0, noise)
    if idle is not None:
        checks.add("Idle decay follows the closed forms for T1 and T2", idle[0], idle[1])
    return checks


def idle_decay_check(r0: np.ndarray, noise: Noise, times: Sequence[float] | None = None) -> tuple[bool, str] | None:
    """Compare propagated idle decay from `r0` with
        r_z(t) = 1 + (r_z0 - 1) e^{-t/T1}          (with depolarising: r_z0 e^{-γd t} + ... folded into rate_z),
        r_xy(t) = r_xy0 e^{-t/T2}.
    The z closed form with both amplitude damping and depolarising is r_z(t) = z_eq + (r_z0 - z_eq) e^{-rate_z t}
    with z_eq = γ1 / rate_z. Returns (agrees, detail), or None for a noiseless qubit (nothing decays)."""
    if noise.noiseless:
        return None
    t_scale = 1.0 / max(noise.rate_xy, noise.rate_z)
    ts = np.array(times if times is not None else np.linspace(0.1, 3.0, 8) * t_scale)
    z_eq = noise.gamma1 / noise.rate_z if noise.rate_z > 0 else r0[2]
    worst = 0.0
    for t in ts:
        got = propagate(r0, np.array([0, 0, 1.0]), 0.0, float(t), noise)
        expected = np.array([r0[0] * np.exp(-noise.rate_xy * t), r0[1] * np.exp(-noise.rate_xy * t),
                             z_eq + (r0[2] - z_eq) * np.exp(-noise.rate_z * t)])
        worst = max(worst, float(np.max(np.abs(got - expected))))
    return worst < 1e-10, f"largest difference from the closed form = {worst:.1e} (T1 = {noise.t1}, T2 = {noise.t2})"


def ramsey_curve(noise: Noise, times: np.ndarray, detuning: float = 0.0) -> np.ndarray:
    """The Ramsey fringe: ⟨σx⟩ after (π/2)_y, a free wait t at detuning δ, and the readout axis x, which is
    e^{-t/T2} cos(δ t) -- the standard way to measure T2. (Computed by propagating the Bloch vector.)"""
    start = propagate(np.array([0.0, 0.0, 1.0]), np.array([0.0, 1.0, 0.0]), np.pi / 2, 1.0, Noise())
    return np.array([propagate(start, np.array([0, 0, 1.0]), detuning, float(t), noise)[0] for t in times])


def t1_curve(noise: Noise, times: np.ndarray) -> np.ndarray:
    """r_z(t) after preparing |1> -- the standard T1 (inversion-recovery) measurement."""
    return np.array([propagate(np.array([0.0, 0.0, -1.0]), np.array([0, 0, 1.0]), 0.0, float(t), noise)[2] for t in times])


def fit_exponential_time(times: np.ndarray, values: np.ndarray, floor: float = 0.0) -> float | None:
    """The time constant τ of values ≈ floor + (v0 - floor) e^{-t/τ}, from a log-linear fit (None if it cannot be fit)."""
    y = np.asarray(values, dtype=float) - floor
    mask = np.abs(y) > 1e-9
    if mask.sum() < 3:
        return None
    slope = np.polyfit(np.asarray(times)[mask], np.log(np.abs(y[mask])), 1)[0]
    return float(-1.0 / slope) if slope < 0 else None


def sphere_trajectory_unitary(sequence: Sequence[Gate]) -> np.ndarray:
    """The product of the gates, as one unitary (first gate applied first)."""
    u = IDENTITY.copy()
    for gate in sequence:
        u = gate.unitary @ u
    return u


def sqrtm_fidelity(rho: np.ndarray, sigma: np.ndarray) -> float:
    """(Tr √(√ρ σ √ρ))² computed directly with matrix square roots -- the definition, used to test the closed form."""
    root = sqrtm(rho)
    return float(np.real(np.trace(sqrtm(root @ sigma @ root))) ** 2)
