"""
Circuits and control: RLC transients, Bode plots, step response and PID tuning -- Streamlit-free, no LLM.

Every result is checked against something the method did not use to produce it:
  * RLC TRANSIENTS are integrated numerically from the circuit equations (charge and current as states). The response is then compared
    with the closed form, with scipy's step response of the transfer function, and with quantities MEASURED from the simulated curve:
    the damping ratio from the logarithmic decrement of successive peaks, the ringing frequency from the peak spacing, the overshoot and
    the peak time. The energy balance (work done by the source = energy stored + energy dissipated in R) is evaluated from the same samples.
  * BODE PLOTS are evaluated from the symbolic transfer function SymPy derives from the impedances, compared with scipy's own frequency
    response, and the resonant frequency, peak gain and -3 dB bandwidth read off the numerical curve are compared with the formulas
    (omega_0 = 1/sqrt(LC), zeta = (R/2) sqrt(C/L), Q = 1/(2 zeta), bandwidth = omega_0 / Q).
  * CONTROL LOOPS (plant + PID with a filtered derivative, unity feedback) are analysed by transfer-function algebra and then
    cross-checked three ways: the Routh-Hurwitz criterion must give the same number of right-half-plane poles as the pole locations; the
    closed loop assembled as an explicit feedback simulation (plant and controller as separate state-space systems) must match the
    step response from the closed-loop transfer function; and the final value theorem must match the simulated final value. Gain and
    phase margins are read from the loop's frequency response and each crossover is verified by evaluating the loop there.
  * TUNING rules (Ziegler-Nichols from the ultimate gain or the reaction curve, SIMC, and a numerical ITAE optimisation) are only
    as good as their inputs: the ultimate gain is found twice (from the phase crossover and from the Routh table) and the closed loop at
    that gain must have poles on the imaginary axis at the predicted frequency.

LINEAR THEORY ONLY, STATED. Saturation, quantisation, sampling and noise are not modelled; a delay is a Pade approximation (the order is
shown). A good step response here is a statement about the linear model, not about the hardware.
"""
import math
from dataclasses import dataclass, field
from typing import Any
from functools import lru_cache

import numpy as np
import sympy as sp
from scipy import signal
from scipy.integrate import solve_ivp
from scipy.interpolate import pade
from scipy.optimize import brentq, minimize
from scipy.signal import find_peaks

from modules.quantum_common import CheckList


class CircuitError(ValueError):
    """Bad input, with a reason a student can act on."""


# ------------------------------------------------------------------------------------------ helpers

def _positive(name: str, value: float) -> float:
    if not np.isfinite(value) or value <= 0:
        raise CircuitError(f"{name} must be a positive number.")
    return float(value)


def parse_poly(text: str, name: str = "coefficients") -> np.ndarray:
    """Coefficients typed as numbers separated by spaces or commas, highest power first."""
    try:
        vals = [float(t) for t in text.replace(";", ",").replace(",", " ").split()]
    except ValueError:
        raise CircuitError(f"The {name} must be numbers separated by spaces or commas.") from None
    if not 1 <= len(vals) <= 9:
        raise CircuitError(f"Give between 1 and 9 {name} (highest power first).")
    arr = np.trim_zeros(np.array(vals), "f")
    if arr.size == 0:
        raise CircuitError(f"The {name} cannot all be zero.")
    return arr


def _refine(x: np.ndarray, y: np.ndarray, i: int, log_axis: bool = False) -> tuple[float, float]:
    """Extremum near index i from a parabola through three samples (on a log axis if asked): (position, value)."""
    if i <= 0 or i >= len(y) - 1:
        return float(x[i]), float(y[i])
    xs = np.log(x[i - 1:i + 2]) if log_axis else x[i - 1:i + 2]
    c = np.polyfit(xs, y[i - 1:i + 2], 2)
    if c[0] == 0:
        return float(x[i]), float(y[i])
    xp_ = -c[1] / (2 * c[0])
    return float(math.exp(xp_) if log_axis else xp_), float(np.polyval(c, xp_))


# ------------------------------------------------------------------------------------------ second-order circuits

@dataclass(frozen=True)
class RLC:
    R: float
    L: float
    C: float
    topology: str = "series"       # "series" (voltage step) or "parallel" (current step)

    def validate(self) -> None:
        _positive("R", self.R), _positive("L", self.L), _positive("C", self.C)
        if self.topology not in ("series", "parallel"):
            raise CircuitError("The topology is 'series' or 'parallel'.")

    @property
    def omega0(self) -> float:
        return 1.0 / math.sqrt(self.L * self.C)

    @property
    def f0(self) -> float:
        return self.omega0 / (2 * math.pi)

    @property
    def zeta(self) -> float:
        return 0.5 * self.R * math.sqrt(self.C / self.L) if self.topology == "series" else 0.5 / self.R * math.sqrt(self.L / self.C)

    @property
    def Q(self) -> float:
        return 1.0 / (2.0 * self.zeta)

    @property
    def omega_d(self) -> float | None:
        return self.omega0 * math.sqrt(1 - self.zeta ** 2) if self.zeta < 1 else None

    @property
    def regime(self) -> str:
        z = self.zeta
        return "underdamped" if z < 1 - 1e-9 else "critically damped" if abs(z - 1) <= 1e-9 else "overdamped"

    @property
    def poles(self) -> np.ndarray:
        return np.roots([1.0, 2 * self.zeta * self.omega0, self.omega0 ** 2])


@lru_cache(maxsize=2)
def _symbolic_transfer(topology: str):
    """Transfer functions derived from the impedances: {output: expression in s, R, L, C}."""
    s = sp.Symbol("s")
    R, L, C = sp.symbols("R L C", positive=True)
    if topology == "series":
        total = R + s * L + 1 / (s * C)
        out = {"v_C": (1 / (s * C)) / total, "v_R": R / total, "v_L": (s * L) / total}
    else:
        y = 1 / R + s * C + 1 / (s * L)                       # admittance of the parallel network, driven by a current source
        out = {"v": 1 / y}
    return s, (R, L, C), {k: sp.simplify(v) for k, v in out.items()}


def circuit_transfer_functions(c: RLC) -> dict[str, tuple[sp.Expr, np.ndarray, np.ndarray]]:
    """{output: (H(s) with the component values substituted, numerator, denominator)} from the symbolic derivation."""
    c.validate()
    s, (R, L, C), exprs = _symbolic_transfer(c.topology)
    out = {}
    for name, h in exprs.items():
        hn = sp.together(h.subs({R: sp.Float(c.R, 17), L: sp.Float(c.L, 17), C: sp.Float(c.C, 17)}))
        num, den = sp.fraction(hn)
        pn, pd = sp.Poly(sp.expand(num), s), sp.Poly(sp.expand(den), s)
        lead = float(pd.LC())
        out[name] = (h, np.array([float(x) / lead for x in pn.all_coeffs()]), np.array([float(x) / lead for x in pd.all_coeffs()]))
    return out


def symbolic_transfer_latex(topology: str) -> dict[str, str]:
    s, _, exprs = _symbolic_transfer(topology)
    return {k: sp.latex(v) for k, v in exprs.items()}


def step_closed_form(c: RLC, t: np.ndarray, level: float = 1.0) -> np.ndarray:
    """The primary output's response to a step of height `level` (V for the series circuit's capacitor voltage, A for the parallel circuit's voltage)."""
    z, w0 = c.zeta, c.omega0
    if c.topology == "series":
        if z < 1 - 1e-9:
            wd = w0 * math.sqrt(1 - z * z)
            return level * (1 - np.exp(-z * w0 * t) * (np.cos(wd * t) + z / math.sqrt(1 - z * z) * np.sin(wd * t)))
        if abs(z - 1) <= 1e-9:
            return level * (1 - (1 + w0 * t) * np.exp(-w0 * t))
        r = math.sqrt(z * z - 1)
        s1, s2 = w0 * (-z + r), w0 * (-z - r)
        return level * (1 + (s2 * np.exp(s1 * t) - s1 * np.exp(s2 * t)) / (s1 - s2))
    scale = level / c.C
    if z < 1 - 1e-9:
        wd = w0 * math.sqrt(1 - z * z)
        return scale * np.exp(-z * w0 * t) * np.sin(wd * t) / wd
    if abs(z - 1) <= 1e-9:
        return scale * t * np.exp(-w0 * t)
    r = math.sqrt(z * z - 1)
    s1, s2 = w0 * (-z + r), w0 * (-z - r)
    return scale * (np.exp(s1 * t) - np.exp(s2 * t)) / (s1 - s2)


def _peaks(t: np.ndarray, y: np.ndarray, floor_fraction: float = 1e-4) -> tuple[np.ndarray, np.ndarray]:
    """Positive local maxima of y with sub-sample position and height."""
    idx, _ = find_peaks(y, height=floor_fraction * max(float(np.max(np.abs(y))), 1e-300))
    pos = [_refine(t, y, int(i)) for i in idx]
    return np.array([p[0] for p in pos]), np.array([p[1] for p in pos])


@dataclass
class Transient:
    circuit: RLC
    level: float
    t: np.ndarray
    signals: dict[str, np.ndarray]
    primary: str
    closed_form: np.ndarray
    measured: dict[str, Any]
    theory: dict[str, Any]
    energy: dict[str, np.ndarray]
    checks: CheckList = field(default_factory=CheckList)
    notes: list[str] = field(default_factory=list)


def _auto_duration(c: RLC, periods_cap: float = 300.0) -> float:
    z, w0 = c.zeta, c.omega0
    if z < 1 - 1e-9:
        wd = w0 * math.sqrt(1 - z * z)
        return float(min(max(7.0 / (z * w0), 4 * 2 * math.pi / wd), periods_cap * 2 * math.pi / wd))
    slow = float(min(abs(p.real) for p in c.poles))
    return 9.0 / slow


def simulate_transient(c: RLC, level: float = 1.0, t_end: float | None = None, n: int = 20001) -> Transient:
    """A step into the circuit from rest (series: voltage step, parallel: current step), integrated from the circuit equations."""
    c.validate()
    if not np.isfinite(level) or level == 0:
        raise CircuitError("The step height must be a non-zero number.")
    t_end = float(t_end) if t_end else _auto_duration(c)
    if t_end <= 0:
        raise CircuitError("The duration must be positive.")
    t = np.linspace(0.0, t_end, int(n))
    R, L, C = c.R, c.L, c.C
    if c.topology == "series":
        # states: charge q, current i, and the running integrals of the source's work and of the heat in R (so the energy balance is integrated, not estimated)
        rhs = lambda _t, y: [y[1], (level - R * y[1] - y[0] / C) / L, level * y[1], R * y[1] ** 2]      # noqa: E731
        sol = solve_ivp(rhs, (0, t_end), [0.0, 0.0, 0.0, 0.0], method="DOP853", t_eval=t, rtol=1e-11, atol=[1e-14 * abs(level) * C, 1e-14 * abs(level) / R, 1e-30, 1e-30])
        q, i, work, dissipated = sol.y
        sig = {"v_C": q / C, "v_R": R * i, "v_L": level - R * i - q / C, "i": i}
        primary = "v_C"
        stored = 0.5 * C * sig["v_C"] ** 2 + 0.5 * L * i ** 2
    else:
        rhs = lambda _t, y: [(level - y[0] / R - y[1]) / C, y[0] / L, level * y[0], y[0] ** 2 / R]       # noqa: E731 -- states: v, iL, work, heat
        sol = solve_ivp(rhs, (0, t_end), [0.0, 0.0, 0.0, 0.0], method="DOP853", t_eval=t, rtol=1e-11, atol=[1e-14 * abs(level) * R, 1e-14 * abs(level), 1e-30, 1e-30])
        v, il, work, dissipated = sol.y
        sig = {"v": v, "i_L": il, "i_R": v / R, "i_C": level - v / R - il}
        primary = "v"
        stored = 0.5 * C * v ** 2 + 0.5 * L * il ** 2
    closed = step_closed_form(c, t, level)
    measured: dict[str, Any] = {}
    theory: dict[str, Any] = {}
    ck = CheckList()
    notes: list[str] = []
    y = sig[primary]
    scale = float(np.max(np.abs(closed))) or 1.0
    err = float(np.max(np.abs(y - closed))) / scale
    ck.add("The numerical solution matches the closed form", err < 1e-6, f"largest difference {err:.1e} of the response's size (the circuit equations integrated with an 8th-order solver)")
    num, den = circuit_transfer_functions(c)[primary][1:]
    _, ys = signal.step(signal.TransferFunction(num, den), T=t)
    ys = ys * level
    err2 = float(np.max(np.abs(y - ys))) / scale
    ck.add("scipy's step response of the transfer function agrees", err2 < 1e-4, f"largest difference {err2:.1e} (the transfer function was derived symbolically from the impedances)")
    z, w0, wd = c.zeta, c.omega0, c.omega_d
    final = level if c.topology == "series" else 0.0
    if wd is not None:
        px, ph = _peaks(t, (y - final) * (1 if level > 0 else -1)) if c.topology == "series" else _peaks(t, y * (1 if level > 0 else -1))
        theory["omega_d"] = wd
        if c.topology == "series":
            theory["overshoot"] = math.exp(-math.pi * z / math.sqrt(1 - z * z))
            theory["peak_time"] = math.pi / wd
            if len(ph):
                measured["overshoot"] = float(ph[0] / abs(level))
                measured["peak_time"] = float(px[0])
                rel = abs(measured["overshoot"] - theory["overshoot"]) / theory["overshoot"]
                ck.add("Overshoot = exp(-pi zeta / sqrt(1 - zeta^2))", rel < 0.005, f"measured {measured['overshoot']:.4%}, formula {theory['overshoot']:.4%} (difference {rel:.2%})")
                relp = abs(px[0] - theory["peak_time"]) / theory["peak_time"]
                ck.add("Peak time = pi / omega_d", relp < 0.005, f"measured {px[0] * 1e3:.5g} ms, formula {theory['peak_time'] * 1e3:.5g} ms (difference {relp:.2%})")
        if len(px) >= 3:
            gaps = np.diff(px)
            measured["omega_d"] = float(2 * math.pi / np.mean(gaps))
            ratio = ph[:-1] / ph[1:]
            delta = float(np.mean(np.log(ratio)))
            measured["zeta"] = delta / math.sqrt(4 * math.pi ** 2 + delta ** 2)
            theory["zeta"] = z
            ck.add("Damping ratio from the logarithmic decrement = (R/2) sqrt(C/L)" if c.topology == "series" else "Damping ratio from the logarithmic decrement = (1/2R) sqrt(L/C)",
                   abs(measured["zeta"] - z) / z < 0.01, f"from {len(px)} successive peaks: measured {measured['zeta']:.5f}, circuit values give {z:.5f}")
            ck.add("Ringing frequency from the peak spacing = omega_0 sqrt(1 - zeta^2)", abs(measured["omega_d"] - wd) / wd < 0.005,
                   f"measured {measured['omega_d']:.5g} rad/s ({measured['omega_d'] / 2 / math.pi:.5g} Hz), formula {wd:.5g} rad/s")
        else:
            notes.append("Fewer than three peaks fit in the record, so the damping ratio could not be measured from the decay: lower R (series) or raise it (parallel) to see ringing.")
        if c.topology == "series":
            band = np.abs(y - final) > 0.02 * abs(level)
            ts = float(t[np.nonzero(band)[0][-1]]) if band.any() else 0.0
            bound = -math.log(0.02 * math.sqrt(1 - z * z)) / (z * w0)
            measured["settling_time"], theory["settling_time"] = ts, bound
            ck.add("The 2 % settling time lies within the decay-envelope bound", ts <= bound * 1.001, f"measured {ts * 1e3:.4g} ms; the envelope e^(-zeta omega_0 t) / sqrt(1 - zeta^2) reaches 2 % at {bound * 1e3:.4g} ms (the textbook 4/(zeta omega_0) = {4 / (z * w0) * 1e3:.4g} ms)")
    else:
        slow = float(min(abs(p.real) for p in c.poles))
        theory["decay_rate"] = slow
        resid = np.abs(y - final)
        sel = (resid < 1e-3 * scale) & (resid > 1e-6 * scale) & (t > 0)
        if sel.sum() > 20:
            slope = float(np.polyfit(t[sel], np.log(resid[sel]), 1)[0])
            measured["decay_rate"] = -slope
            rel = abs(-slope - slow) / slow
            ck.add("The late-time decay follows the slower pole", rel < 0.02, f"fitted decay rate {-slope:.5g} /s, slow pole at {slow:.5g} /s (difference {rel:.2%})")
        if c.regime == "overdamped":
            notes.append("Overdamped: two real poles, no ringing. The response creeps up on its final value at the rate of the slower pole.")
    energy_err = float(np.max(np.abs(work - stored - dissipated))) / max(float(np.max(np.abs(work))), 1e-300)
    ck.add("Energy balance: work by the source = energy stored + energy dissipated", energy_err < 1e-7, f"largest imbalance {energy_err:.1e} of the work done (the work and the heat are integrated alongside the circuit equations)")
    return Transient(c, level, t, sig, primary, closed, measured, theory, {"work": work, "stored": stored, "dissipated": dissipated}, ck, notes)


# ------------------------------------------------------------------------------------------ frequency response (Bode)

@dataclass
class Bode:
    omega: np.ndarray
    response: dict[str, np.ndarray]            # complex H(j w) per output
    magnitude_db: dict[str, np.ndarray]
    phase_deg: dict[str, np.ndarray]
    measured: dict[str, dict[str, Any]]
    theory: dict[str, dict[str, Any]]
    checks: CheckList = field(default_factory=CheckList)


def circuit_bode(c: RLC, n: int = 4001, decades: float = 2.5) -> Bode:
    c.validate()
    w0 = c.omega0
    w = np.logspace(math.log10(w0) - decades, math.log10(w0) + decades, int(n))
    tfs = circuit_transfer_functions(c)
    s, (R, L, C), _ = _symbolic_transfer(c.topology)
    resp, mag, ph, measured, theory = {}, {}, {}, {}, {}
    ck = CheckList()
    z = c.zeta
    worst = 0.0
    for name, (h, num, den) in tfs.items():
        f = sp.lambdify(s, h.subs({R: c.R, L: c.L, C: c.C}), "numpy")
        hj = np.asarray(f(1j * w), dtype=complex)
        _, ref = signal.freqresp(signal.TransferFunction(num, den), w)
        worst = max(worst, float(np.max(np.abs(20 * np.log10(np.abs(hj)) - 20 * np.log10(np.abs(ref))))))
        resp[name] = hj
        mag[name] = 20 * np.log10(np.abs(hj))
        ph[name] = np.degrees(np.unwrap(np.angle(hj)))
    ck.add("The symbolic transfer function agrees with scipy's frequency response", worst < 1e-6, f"largest difference in magnitude over {len(w):,} frequencies: {worst:.1e} dB")
    for name in tfs:
        m = np.abs(resp[name])
        i = int(np.argmax(m))
        wp, gp = _refine(w, m, i, log_axis=True)
        meas: dict[str, Any] = {"peak_omega": wp, "peak_gain": gp, "bandwidth": None}
        th: dict[str, Any] = {"peak_omega": None, "peak_gain": None, "bandwidth": None}
        if name in ("v_R", "v"):
            th.update(peak_omega=w0, peak_gain=1.0 if name == "v_R" else c.R, bandwidth=2 * z * w0)
            half = gp / math.sqrt(2)
            above = np.nonzero(m >= half)[0]
            if len(above) and above[0] > 0 and above[-1] < len(w) - 1:
                lo = float(np.interp(half, [m[above[0] - 1], m[above[0]]], np.log([w[above[0] - 1], w[above[0]]])))
                hi = float(np.interp(half, [m[above[-1] + 1], m[above[-1]]], np.log([w[above[-1] + 1], w[above[-1]]])))
                meas["bandwidth"] = math.exp(hi) - math.exp(lo)
        elif z < 1 / math.sqrt(2):
            factor = math.sqrt(1 - 2 * z * z)
            th.update(peak_omega=w0 * factor if name == "v_C" else w0 / factor, peak_gain=1 / (2 * z * math.sqrt(1 - z * z)))
        measured[name], theory[name] = meas, th
        if th["peak_omega"] is not None:
            rel = abs(meas["peak_omega"] - th["peak_omega"]) / th["peak_omega"]
            label = {"v_R": "Resonance of the resistor voltage at omega_0 = 1/sqrt(LC)", "v": "Resonance at omega_0 = 1/sqrt(LC)",
                     "v_C": "Capacitor-voltage peak at omega_0 sqrt(1 - 2 zeta^2)", "v_L": "Inductor-voltage peak at omega_0 / sqrt(1 - 2 zeta^2)"}[name]
            ck.add(label, rel < 2e-3, f"measured {meas['peak_omega']:.5g} rad/s ({meas['peak_omega'] / 2 / math.pi:.5g} Hz), formula {th['peak_omega']:.5g} rad/s (difference {rel:.2%})")
            relg = abs(meas["peak_gain"] - th["peak_gain"]) / th["peak_gain"]
            ck.add(f"Peak gain of {name}", relg < 2e-3, f"measured {meas['peak_gain']:.5g}, formula {th['peak_gain']:.5g}")
        if th["bandwidth"] is not None and meas["bandwidth"] is not None:
            relb = abs(meas["bandwidth"] - th["bandwidth"]) / th["bandwidth"]
            ck.add("-3 dB bandwidth = omega_0 / Q = 2 zeta omega_0", relb < 0.01, f"measured {meas['bandwidth']:.5g} rad/s, formula {th['bandwidth']:.5g} rad/s; Q = {meas['peak_omega'] / meas['bandwidth']:.4g} (1/(2 zeta) = {c.Q:.4g})")
    return Bode(w, resp, mag, ph, measured, theory, ck)


# ------------------------------------------------------------------------------------------ polynomial and transfer-function tools

@dataclass(frozen=True)
class TF:
    num: np.ndarray
    den: np.ndarray

    def __call__(self, s):
        return np.polyval(self.num, s) / np.polyval(self.den, s)

    @property
    def poles(self) -> np.ndarray:
        return np.roots(self.den)

    @property
    def zeros(self) -> np.ndarray:
        return np.roots(self.num) if len(self.num) > 1 else np.array([])

    def series(self, other: "TF") -> "TF":
        return TF(np.polymul(self.num, other.num), np.polymul(self.den, other.den))

    def feedback(self) -> "TF":
        """Unity negative feedback: T = L / (1 + L)."""
        n = max(len(self.num), len(self.den))
        den = np.polyadd(self.den, self.num) if n else self.den
        return TF(self.num, np.trim_zeros(den, "f"))

    @property
    def dc_gain(self) -> float | None:
        d0 = self.den[-1]
        return None if abs(d0) < 1e-14 else float(self.num[-1] / d0)


def routh_rhp_count(coeffs: np.ndarray) -> tuple[int, int, np.ndarray]:
    """Routh-Hurwitz: (roots in the right half-plane, roots on the imaginary axis, the array) of the polynomial with `coeffs` (highest power first).
    A zero first element is replaced by a small epsilon; a row of zeros is replaced by the derivative of its auxiliary polynomial (the roots of which are the
    symmetric root pairs, counted as imaginary-axis roots when the first column does not change sign)."""
    c = np.array(np.trim_zeros(np.asarray(coeffs, dtype=float), "f"))
    n = len(c) - 1
    if n < 1:
        return 0, 0, np.array([c])
    cols = (n + 2) // 2
    table = np.zeros((n + 1, cols))
    table[0, :len(c[0::2])] = c[0::2]
    table[1, :len(c[1::2])] = c[1::2]
    eps = 1e-9 * max(1.0, float(np.max(np.abs(c))))
    imaginary = 0
    for i in range(2, n + 1):
        if not np.any(table[i - 1]):
            order = n - (i - 1) + 1                                    # order of the auxiliary polynomial from the row above
            aux = table[i - 2, :]
            powers = [order - 2 * k for k in range(cols)]
            table[i - 1, :] = [p * a if p > 0 else 0.0 for p, a in zip(powers, aux)]
            imaginary += max(order, 0)
        if abs(table[i - 1, 0]) < 1e-14:
            table[i - 1, 0] = eps
        pivot = table[i - 1, 0]
        for j in range(cols - 1):
            table[i, j] = (pivot * table[i - 2, j + 1] - table[i - 2, 0] * table[i - 1, j + 1]) / pivot
    first = table[:, 0]
    first = np.where(np.abs(first) < 1e-14, eps, first)
    changes = int(np.sum(np.sign(first[:-1]) * np.sign(first[1:]) < 0))
    return changes, imaginary, table


def margins(loop: TF, w_lo: float = 1e-4, w_hi: float = 1e4, n: int = 6000) -> dict[str, float | None]:
    """Gain and phase margins and crossover frequencies of the open loop, from its frequency response; each crossover is refined and verified."""
    w = np.logspace(math.log10(w_lo), math.log10(w_hi), n)
    h = loop(1j * w)
    mag = np.abs(h)
    phase = np.degrees(np.unwrap(np.angle(h)))
    out: dict[str, Any] = {"w_gc": None, "pm": None, "w_pc": None, "gm_db": None, "gc_error": None, "pc_error": None}
    s1 = np.sign(mag - 1.0)
    idx = np.nonzero(s1[:-1] * s1[1:] < 0)[0]
    if len(idx):
        i = int(idx[0])
        wg = brentq(lambda x: abs(loop(1j * x)) - 1.0, w[i], w[i + 1], xtol=1e-14, rtol=1e-12)
        ph = float(np.degrees(np.angle(loop(1j * wg))))
        # continue the unwrapped phase at the crossover
        base = float(np.interp(wg, w, phase))
        ph = base + (ph - base + 180) % 360 - 180
        out["w_gc"], out["pm"], out["gc_error"] = wg, 180.0 + ph, abs(abs(loop(1j * wg)) - 1.0)
    target = np.sign(phase + 180.0)
    jdx = np.nonzero(target[:-1] * target[1:] < 0)[0]
    if len(jdx):
        j = int(jdx[0])
        f = lambda x: float(np.interp(x, w, phase)) + 180.0                    # noqa: E731
        wp = brentq(f, w[j], w[j + 1], xtol=1e-14, rtol=1e-12)
        gain = abs(loop(1j * wp))
        out["w_pc"], out["gm_db"], out["pc_error"] = wp, -20 * math.log10(gain), abs(((float(np.degrees(np.angle(loop(1j * wp)))) + 360.0) % 360.0) - 180.0)
    return out


# ------------------------------------------------------------------------------------------ plants and PID

PLANT_KINDS = ("First order", "Second order", "First order with delay (FOPDT)", "Integrator with lag", "RLC capacitor voltage", "Custom transfer function")


@dataclass(frozen=True)
class Plant:
    name: str
    tf: TF
    params: dict
    delay: float = 0.0
    pade_order: int = 0

    def description(self) -> str:
        return self.name + (f" (delay {self.delay:g} s as a Padé approximation of order {self.pade_order})" if self.delay else "")


def pade_delay(theta: float, order: int = 3) -> TF:
    """The order-n Pade approximation of exp(-theta s)."""
    if theta <= 0:
        return TF(np.array([1.0]), np.array([1.0]))
    coeffs = [(-theta) ** k / math.factorial(k) for k in range(2 * order + 1)]
    p, q = pade(coeffs, order, order)
    return TF(np.array(p.coeffs, dtype=float), np.array(q.coeffs, dtype=float))


def make_plant(kind: str, **p) -> Plant:
    if kind == "First order":
        K, tau = float(p.get("K", 1.0)), _positive("The time constant", p.get("tau", 1.0))
        return Plant(kind, TF(np.array([K]), np.array([tau, 1.0])), dict(K=K, tau=tau))
    if kind == "Second order":
        K, wn, z = float(p.get("K", 1.0)), _positive("The natural frequency", p.get("wn", 1.0)), float(p.get("zeta", 0.5))
        if z < 0:
            raise CircuitError("The damping ratio cannot be negative.")
        return Plant(kind, TF(np.array([K * wn ** 2]), np.array([1.0, 2 * z * wn, wn ** 2])), dict(K=K, wn=wn, zeta=z))
    if kind == "First order with delay (FOPDT)":
        K, tau, theta = float(p.get("K", 1.0)), _positive("The time constant", p.get("tau", 1.0)), float(p.get("theta", 0.3))
        if theta < 0:
            raise CircuitError("The delay cannot be negative.")
        order = int(p.get("pade_order", 3))
        d = pade_delay(theta, order)
        base = TF(np.array([K]), np.array([tau, 1.0]))
        return Plant(kind, base.series(d), dict(K=K, tau=tau, theta=theta), theta, order if theta > 0 else 0)
    if kind == "Integrator with lag":
        K, tau = float(p.get("K", 1.0)), _positive("The time constant", p.get("tau", 1.0))
        return Plant(kind, TF(np.array([K]), np.array([tau, 1.0, 0.0])), dict(K=K, tau=tau))
    if kind == "RLC capacitor voltage":
        c = RLC(float(p["R"]), float(p["L"]), float(p["C"]), "series")
        c.validate()
        _, num, den = circuit_transfer_functions(c)["v_C"]
        return Plant(kind, TF(num, den), dict(R=c.R, L=c.L, C=c.C))
    if kind == "Custom transfer function":
        num, den = parse_poly(p["num"], "numerator coefficients"), parse_poly(p["den"], "denominator coefficients")
        if len(den) < len(num):
            raise CircuitError("The denominator must have at least as high a degree as the numerator (a proper transfer function).")
        return Plant(kind, TF(num, den), dict(num=num.tolist(), den=den.tolist()))
    raise CircuitError(f"Choose one of: {', '.join(PLANT_KINDS)}.")


def pid_tf(kp: float, ki: float, kd: float, tf: float) -> TF:
    """C(s) = Kp + Ki/s + Kd s / (Tf s + 1)  (the derivative filtered with time constant Tf)."""
    if min(kp, ki, kd) < 0:
        raise CircuitError("The PID gains must not be negative.")
    if kd > 0:
        _positive("The derivative filter time constant", tf)
        return TF(np.array([kp * tf + kd, kp + ki * tf, ki]), np.array([tf, 1.0, 0.0])) if ki > 0 else TF(np.array([kp * tf + kd, kp]), np.array([tf, 1.0]))
    if ki > 0:
        return TF(np.array([kp, ki]), np.array([1.0, 0.0]))
    return TF(np.array([kp]), np.array([1.0]))


# ------------------------------------------------------------------------------------------ step response, metrics, loop analysis

@dataclass
class StepMetrics:
    final: float
    rise_time: float | None
    overshoot_pct: float
    peak_time: float
    settling_time: float | None
    steady_state_error: float
    iae: float
    ise: float
    itae: float


def step_metrics(t: np.ndarray, y: np.ndarray, final: float | None = None) -> StepMetrics:
    fin = float(y[-1]) if final is None else float(final)
    ref = 1.0
    span = fin if abs(fin) > 1e-12 else 1.0
    lo, hi = 0.1 * span, 0.9 * span
    rise = None
    try:
        i10 = int(np.nonzero(y >= lo)[0][0]) if span > 0 else int(np.nonzero(y <= lo)[0][0])
        i90 = int(np.nonzero(y >= hi)[0][0]) if span > 0 else int(np.nonzero(y <= hi)[0][0])
        t10 = float(np.interp(lo, [y[i10 - 1], y[i10]], [t[i10 - 1], t[i10]])) if i10 > 0 else float(t[i10])
        t90 = float(np.interp(hi, [y[i90 - 1], y[i90]], [t[i90 - 1], t[i90]])) if i90 > 0 else float(t[i90])
        rise = t90 - t10
    except IndexError:
        rise = None
    k = int(np.argmax(y)) if span > 0 else int(np.argmin(y))
    overshoot = max(0.0, (float(y[k]) - fin) / abs(span) * 100.0)
    band = np.abs(y - fin) > 0.02 * abs(span)
    settle: float | None = float(t[np.nonzero(band)[0][-1]]) if band.any() else 0.0
    if band.any() and np.nonzero(band)[0][-1] >= len(t) - 2:
        settle = None
    e = ref - y
    return StepMetrics(fin, rise, overshoot, float(t[k]), settle, ref - fin, float(np.trapezoid(np.abs(e), t)), float(np.trapezoid(e * e, t)), float(np.trapezoid(t * np.abs(e), t)))


@dataclass
class LoopAnalysis:
    plant: Plant
    gains: tuple[float, float, float, float]
    controller: TF
    loop: TF
    closed: TF
    poles: np.ndarray
    stable: bool
    t: np.ndarray
    y: np.ndarray
    u: np.ndarray | None
    metrics: StepMetrics | None
    margins: dict[str, float | None]
    routh_rhp: int
    checks: CheckList = field(default_factory=CheckList)
    notes: list[str] = field(default_factory=list)


def _auto_time(poles: np.ndarray, stable: bool) -> float:
    if stable:
        sigma = float(min(-p.real for p in poles if -p.real > 1e-12)) if np.any(-poles.real > 1e-12) else 1.0
        return float(np.clip(8.0 / sigma, 1e-6, 1e6))
    growth = float(max(p.real for p in poles))
    return float(min(8.0 / max(growth, 1e-9), 1e6))


def closed_loop_explicit(plant: TF, controller: TF, t: np.ndarray) -> np.ndarray:
    """The unit-step response of the loop assembled as an explicit feedback simulation: plant and controller as separate state-space systems, e = r - y,
    u = C(e), y = P(u). Does not form the closed-loop transfer function at all."""
    ap, bp, cp, dp = signal.tf2ss(plant.num, plant.den)
    ac, bc, cc, dc = signal.tf2ss(controller.num, controller.den)
    npl, nc = ap.shape[0], ac.shape[0]
    dpp, dcc = float(np.squeeze(dp)), float(np.squeeze(dc))
    if abs(dpp) > 1e-14 and abs(dcc) > 1e-14:
        raise CircuitError("The loop has a direct feedthrough through both the plant and the controller (an algebraic loop).")
    cp, cc = np.atleast_2d(cp), np.atleast_2d(cc)

    def signals(xp_, xc):
        """(y, e, u) for the states, resolving the single feedthrough path."""
        if abs(dpp) > 1e-14:                                  # plant has direct feedthrough: u does not depend on e
            u = float((cc @ xc)[0])
            y = float((cp @ xp_)[0]) + dpp * u
            return y, 1.0 - y, u
        y = float((cp @ xp_)[0])
        e = 1.0 - y
        return y, e, float((cc @ xc)[0]) + dcc * e

    def rhs(_t, x):
        xp_, xc = x[:npl], x[npl:]
        _, e, u = signals(xp_, xc)
        return np.concatenate([ap @ xp_ + bp.ravel() * u, ac @ xc + bc.ravel() * e])

    sol = solve_ivp(rhs, (float(t[0]), float(t[-1])), np.zeros(npl + nc), method="LSODA", t_eval=t, rtol=1e-9, atol=1e-12)
    return np.array([signals(sol.y[:npl, k], sol.y[npl:, k])[0] for k in range(sol.y.shape[1])])


def analyse_loop(plant: Plant, kp: float, ki: float, kd: float, tf: float, t_end: float | None = None, n: int = 3000) -> LoopAnalysis:
    """Close the loop with a PID and report the step response, margins and the cross-checks."""
    ctrl = pid_tf(kp, ki, kd, tf)
    loop = plant.tf.series(ctrl)
    closed = loop.feedback()
    if len(closed.num) > len(closed.den):
        raise CircuitError("The closed loop is improper (more zeros than poles): add a derivative filter or a pole to the plant.")
    poles = closed.poles
    stable = bool(np.all(poles.real < -1e-9))
    t_end = float(t_end) if t_end else _auto_time(poles, stable)
    t = np.linspace(0.0, t_end, int(n))
    _, y = signal.step(signal.TransferFunction(closed.num, closed.den), T=t)
    ck = CheckList()
    notes: list[str] = []
    rhp_roots = int(np.sum(poles.real > 1e-9))
    rhp_routh, imag, _ = routh_rhp_count(closed.den)
    ck.add("Routh-Hurwitz and the pole locations agree on the number of unstable poles", rhp_routh == rhp_roots or (imag > 0 and abs(rhp_routh - rhp_roots) <= 0),
           f"the Routh table's first column has {rhp_routh} sign change(s); the closed-loop polynomial has {rhp_roots} root(s) with positive real part")
    try:
        y2 = closed_loop_explicit(plant.tf, ctrl, t)
        scale = max(float(np.max(np.abs(y))), 1.0)
        mask = np.abs(y) < 1e3 * scale if not stable else np.ones(len(t), bool)
        diff = float(np.max(np.abs(y[mask] - y2[mask]))) / scale
        ck.add("The closed loop built as an explicit feedback simulation matches the transfer-function algebra", diff < 2e-3,
               f"largest difference in the step response {diff:.1e} of its size (plant and PID integrated as separate systems with e = r - y)")
    except CircuitError as exc:
        notes.append(str(exc))
    metrics = None
    if stable:
        dc = closed.dc_gain
        if dc is not None:
            ck.add("The final value theorem agrees with the simulated final value", abs(dc - float(y[-1])) < 5e-3 * max(1.0, abs(dc)),
                   f"T(0) = {dc:.5g}; the simulation ends at {float(y[-1]):.5g} (t = {t_end:.3g} s)")
        metrics = step_metrics(t, y, dc)
        if ki == 0 and plant.tf.dc_gain is not None and plant.tf.den[-1] != 0:
            notes.append(f"Without integral action the loop settles at {metrics.final:.4g}, not 1: a steady-state error of {metrics.steady_state_error:.3g}. Add Ki to remove it.")
    else:
        notes.append(f"The closed loop is UNSTABLE: {rhp_roots} pole(s) in the right half-plane (largest real part {float(np.max(poles.real)):+.3g}). The response grows without bound.")
    m = margins(loop)
    if m["w_gc"] is not None:
        ck.add("The gain crossover was verified", (m["gc_error"] or 0) < 1e-9, f"|L(jω)| = 1 at ω = {m['w_gc']:.5g} rad/s (residual {m['gc_error']:.1e}); phase margin {m['pm']:.1f} degrees")
    if m["w_pc"] is not None:
        ck.add("The phase crossover was verified", (m["pc_error"] or 0) < 1e-3, f"phase = -180 degrees at ω = {m['w_pc']:.5g} rad/s (residual {m['pc_error']:.1e} degrees); gain margin {m['gm_db']:.1f} dB")
    return LoopAnalysis(plant, (kp, ki, kd, tf), ctrl, loop, closed, poles, stable, t, y, None, metrics, m, rhp_routh, ck, notes)


# ------------------------------------------------------------------------------------------ tuning

@dataclass
class Tuning:
    method: str
    kp: float
    ki: float
    kd: float
    tf: float
    detail: str
    checks: CheckList = field(default_factory=CheckList)


def ultimate_gain(plant: Plant) -> tuple[float, float, CheckList] | None:
    """(Ku, wu): the proportional gain at which the loop reaches the edge of stability, and the frequency of the oscillation there. Found from the phase
    crossover of the plant AND from the Routh table; at Ku the closed-loop poles must lie on the imaginary axis at wu. None when no gain destabilises the plant."""
    m = margins(plant.tf)
    if m["w_pc"] is None or m["gm_db"] is None:
        return None
    wu = float(m["w_pc"])
    ku_nyq = 10 ** (m["gm_db"] / 20)
    ck = CheckList()

    def rhp(k: float) -> int:
        den = np.polyadd(plant.tf.den, k * plant.tf.num)
        return routh_rhp_count(den)[0]

    lo, hi = ku_nyq * 0.5, ku_nyq * 2.0
    stable_lo, stable_hi = rhp(lo) == 0, rhp(hi) == 0
    if not stable_lo or stable_hi:
        ku_routh = float("nan")
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if rhp(mid) == 0 else (lo, mid)
        ku_routh = 0.5 * (lo + hi)
    ck.add("The ultimate gain from the phase crossover matches the one from the Routh table", abs(ku_routh - ku_nyq) / ku_nyq < 1e-3,
           f"Nyquist/phase crossover: {ku_nyq:.5g}; Routh bisection: {ku_routh:.5g}")
    roots = np.roots(np.polyadd(plant.tf.den, ku_nyq * plant.tf.num))
    nearest = roots[np.argmin(np.abs(roots.real))]
    ck.add("At the ultimate gain the closed loop oscillates at the predicted frequency", abs(nearest.real) < 1e-4 * wu and abs(abs(nearest.imag) - wu) / wu < 1e-4,
           f"closed-loop poles at {nearest.real:+.2e} ± {abs(nearest.imag):.5g}j; predicted ±{wu:.5g}j")
    return ku_nyq, wu, ck


def zn_closed_loop(ku: float, wu: float, kind: str = "PID") -> Tuning:
    tu = 2 * math.pi / wu
    if kind == "P":
        kp, ki, kd = 0.5 * ku, 0.0, 0.0
    elif kind == "PI":
        kp = 0.45 * ku
        ki, kd = kp / (tu / 1.2), 0.0
    else:
        kp = 0.6 * ku
        ki, kd = kp / (tu / 2), kp * tu / 8
    return Tuning(f"Ziegler-Nichols (ultimate gain), {kind}", kp, ki, kd, (tu / 8) / 10 if kd > 0 else 0.0, f"Ku = {ku:.4g}, Tu = {tu:.4g} s")


def fopdt_from_step(plant: Plant) -> tuple[float, float, float] | None:
    """(K, tau, theta) of the best first-order-plus-dead-time fit by the two-point (28.3 % and 63.2 %) method; None for a plant without a settling step response."""
    poles = plant.tf.poles
    if np.any(poles.real >= -1e-9) or plant.tf.dc_gain is None:
        return None
    K = plant.tf.dc_gain
    t_end = _auto_time(poles, True)
    t = np.linspace(0, t_end, 6000)
    _, y = signal.step(signal.TransferFunction(plant.tf.num, plant.tf.den), T=t)
    if K == 0 or abs(y[-1] - K) > 0.02 * abs(K):
        return None
    yn = y / K
    if float(np.min(yn)) < -0.2 or float(np.max(yn)) > 1.1:        # a Pade delay dips a little at first; anything bigger is not a first-order-like curve
        return None
    t283, t632 = float(np.interp(0.283, yn, t)), float(np.interp(0.632, yn, t))
    tau = 1.5 * (t632 - t283)
    theta = max(t632 - tau, 0.0)
    return float(K), float(tau), float(theta)


def zn_open_loop(K: float, tau: float, theta: float, kind: str = "PID") -> Tuning:
    if theta <= 1e-9:
        raise CircuitError("The reaction-curve rule needs a dead time greater than zero (use SIMC or the ultimate-gain rule instead).")
    r = tau / (K * theta)
    if kind == "P":
        kp, ki, kd = r, 0.0, 0.0
    elif kind == "PI":
        kp = 0.9 * r
        ki, kd = kp / (theta / 0.3), 0.0
    else:
        kp = 1.2 * r
        ki, kd = kp / (2 * theta), kp * 0.5 * theta
    return Tuning(f"Ziegler-Nichols (reaction curve), {kind}", kp, ki, kd, 0.5 * theta / 10 if kd > 0 else 0.0, f"K = {K:.4g}, τ = {tau:.4g} s, θ = {theta:.4g} s")


def simc_pi(K: float, tau: float, theta: float, tau_c: float | None = None) -> Tuning:
    """Skogestad's SIMC tuning of a PI controller for a first-order-plus-dead-time process (tau_c defaults to theta)."""
    tc = max(theta, 0.05 * tau) if tau_c is None else float(tau_c)
    kp = tau / (K * (tc + theta))
    ti = min(tau, 4 * (tc + theta))
    return Tuning("SIMC (Skogestad), PI", kp, kp / ti, 0.0, 0.0, f"K = {K:.4g}, τ = {tau:.4g} s, θ = {theta:.4g} s, τc = {tc:.4g} s")


def optimise_itae(plant: Plant, start: tuple[float, float, float], tf: float, t_end: float | None = None, max_evals: int = 160) -> Tuning:
    """Minimise the integral of time-weighted absolute error of the step response over (Kp, Ki, Kd) by Nelder-Mead from `start`; unstable designs are rejected."""
    def cost(logp: np.ndarray) -> float:
        kp, ki, kd = np.exp(logp)
        try:
            ctrl = pid_tf(kp, ki, kd if kd > 1e-9 else 0.0, tf)
            closed = plant.tf.series(ctrl).feedback()
            if len(closed.num) > len(closed.den):
                return 1e9
            poles = closed.poles
            if np.any(poles.real > -1e-9):
                return 1e9
            te = t_end or _auto_time(poles, True)
            t = np.linspace(0, te, 800)
            _, y = signal.step(signal.TransferFunction(closed.num, closed.den), T=t)
            return float(np.trapezoid(t * np.abs(1 - y), t))
        except (ValueError, np.linalg.LinAlgError):
            return 1e9

    x0 = np.log(np.maximum(np.array(start, dtype=float), 1e-6))
    res = minimize(cost, x0, method="Nelder-Mead", options=dict(maxfev=max_evals, xatol=1e-3, fatol=1e-6))
    kp, ki, kd = np.exp(res.x)
    if res.fun >= 1e9:
        raise CircuitError("No stable design was found from that starting point; try the Ziegler-Nichols or SIMC rule first.")
    return Tuning("Numerical ITAE optimisation, PID", float(kp), float(ki), float(kd) if kd > 1e-6 else 0.0, tf, f"minimised the integral of t·|e| (final cost {res.fun:.4g}, {res.nfev} evaluations)")


def root_locus(plant: Plant, controller_gain_shape: TF | None = None, gains: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Closed-loop poles as a proportional gain k runs from small to large: the roots of den + k num. Returns (gains, roots[len(gains), n])."""
    loop = plant.tf if controller_gain_shape is None else plant.tf.series(controller_gain_shape)
    ks = np.logspace(-3, 3, 400) if gains is None else np.asarray(gains, dtype=float)
    n = max(len(loop.den), len(loop.num)) - 1
    rows = []
    for k in ks:
        r = np.roots(np.polyadd(loop.den, k * loop.num))
        rows.append(np.pad(r, (0, n - len(r)), constant_values=np.nan))
    return ks, np.array(rows)
