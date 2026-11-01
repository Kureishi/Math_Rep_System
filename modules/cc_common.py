"""
Shared pieces of the cross-cutting numerics tools (modules/cc_*.py) -- Streamlit-free.

  * CrossCuttingError   -- the one exception type for unusable input (messages say what to change);
  * DriftReport         -- how a quantity that SHOULD be constant actually behaves along a numerical run, and above all WHETHER
                           THE DRIFT IS SECULAR (grows steadily: the integrator is leaking) OR BOUNDED (oscillates around the
                           true value: the usual signature of a symplectic method). The distinction matters more than the size.
  * parse helpers       -- expressions typed by the user go through the same guarded parser as the GR and statistical tools.
"""
from dataclasses import dataclass

import numpy as np
import sympy as sp

from modules.gr_metrics import _names_in, parse_in_symbols
from modules.quantum_common import CheckList


class CrossCuttingError(ValueError):
    """The input is not usable (the message says why, in terms the user can act on)."""


def parse_with(text: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    try:
        return parse_in_symbols(text, symbols)
    except ValueError as exc:
        raise CrossCuttingError(str(exc)) from None


def names_in(text: str) -> set[str]:
    return _names_in(text)


@dataclass
class DriftReport:
    name: str
    times: np.ndarray
    values: np.ndarray
    reference: float
    max_abs: float
    max_rel: float
    kind: str                 # "constant" | "bounded oscillation" | "secular drift"
    secular_change: float     # segment-mean change over the run, in units of the quantity (0 unless secular)
    noise: float              # typical oscillation amplitude within a segment

    @property
    def secular(self) -> bool:
        return self.kind == "secular drift"

    def summary(self) -> str:
        return (f"max |Δ| = {self.max_abs:.2e} (relative {self.max_rel:.1e}); {self.kind}"
                + (f", net change {self.secular_change:+.2e}" if self.secular else ""))


def drift_report(name: str, times: np.ndarray, values: np.ndarray, reference: float | None = None,
                 constant_tol: float = 1e-13, scale: float | None = None) -> DriftReport:
    """Classify the behaviour of `values(times)`, which should equal `reference` (default: the first value).

    Rounding-level variation is "constant". Otherwise the run is cut into four equal segments: if the segment MEANS move
    monotonically by more than three times the oscillation inside a segment (after removing each segment's own trend) the
    drift is "secular"; anything else is a "bounded oscillation". Fewer than ~1.25 oscillation periods in the whole run cannot
    be told apart from a slow drift, which is why runs for this purpose should cover many periods."""
    t = np.asarray(times, dtype=float)
    v = np.asarray(values, dtype=float)
    if len(t) != len(v) or len(v) < 8:
        raise CrossCuttingError("Need at least 8 samples of the monitored quantity.")
    ref = float(v[0]) if reference is None else float(reference)
    dev = v - ref
    scale = float(scale) if scale is not None else max(abs(ref), float(np.max(np.abs(v))), 1e-300)     # `scale`: for a quantity whose value is 0, say 1
    max_abs = float(np.max(np.abs(dev)))
    max_rel = max_abs / scale
    if max_abs <= constant_tol * scale:
        return DriftReport(name, t, v, ref, max_abs, max_rel, "constant", 0.0, max_abs)
    segs = np.array_split(np.arange(len(v)), 4)
    means = np.array([dev[s].mean() for s in segs])
    spreads = []
    for s in segs:
        x = t[s] - t[s].mean()
        d = dev[s]
        p = np.polyfit(x, d, 1) if np.ptp(x) > 0 else np.array([0.0, d.mean()])
        spreads.append(float(np.std(d - np.polyval(p, x))))
    noise = float(np.median(spreads))
    change = float(means[-1] - means[0])
    monotone = bool(np.all(np.diff(means) > 0) or np.all(np.diff(means) < 0))
    secular = monotone and abs(change) > 3.0 * max(noise, constant_tol * scale)
    return DriftReport(name, t, v, ref, max_abs, max_rel, "secular drift" if secular else "bounded oscillation",
                       change if secular else 0.0, noise)


def reports_to_checks(reports: list[DriftReport], tol_rel: float = 1e-6, allow_secular: bool = False) -> CheckList:
    """One check per monitored quantity: relative drift below `tol_rel` (and, unless allowed, not secular)."""
    ck = CheckList()
    for r in reports:
        ok = r.max_rel <= tol_rel and (allow_secular or not r.secular)
        ck.add(f"{r.name} conserved", ok, r.summary() + f" (tolerance {tol_rel:g})")
    return ck
