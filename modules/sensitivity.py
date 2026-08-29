"""
Sensitivity / what-if analysis: reuses uncertainty.py's approach of
re-solving the un-substituted system symbolically (solve_symbolic_for_target)
to get the target as a formula in terms of the knowns, then sweeps ONE
input across a range -- holding every other known fixed -- to see how
the answer moves.

Distinct from uncertainty.py's error propagation: that asks "how much
does the answer move given the STATED MEASUREMENT UNCERTAINTY on each
input"; this asks "if I could deliberately change this input, how much
does the answer move" -- the classic tornado-chart "which input matters
most" question from spreadsheet-style sensitivity analysis, independent
of whether any input actually carries a stated uncertainty at all.
"""
from dataclasses import dataclass, field
import numpy as np
import sympy as sp

from modules.equation_engine import ProblemModel
from modules.uncertainty import solve_symbolic_for_target


@dataclass
class SweepResult:
    symbol: str
    values: list[float]           # swept input values
    target_values: list[float]    # corresponding target values
    nominal_input: float
    nominal_target: float | None  # None if the target doesn't fully evaluate at nominal knowns


@dataclass
class TornadoEntry:
    symbol: str
    low_value: float
    high_value: float
    low_target: float
    high_target: float
    swing: float  # |high_target - low_target| -- how much the answer moves across this input's range


def _range_for(nominal: float, pct_range: float) -> tuple[float, float]:
    """+/- pct_range around nominal, or a fixed +/-1 window if nominal is
    zero (a percentage range around zero is meaningless)."""
    if nominal == 0:
        return -1.0, 1.0
    return nominal * (1 - pct_range), nominal * (1 + pct_range)


def sweep_input(model: ProblemModel, target_name: str, sweep_symbol_name: str,
                 knowns: dict, pct_range: float = 0.2, n_points: int = 21) -> SweepResult | None:
    """Sweeps sweep_symbol_name from nominal*(1-pct_range) to
    nominal*(1+pct_range) (n_points evenly spaced values), holding every
    other known fixed at its nominal value, evaluating the target's
    formula at each point. Returns None if the target has no closed
    symbolic formula, doesn't actually depend on sweep_symbol_name, or
    sweep_symbol_name isn't a known value to sweep in the first place."""
    expr = solve_symbolic_for_target(model, target_name)
    if expr is None:
        return None
    sweep_sym = sp.Symbol(sweep_symbol_name)
    if sweep_sym not in expr.free_symbols or sweep_sym not in knowns:
        return None

    nominal = float(knowns[sweep_sym])
    lo, hi = _range_for(nominal, pct_range)
    xs = list(np.linspace(lo, hi, n_points))

    other_knowns = {k: v for k, v in knowns.items() if k != sweep_sym}
    fixed_expr = expr.subs(other_knowns)
    try:
        f = sp.lambdify(sweep_sym, fixed_expr, "numpy")
        ys_arr = np.asarray(f(np.array(xs)), dtype=float).flatten()
        ys = list(ys_arr) if len(ys_arr) == len(xs) else [float(f(x)) for x in xs]
    except Exception:  # noqa: BLE001
        try:
            ys = [float(fixed_expr.subs(sweep_sym, x)) for x in xs]
        except Exception:  # noqa: BLE001
            return None

    nominal_full = expr.subs(knowns)
    nominal_target = float(nominal_full) if not nominal_full.free_symbols else None
    return SweepResult(symbol=sweep_symbol_name, values=xs, target_values=ys,
                         nominal_input=nominal, nominal_target=nominal_target)


def tornado_analysis(model: ProblemModel, target_name: str, knowns: dict,
                       pct_range: float = 0.2) -> list[TornadoEntry]:
    """For every known the target's formula actually depends on,
    computes the target's value at that input's low/high swept extremes
    (every other input held at nominal), ranked by swing magnitude --
    largest first, the classic tornado-chart ordering. Skips any input
    whose swept value can't be evaluated numerically (e.g. it would put
    another part of the formula out of its domain) rather than raising."""
    expr = solve_symbolic_for_target(model, target_name)
    if expr is None:
        return []

    entries: list[TornadoEntry] = []
    for sym in sorted(expr.free_symbols, key=lambda s: s.name):
        if sym not in knowns:
            continue
        nominal = float(knowns[sym])
        lo, hi = _range_for(nominal, pct_range)
        try:
            low_target = float(expr.subs({**knowns, sym: lo}))
            high_target = float(expr.subs({**knowns, sym: hi}))
        except (TypeError, ValueError):
            continue
        entries.append(TornadoEntry(
            symbol=sym.name, low_value=lo, high_value=hi,
            low_target=low_target, high_target=high_target,
            swing=abs(high_target - low_target),
        ))
    entries.sort(key=lambda e: e.swing, reverse=True)
    return entries
