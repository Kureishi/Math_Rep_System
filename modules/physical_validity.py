"""
Physical-validity filtering: when an algebraic solve produces more than
one root (a quadratic time-of-flight equation is the classic case), the
raw sp.solve() result has no notion of which root is "the" physically
meaningful answer -- it just returns branches in whatever internal order
SymPy's solver happens to produce, and solver.py used to always take
branch [0] unconditionally. That's not a hypothetical bug: for
"-4.9*t**2 + 20*t + 1.5 = 0" (projectile height vs. time),
sp.solve(..., dict=True) returns the NEGATIVE time first --
t = -0.074s, then t = 4.16s. Silently reporting solutions[0] would have
handed back a nonsensical negative time as "the" answer.

This module filters candidate solutions against each unknown's declared
domain (see Variable.domain: "nonnegative"/"positive"/"nonpositive"/
"negative") and explains, for each discarded root, exactly which
constraint it violated -- rather than silently keeping OR silently
discarding a root with no explanation either way. Only symbols with an
explicitly declared domain are checked; nothing is filtered based on a
guessed convention (e.g. assuming all times are nonnegative without the
extraction step having said so).
"""
from dataclasses import dataclass, field
import sympy as sp

from modules.equation_engine import ProblemModel

DOMAIN_CONDITIONS = {
    "nonnegative": lambda val: val >= 0,
    "positive": lambda val: val > 0,
    "nonpositive": lambda val: val <= 0,
    "negative": lambda val: val < 0,
}

DOMAIN_DESCRIPTIONS = {
    "nonnegative": "\u2265 0",
    "positive": "> 0",
    "nonpositive": "\u2264 0",
    "negative": "< 0",
}


@dataclass
class FilterResult:
    valid: list[dict] = field(default_factory=list)              # solutions satisfying every declared domain
    discarded: list[tuple[dict, list[str]]] = field(default_factory=list)  # (solution, reasons)
    checked_any_domain: bool = False   # False if no relevant symbol had a domain declared at all


def _domain_for_symbol(model: ProblemModel, symbol_name: str) -> str | None:
    v = next((v for v in model.variables if v.symbol == symbol_name), None)
    return v.domain if v else None


def filter_physically_valid(model: ProblemModel, solutions: list[dict],
                              relevant_symbols: list[sp.Symbol]) -> FilterResult:
    """Filters `solutions` (as returned by sp.solve(eqs, targets, dict=True))
    down to the ones consistent with each relevant symbol's declared
    domain. A solution is only discarded if it has a NUMERIC value for a
    symbol that has a DECLARED domain and violates it -- symbolic/
    non-numeric values, and symbols with no declared domain, are never
    grounds for discarding a solution (nothing to check them against)."""
    if not solutions:
        return FilterResult()

    checked_any = any(_domain_for_symbol(model, s.name) in DOMAIN_CONDITIONS for s in relevant_symbols)
    if not checked_any:
        return FilterResult(valid=list(solutions), checked_any_domain=False)

    valid, discarded = [], []
    for sol in solutions:
        reasons = []
        for sym in relevant_symbols:
            domain = _domain_for_symbol(model, sym.name)
            if domain not in DOMAIN_CONDITIONS:
                continue
            val = sol.get(sym)
            if val is None or not getattr(val, "is_number", False):
                continue
            try:
                numeric = float(val)
            except (TypeError, ValueError):
                continue
            if not DOMAIN_CONDITIONS[domain](numeric):
                reasons.append(f"{sym.name} = {numeric:g} violates its declared domain "
                                f"({sym.name} {DOMAIN_DESCRIPTIONS[domain]})")
        if reasons:
            discarded.append((sol, reasons))
        else:
            valid.append(sol)
    return FilterResult(valid=valid, discarded=discarded, checked_any_domain=True)
