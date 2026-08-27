"""
Domain-of-validity tracking: finds the conditions under which a derived
formula is actually defined -- denominators that must not be zero, even
roots requiring a nonnegative argument, logarithms requiring a positive
argument, inverse sine/cosine requiring an argument in [-1, 1] -- and
checks those conditions against the SPECIFIC known values in the current
problem, so an actively-violated restriction (e.g. dividing by t when
t happens to be 0) shows up as a genuine verification failure instead of
a silent NaN or a SymPy exception three steps later. Restrictions that
aren't currently violated are still surfaced as informational notes
("this formula is only valid for t != 0"), since knowing the boundary of
a formula's validity is useful even when today's inputs don't cross it.

Deliberately restricted to the four sources of domain restriction that
can be found structurally by walking the expression tree once, without
solving the whole expression's domain from scratch the way
sp.calculus.util.continuous_domain does (and that function only handles
one free variable at a time, whereas most of this app's formulas have
several). Trig functions like tan() aren't included: they have
infinitely many isolated singularities, and reporting "x != pi/2 + n*pi
for every integer n" as a blanket restriction on every trig-flavored
word problem would be more noise than signal.
"""
from dataclasses import dataclass
import sympy as sp


@dataclass
class DomainRestriction:
    description: str       # human-readable, e.g. "t != 0 (division)"
    condition: sp.Basic    # symbolic relational/boolean, e.g. sp.Ne(t, 0)
    kind: str              # "nonzero" | "nonneg" | "positive" | "range"


def find_domain_restrictions(expr: sp.Expr | None) -> list[DomainRestriction]:
    """Walks the expression tree once, collecting one restriction per
    division, even root, logarithm, and inverse-sine/cosine subexpression
    found. Deduplicated by the resulting condition (the same denominator
    or root often reappears more than once in an expanded expression)."""
    if expr is None:
        return []
    found: dict[str, DomainRestriction] = {}
    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow):
            base, exponent = node.args
            if base.is_number or not exponent.is_number:
                continue
            if exponent.is_negative:
                cond = sp.Ne(base, 0)
                found[str(cond)] = DomainRestriction(f"{base} \u2260 0 (division)", cond, "nonzero")
            elif exponent.is_Rational and exponent.q % 2 == 0:
                cond = sp.Ge(base, 0)
                found[str(cond)] = DomainRestriction(f"{base} \u2265 0 (even root)", cond, "nonneg")
        elif isinstance(node, sp.log) and len(node.args) == 1:
            arg = node.args[0]
            if not arg.is_number:
                cond = sp.Gt(arg, 0)
                found[str(cond)] = DomainRestriction(f"{arg} > 0 (logarithm)", cond, "positive")
        elif isinstance(node, (sp.asin, sp.acos)):
            arg = node.args[0]
            if not arg.is_number:
                cond = sp.And(arg >= -1, arg <= 1)
                found[str(cond)] = DomainRestriction(f"-1 \u2264 {arg} \u2264 1 (inverse trig)", cond, "range")
    return list(found.values())


def domain_restrictions_for_equation(sympy_eq: sp.Eq) -> list[DomainRestriction]:
    """Combines restrictions found on both sides of an equation
    (dedup'd), since either side might contain a division/root/log."""
    if sympy_eq is None:
        return []
    seen: dict[str, DomainRestriction] = {}
    for r in find_domain_restrictions(sympy_eq.lhs) + find_domain_restrictions(sympy_eq.rhs):
        seen[str(r.condition)] = r
    return list(seen.values())


def evaluate_restriction(restriction: DomainRestriction, point: dict) -> bool | None:
    """True/False if `point` (a {Symbol: value} dict) supplies every
    symbol the restriction's condition needs; None if it can't yet be
    checked (some symbol involved is itself still unknown)."""
    free = restriction.condition.free_symbols
    if not free.issubset(point.keys()):
        return None
    try:
        return bool(restriction.condition.subs(point))
    except TypeError:
        return None
