"""
Interval arithmetic: a genuinely different flavor of "how wrong could
this be" than monte_carlo.py's statistical sampling or
error_propagation.py's first-order approximation -- given each uncertain
input as a guaranteed range [lo, hi] (not a probability distribution),
computes a GUARANTEED range for the target: not "95% of samples fall in
this band," but "the target CANNOT be outside this band, given the
input ranges are correct." Fits this project's verification-first
instinct: a proven bound rather than a confidence interval.

Implemented as a small, self-contained Interval type with the standard
interval-arithmetic rules for +, -, *, / and a handful of elementary
functions, rather than pulling in a new dependency -- Python's operator
overloading means sp.lambdify()'s generated expression (ordinary +, -,
*, /, ** operators) works unmodified against Interval operands, so the
SAME "solve symbolically once, evaluate the closed form" approach
monte_carlo.py and error_propagation.py both use applies here too; only
the actual arithmetic underneath is different.

Deliberately conservative: multiplication/division always consider all
four corner combinations rather than assuming both operands are
positive (a shortcut that's wrong whenever an interval spans zero), and
even-integer powers and non-integer powers require extra care (an
even power of an interval spanning zero has its minimum AT zero, not at
either endpoint; a non-integer power of an interval reaching below zero
has no real result and is deliberately rejected rather than silently
producing a wrong answer).
"""
from dataclasses import dataclass

import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError


class Interval:
    """A closed real interval [lo, hi]. Supports +, -, *, /, ** and the
    handful of elementary functions registered in INTERVAL_FUNCTIONS
    below, all via the standard (conservative) interval-arithmetic
    rules -- every operation returns a range GUARANTEED to contain every
    possible result, never a probabilistic estimate."""

    __slots__ = ("lo", "hi")

    def __init__(self, lo: float, hi: float):
        if lo > hi:
            lo, hi = hi, lo
        self.lo = float(lo)
        self.hi = float(hi)

    def __repr__(self):
        return f"Interval({self.lo:g}, {self.hi:g})"

    @staticmethod
    def _coerce(other):
        if isinstance(other, Interval):
            return other
        return Interval(other, other)

    def __add__(self, other):
        o = self._coerce(other)
        return Interval(self.lo + o.lo, self.hi + o.hi)

    __radd__ = __add__

    def __neg__(self):
        return Interval(-self.hi, -self.lo)

    def __sub__(self, other):
        o = self._coerce(other)
        return Interval(self.lo - o.hi, self.hi - o.lo)

    def __rsub__(self, other):
        return self._coerce(other).__sub__(self)

    def __mul__(self, other):
        o = self._coerce(other)
        corners = (self.lo * o.lo, self.lo * o.hi, self.hi * o.lo, self.hi * o.hi)
        return Interval(min(corners), max(corners))

    __rmul__ = __mul__

    def __truediv__(self, other):
        o = self._coerce(other)
        if o.lo <= 0 <= o.hi:
            raise ZeroDivisionError("Interval division by a range that includes zero is undefined.")
        return self * Interval(1.0 / o.hi, 1.0 / o.lo)

    def __rtruediv__(self, other):
        return self._coerce(other).__truediv__(self)

    def __pow__(self, n):
        if not isinstance(n, (int, float)) or isinstance(n, Interval):
            raise TypeError("Interval exponents aren't supported -- only a fixed numeric power.")
        if float(n).is_integer() and n >= 0:
            n = int(n)
            if n == 0:
                return Interval(1.0, 1.0)
            if n % 2 == 1:  # odd power: monotonic, endpoints map straight across
                return Interval(self.lo ** n, self.hi ** n)
            # even power: minimum is 0 if the interval spans zero, since
            # the closest-to-zero point dominates, not either endpoint
            if self.lo >= 0:
                return Interval(self.lo ** n, self.hi ** n)
            if self.hi <= 0:
                return Interval(self.hi ** n, self.lo ** n)
            return Interval(0.0, max(self.lo ** n, self.hi ** n))
        # non-integer or negative power: only well-defined (as a real
        # result) when the whole interval is non-negative
        if self.lo < 0:
            raise ValueError("A non-integer or negative power of an interval reaching below "
                              "zero has no real result.")
        return Interval(self.lo ** n, self.hi ** n)

    def width(self) -> float:
        return self.hi - self.lo

    def midpoint(self) -> float:
        return (self.lo + self.hi) / 2


def _interval_sqrt(x):
    x = Interval._coerce(x)
    if x.lo < 0:
        raise ValueError("sqrt of an interval reaching below zero has no real result.")
    return Interval(x.lo ** 0.5, x.hi ** 0.5)


def _interval_exp(x):
    x = Interval._coerce(x)
    import math
    return Interval(math.exp(x.lo), math.exp(x.hi))  # exp is monotonic increasing


def _interval_log(x):
    x = Interval._coerce(x)
    if x.lo <= 0:
        raise ValueError("log of an interval reaching to zero or below has no real result.")
    import math
    return Interval(math.log(x.lo), math.log(x.hi))  # monotonic increasing


# passed to sp.lambdify's `modules` argument so that sympy function
# calls (sqrt(...), exp(...), log(...)) in a lambdified expression
# resolve to these interval-aware versions instead of numpy/math's --
# ordinary + - * / ** need no such mapping since Interval's own operator
# overloads handle those directly
INTERVAL_FUNCTIONS = {"sqrt": _interval_sqrt, "exp": _interval_exp, "log": _interval_log}


@dataclass
class IntervalResult:
    target: str
    lo: float
    hi: float
    midpoint: float
    width: float
    formula_latex: str | None = None


def propagate_interval(model: ProblemModel, target: str,
                        ranges: dict[str, tuple[float, float]]) -> IntervalResult:
    """Given each uncertain input's GUARANTEED range (not a
    distribution), computes a guaranteed range for `target`. `ranges`
    maps variable symbol -> (lo, hi)."""
    if target not in model.solve_for or target_kind(model, target) != "equation":
        raise ValueError(f"'{target}' isn't an algebraic solve_for target of this model.")
    if not ranges:
        raise ValueError("Need at least one variable's range to propagate.")
    for sym, (lo, hi) in ranges.items():
        if lo > hi:
            raise ValueError(f"'{sym}' has an inverted range ({lo}, {hi}) -- lo must be <= hi.")

    interval_symbols = set(ranges)
    fixed_subs = {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables
        if v.known_value is not None and v.symbol not in interval_symbols
    }
    eqs = [e.sympy_eq.subs(fixed_subs) for e in model.equations
           if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        raise ValueError("This model has no algebraic equations to solve.")

    algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    target_syms = [sp.Symbol(t) for t in algebraic_targets]
    try:
        sol = run_with_timeout(sp.solve, eqs, target_syms, dict=True, label="interval solve")
    except ComputationTimeoutError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't symbolically solve for '{target}': {e}") from e
    if not sol:
        raise ValueError(f"Couldn't symbolically solve for '{target}' given the fixed inputs.")

    target_expr = sol[0].get(sp.Symbol(target))
    if target_expr is None:
        raise ValueError(f"'{target}' didn't appear in the symbolic solution.")

    used_symbols = [s for s in target_expr.free_symbols if s.name in ranges]
    if not used_symbols:
        # the target doesn't depend on any of the given ranges -- deterministic
        value = float(target_expr)
        return IntervalResult(target=target, lo=value, hi=value, midpoint=value, width=0.0,
                                formula_latex=sp.latex(target_expr))

    try:
        f = sp.lambdify(used_symbols, target_expr, modules=[INTERVAL_FUNCTIONS])
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't build an interval-evaluable form of the target: {e}") from e

    args = [Interval(*ranges[s.name]) for s in used_symbols]
    try:
        result = f(*args)
    except (ZeroDivisionError, ValueError, TypeError) as e:
        raise ValueError(f"Couldn't evaluate the target over these ranges: {e}") from e
    if not isinstance(result, Interval):
        # the expression turned out to not actually depend on any interval
        # (e.g. simplified away) -- treat as a degenerate, zero-width interval
        result = Interval(float(result), float(result))

    return IntervalResult(target=target, lo=result.lo, hi=result.hi, midpoint=result.midpoint(),
                            width=result.width(), formula_latex=sp.latex(target_expr))
