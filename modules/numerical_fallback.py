"""
Numerical fallback: when sp.solve() can't find a closed form -- common
for equations that mix polynomial and transcendental terms (e.g.
x + sin(x) = 5, or x*exp(x) = 10, both perfectly ordinary physics/
engineering "solve for x" problems with no symbolic solution; sp.solve
raises NotImplementedError on exactly this shape) -- this falls back to
numerical root-finding via mpmath.findroot rather than showing nothing.
mpmath is already a SymPy dependency (SymPy uses it internally for
numeric evaluation), so this adds no new dependency.

Deliberately, unmistakably labeled as an APPROXIMATION, not an exact
symbolic answer: every NumericalRootResult carries is_numerical=True
(always True -- explicit on the object, not implicit via type), and
callers are responsible for surfacing that distinction in the UI/export
rather than letting a numerically-approximated root blend in silently
next to exact symbolic ones. This app's whole premise is verification-
first; a numerical fallback that looked exactly like a verified
symbolic answer would undermine that.

Scoped to a SINGLE equation in a SINGLE remaining unknown (after known
values are substituted in) -- coupled numerical solving across multiple
equations/unknowns simultaneously is a substantially harder and less
predictable problem (starting points interact with each other), and
isn't attempted here.
"""
from dataclasses import dataclass

import mpmath
import sympy as sp

from modules.timeout_utils import run_with_timeout

# Starting points tried, in order, for findroot -- covers the ranges
# word problems typically land in (small numbers, negative numbers, or
# comfortably large ones) without an unbounded/expensive search.
_START_POINTS = [1.0, -1.0, 0.1, 10.0, -10.0, 100.0, 0.01, 1000.0]


@dataclass
class NumericalRootResult:
    value: float
    residual: float          # |f(value)| at the found root -- a sanity figure to display
    start_point: float       # which starting guess actually converged to this root
    is_numerical: bool = True  # always True -- explicit flag, not left implicit


def find_numerical_roots(expr: sp.Expr, symbol: sp.Symbol, max_roots: int = 3,
                          tol: float = 1e-8) -> list[NumericalRootResult]:
    """Finds up to max_roots distinct real roots of expr == 0 in
    `symbol`, trying each of _START_POINTS as a starting guess for
    mpmath.findroot and keeping results that (a) actually converged to
    something with a near-zero residual and (b) aren't within 1e-6 of a
    root already found from a different starting point. Returns []
    (never raises) if nothing converges from any starting point -- "no
    numerical root found either" is a legitimate, expected outcome for
    a genuinely unsolvable or misspecified equation, not an error."""
    try:
        f = sp.lambdify(symbol, expr, "mpmath")
    except Exception:  # noqa: BLE001
        return []

    results: list[NumericalRootResult] = []
    for start in _START_POINTS:
        if len(results) >= max_roots:
            break
        try:
            root = run_with_timeout(mpmath.findroot, f, start, label="numerical root-finding")
            root_val = float(root)
            residual = abs(float(f(root)))
        except Exception:  # noqa: BLE001 -- includes ComputationTimeoutError; this loop tries
            continue        # several starting points, one failing shouldn't abort the others
        if residual > tol:
            continue  # didn't actually converge to a genuine root
        if any(abs(root_val - r.value) < 1e-6 for r in results):
            continue  # duplicate of a root already found from a different start
        results.append(NumericalRootResult(value=root_val, residual=residual, start_point=start))
    return results


def numerical_fallback_for_equation(equation: sp.Eq, target: sp.Symbol,
                                     max_roots: int = 3) -> list[NumericalRootResult]:
    """Convenience wrapper: finds numerical roots of equation.lhs -
    equation.rhs == 0 in `target`, but only when `target` is the ONLY
    free symbol left in the equation (everything else already
    substituted with known values) -- see the module docstring for why
    multi-unknown numerical solving isn't attempted here. Returns []
    if target isn't the sole free symbol."""
    expr = equation.lhs - equation.rhs
    if expr.free_symbols != {target}:
        return []
    return find_numerical_roots(expr, target, max_roots=max_roots)
