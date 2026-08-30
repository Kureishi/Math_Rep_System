"""
"Are these two expressions the same" as a standalone utility -- explicitly
NOT a new representation capability the way matrix/vector support was; this
is a building block other code (or a person manually, via the app) can
call to compare two symbolic expressions.

Built on sp.Expr.equals(), which already does the right two-tier check
internally: try to simplify the difference to zero symbolically, and if
that's inconclusive, evaluate both sides at randomly-chosen points and
compare numerically. Verified against real SymPy behavior before writing
this docstring (not assumed): equals() correctly returns True for
sin(x)**2+cos(x)**2 vs 1 and (x+1)**2 vs its expansion, and False for
log(x*y) vs log(x)+log(y) over the reals (true only for positive x,y --
domain-dependent, so *not* a universal equivalence). For a
domain-conditional case like sqrt(x**2) vs x (equal for x>=0, not for
x<0), equals() is observed to return False on some runs and None on
others, because its internal check samples random points -- it's a
legitimate False either way (the two are NOT universally equal), just
not deterministic about which run notices via a random negative sample
versus giving up outright.

For the None case, this module goes one step further than equals() by
itself and reports which of a handful of numeric sample points actually
agreed -- e.g. "agrees for x > 0, disagrees for x < 0" -- since a bare
"undetermined" throws away information a person could use to see WHY.
"""
from dataclasses import dataclass, field
import random

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

_SAMPLE_POINTS = (-3, -1.5, -0.5, 0.5, 1.5, 3, 5, 7)


@dataclass
class EquivalenceResult:
    equivalent: bool | None      # True, False, or None (genuinely undetermined either way)
    method: str                  # "symbolic" | "numeric sampling" | "undetermined"
    difference_simplified: sp.Expr | None
    detail: str
    error: str | None = None
    raw_difference: sp.Expr | None = None  # e1 - e2 BEFORE simplification -- proof.py starts
                                             # here rather than from difference_simplified, which
                                             # by this point has already been fully reduced and so
                                             # has nothing left to show step-by-step


def _parse(expr_str: str, extra_symbols: list[str] | None = None) -> sp.Expr:
    local_dict = {s: sp.Symbol(s) for s in (extra_symbols or [])}
    return parse_expr(expr_str, local_dict=local_dict, transformations=_TRANSFORMS)


def _sample_agreement(e1: sp.Expr, e2: sp.Expr, symbols: list[sp.Symbol]) -> tuple[int, int, sp.Point | None]:
    """Evaluates e1 - e2 numerically at combinations of _SAMPLE_POINTS for
    each free symbol (skipping any combination that raises -- singularities,
    complex results, etc.). Returns (agree_count, tried_count, an example
    disagreeing assignment as a dict-like list of (symbol, value) or None)."""
    diff_func = sp.lambdify(symbols, e1 - e2, modules=["math"])
    agree, tried, example_disagreement = 0, 0, None
    rng = random.Random(0)  # deterministic across runs, for reproducible reports
    combos = []
    if len(symbols) == 1:
        combos = [(p,) for p in _SAMPLE_POINTS]
    else:
        for _ in range(12):
            combos.append(tuple(rng.choice(_SAMPLE_POINTS) for _ in symbols))
    for combo in combos:
        try:
            val = diff_func(*combo)
            if isinstance(val, complex):
                continue
            tried += 1
            if abs(val) < 1e-6:
                agree += 1
            elif example_disagreement is None:
                example_disagreement = list(zip([s.name for s in symbols], combo))
        except (ZeroDivisionError, ValueError, OverflowError, TypeError):
            continue
    return agree, tried, example_disagreement


def check_equivalence_exprs(e1: sp.Expr, e2: sp.Expr) -> EquivalenceResult:
    """Core equivalence check on already-parsed SymPy expressions --
    factored out of check_equivalence() so other modules (e.g.
    grading.py, comparing a student's algebra against the verified
    derivation) can reuse the exact same tested logic without a
    string-parse-reparse round trip through check_equivalence()'s
    string-based API."""
    raw_diff = e1 - e2
    try:
        diff = run_with_timeout(sp.simplify, raw_diff, label="equivalence simplify")
    except ComputationTimeoutError as e:
        return EquivalenceResult(None, "undetermined", raw_diff,
                                   f"Timed out while simplifying the difference: {e}",
                                   raw_difference=raw_diff)

    try:
        verdict = run_with_timeout(lambda: e1.equals(e2), label="equivalence .equals() check")
    except ComputationTimeoutError as e:
        return EquivalenceResult(None, "undetermined", diff,
                                   f"Timed out while checking equivalence: {e}",
                                   raw_difference=raw_diff)
    except Exception:  # noqa: BLE001
        verdict = None

    if verdict is True:
        return EquivalenceResult(
            True, "symbolic", diff,
            "Equivalent -- the difference simplifies to zero (or SymPy's internal check confirms "
            "equality) for every value of the free symbols.",
            raw_difference=e1 - e2,
        )
    if verdict is False:
        return EquivalenceResult(
            False, "symbolic", diff,
            "Not equivalent -- confirmed to differ (symbolically or at tested numeric points). If "
            "this is surprising, check whether the two forms are only equal under an extra domain "
            "restriction (e.g. log(x*y) = log(x) + log(y) only holds for positive x, y).",
        )

    # verdict is None: genuinely undecided by SymPy's own check. Do our
    # own numeric sampling as a supplementary, clearly-labeled signal.
    symbols = sorted((e1.free_symbols | e2.free_symbols), key=lambda s: s.name)
    if not symbols:
        return EquivalenceResult(
            None, "undetermined", diff,
            "Both expressions are constant but SymPy couldn't confirm equality or inequality "
            "numerically -- inspect the simplified difference directly.",
        )

    agree, tried, example = _sample_agreement(e1, e2, symbols)
    if tried == 0:
        return EquivalenceResult(
            None, "undetermined", diff,
            "Couldn't evaluate either expression at any sample point (every attempt hit a "
            "domain error) -- try supplying explicit assumptions (e.g. positive symbols) instead.",
        )
    if agree == tried:
        return EquivalenceResult(
            None, "numeric sampling",
            diff,
            f"Not proven equivalent, but agrees at all {tried} tested numeric point(s) -- likely "
            "equivalent under some domain restriction SymPy couldn't confirm symbolically (this is "
            "evidence, not proof).",
        )
    if agree == 0:
        return EquivalenceResult(
            False, "numeric sampling", diff,
            f"Disagrees at all {tried} tested numeric point(s) -- not equivalent.",
        )
    where = ", ".join(f"{name}={val}" for name, val in example) if example else "some tested points"
    return EquivalenceResult(
        None, "numeric sampling", diff,
        f"Agrees at {agree}/{tried} tested numeric points but disagrees at others (e.g. {where}) -- "
        "equivalent only over part of the domain, not universally.",
    )


def check_equivalence(expr1_str: str, expr2_str: str,
                       extra_symbols: list[str] | None = None) -> EquivalenceResult:
    """Parses both expressions (same permissive syntax as the rest of the
    app: implicit multiplication, ^ as power) and checks equivalence."""
    try:
        e1 = _parse(expr1_str, extra_symbols)
    except Exception as e:  # noqa: BLE001
        return EquivalenceResult(None, "undetermined", None, "", error=f"Couldn't parse the first expression: {e}")
    try:
        e2 = _parse(expr2_str, extra_symbols)
    except Exception as e:  # noqa: BLE001
        return EquivalenceResult(None, "undetermined", None, "", error=f"Couldn't parse the second expression: {e}")

    return check_equivalence_exprs(e1, e2)
