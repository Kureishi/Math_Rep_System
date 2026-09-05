"""
Goal-seek / inverse solve: chains.py's sweep_step_binding() already
answers "if I vary this input, what happens to the output" -- sweep a
range and read the resulting curve. This answers the inverse, and much
more commonly-asked, question directly: "what value of this input makes
the output hit a SPECIFIC target number", without sweeping a range and
eyeballing a chart to find where it crosses.

Works by substituting every OTHER known value in as usual, but then
ALSO substituting the DESIRED value in place of the target's own
symbol -- turning "solve the model for target, given these inputs" into
"solve the model for one input, given the desired target" -- and
solving THAT system for the seek variable. Tries a symbolic sp.solve()
first (exact, and often literally the original formula algebraically
rearranged, which is worth showing to a student as its own small
derivation); if the system doesn't yield to that (implicit,
transcendental, etc.), falls back to numerical_fallback.py's
mpmath.findroot machinery on the residual -- the same fallback
verifier.py itself uses whenever sp.solve can't invert an equation
symbolically, applied here to the inverted system instead of the
original one.
"""
from dataclasses import dataclass

import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.numerical_fallback import find_numerical_roots
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_DOMAIN_PREDICATES = {
    "positive": lambda v: v > 0,
    "nonnegative": lambda v: v >= 0,
    "negative": lambda v: v < 0,
    "nonpositive": lambda v: v <= 0,
}


def _extract_solve_values(sol, seek_sym: sp.Symbol) -> list:
    """Normalizes sp.solve()'s several possible return shapes -- a flat
    list (single-equation form), a dict, or a list of dicts/tuples
    (multi-equation "system" form, see the call site) -- into a plain
    list of candidate expressions for `seek_sym`."""
    if isinstance(sol, dict):
        val = sol.get(seek_sym)
        return [val] if val is not None else []
    if isinstance(sol, (list, tuple)):
        values = []
        for item in sol:
            if isinstance(item, dict):
                v = item.get(seek_sym)
                if v is not None:
                    values.append(v)
            elif isinstance(item, (tuple, list)):
                if item:
                    values.append(item[0])
            else:
                values.append(item)
        return values
    return []


@dataclass
class GoalSeekResult:
    seek_symbol: str
    target: str
    target_value: float
    solutions: list[float]     # every distinct real value found -- a quadratic goal, e.g.,
                                 # commonly has two; a declared domain on seek_symbol (see
                                 # equation_engine.Variable.domain) narrows this when it doesn't
                                 # empty the list entirely
    is_numerical: bool          # True if solutions came from the numerical fallback rather than
                                 # an exact symbolic inversion
    formula: str | None = None  # LaTeX of the symbolic inverse, when sp.solve found one


def goal_seek(model: ProblemModel, target: str, target_value: float, seek_symbol: str) -> GoalSeekResult:
    """Finds the value(s) of `seek_symbol` that make `target` equal
    `target_value`, holding every other known input fixed at its
    current value."""
    if target not in model.solve_for or target_kind(model, target) != "equation":
        raise ValueError(f"'{target}' isn't an algebraic solve_for target of this model.")
    var_by_symbol = {v.symbol: v for v in model.variables}
    if seek_symbol not in var_by_symbol:
        raise ValueError(f"'{seek_symbol}' isn't a variable in this model.")
    if seek_symbol == target:
        raise ValueError("The seek variable can't be the target itself.")

    # fix every OTHER known value, and pin the target's own symbol to the
    # DESIRED value -- this substitution is what turns "solve for target"
    # into "solve for seek_symbol"
    fixed_subs = {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables
        if v.known_value is not None and v.symbol != seek_symbol
    }
    fixed_subs[sp.Symbol(target)] = sp.nsimplify(target_value)

    seek_sym = sp.Symbol(seek_symbol)
    eqs = [e.sympy_eq.subs(fixed_subs) for e in model.equations
           if e.kind == "equation" and e.sympy_eq is not None]
    eqs = [eq for eq in eqs if seek_sym in eq.free_symbols]
    if not eqs:
        raise ValueError(f"No equation relates '{target}' to '{seek_symbol}' once the other "
                          "known values are fixed.")

    # sp.solve()'s return shape depends on how it's called: a single
    # Eq + single symbol reliably returns a flat list of values, while
    # passing even a length-1 LIST of equations switches it into
    # "system" mode and returns a dict (one solution) or a list of
    # tuples (several) instead -- using the single-equation form
    # whenever possible sidesteps that inconsistency for the
    # overwhelmingly common case of exactly one governing equation.
    try:
        if len(eqs) == 1:
            sol = run_with_timeout(sp.solve, eqs[0], seek_sym, label="goal seek")
        else:
            sol = run_with_timeout(sp.solve, eqs, seek_sym, label="goal seek")
    except ComputationTimeoutError:
        sol = []
    except Exception:  # noqa: BLE001
        sol = []

    real_solutions: list[float] = []
    formula = None
    for s in _extract_solve_values(sol, seek_sym):
        try:
            c = complex(s)
        except (TypeError, ValueError):
            continue
        if abs(c.imag) > 1e-9:
            continue
        real_solutions.append(c.real)
    if real_solutions:
        formula = sp.latex(_extract_solve_values(sol, seek_sym)[0])

    is_numerical = False
    if not real_solutions and len(eqs) == 1:
        residual = eqs[0].lhs - eqs[0].rhs
        if residual.free_symbols == {seek_sym}:
            roots = find_numerical_roots(residual, seek_sym)
            real_solutions = [r.value for r in roots]
            is_numerical = bool(real_solutions)

    if not real_solutions:
        raise ValueError(f"Couldn't find a value of '{seek_symbol}' that makes "
                          f"'{target}' = {target_value:g} -- try a different target value, or "
                          "check that the equation actually depends on this input.")

    # a declared domain (equation_engine.Variable.domain) narrows results
    # when a solution set has more than one root -- but never down to
    # zero: an over-eager filter silently hiding every candidate would
    # be worse than showing an out-of-domain root with the domain intact
    # for the caller to flag, so keep the unfiltered set if filtering
    # would empty it
    domain = var_by_symbol[seek_symbol].domain
    predicate = _DOMAIN_PREDICATES.get(domain)
    if predicate is not None:
        filtered = [v for v in real_solutions if predicate(v)]
        if filtered:
            real_solutions = filtered

    distinct = sorted({round(v, 10) for v in real_solutions})
    return GoalSeekResult(seek_symbol=seek_symbol, target=target, target_value=target_value,
                            solutions=distinct, is_numerical=is_numerical, formula=formula)
