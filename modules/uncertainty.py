"""
Standard first-order (linearized) uncertainty propagation, the same
technique taught in intro physics/engineering error analysis:

    sigma_f^2 = sum_i (df/dxi)^2 * sigma_xi^2

for a derived quantity f(x1, x2, ...) where each input xi carries an
independent measurement uncertainty sigma_xi. Valid when the
uncertainties are small relative to the values (so the linear/Taylor
approximation holds) and inputs don't covary -- the standard assumptions
for this kind of analysis, and the ones every intro-level "propagation
of error" formula makes.

Deliberately scoped to algebraic ("equation"-kind) targets only, the
same scope matrix_utils.py uses for its coupled-system detection: the
symbolic pre-substitution formula this needs (see
solve_symbolic_for_target) is straightforward to recover for a plain
algebraic solve, but ODE/recurrence/optimization solutions don't have an
equally clean closed form to differentiate in general.
"""
from dataclasses import dataclass, field
import sympy as sp

from modules.equation_engine import ProblemModel, target_kind


@dataclass
class UncertaintyResult:
    nominal: float
    uncertainty: float                    # combined (propagated) absolute uncertainty
    relative_uncertainty: float | None    # uncertainty / |nominal|, None if nominal == 0
    contributions: dict[str, float]       # variable name -> its share of the propagated uncertainty
    dominant_source: str | None           # the variable name contributing the most, if any


def solve_symbolic_for_target(model: ProblemModel, target_name: str) -> sp.Expr | None:
    """Solves the UN-substituted algebraic system for target_name, as a
    formula in terms of whatever other symbols remain -- which, once a
    caller plugs in known values, are exactly the "known" quantities
    that might carry a measurement uncertainty. This deliberately does
    NOT reuse solver.py's step-generation path, which substitutes known
    numeric values into the equations before calling sp.solve() (so it
    can show a "substitute known values" step) -- by that point there's
    no symbolic trace left in terms of the named knowns to differentiate
    against for error propagation."""
    eqs = [e.sympy_eq for e in model.equations if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        return None
    target = sp.Symbol(target_name)
    other_targets = [sp.Symbol(t) for t in model.solve_for
                      if t != target_name and target_kind(model, t) == "equation"]
    try:
        sol = sp.solve(eqs, [target, *other_targets], dict=True)
    except Exception:  # noqa: BLE001
        return None
    if not sol or target not in sol[0]:
        return None
    return sol[0][target]


def propagate_uncertainty(expr: sp.Expr, values: dict[sp.Symbol, float],
                           uncertainties: dict[sp.Symbol, float]) -> UncertaintyResult | None:
    """Propagates uncertainty through `expr`, evaluated at `values`, for
    whichever of `expr`'s free symbols appear (with a nonzero value) in
    `uncertainties`. Returns None if no relevant symbol carries an
    uncertainty, or if the expression/partial derivatives can't be
    evaluated numerically at the given point."""
    relevant = [s for s in expr.free_symbols if uncertainties.get(s)]
    if not relevant:
        return None
    try:
        nominal = float(expr.subs(values))
    except (TypeError, ValueError):
        return None

    variance = 0.0
    contributions: dict[str, float] = {}
    for sym in relevant:
        try:
            partial_val = float(sp.diff(expr, sym).subs(values))
        except (TypeError, ValueError):
            continue
        term = (partial_val * uncertainties[sym]) ** 2
        variance += term
        contributions[sym.name] = term ** 0.5

    if not contributions:
        return None
    total = variance ** 0.5
    relative = (total / abs(nominal)) if nominal != 0 else None
    dominant = max(contributions, key=contributions.get)
    return UncertaintyResult(nominal=nominal, uncertainty=total, relative_uncertainty=relative,
                               contributions=contributions, dominant_source=dominant)


def uncertainty_for_target(model: ProblemModel, target_name: str,
                            known_values: dict[sp.Symbol, float]) -> UncertaintyResult | None:
    """Top-level entry point: does this target's model even have any
    variable carrying a stated uncertainty? If not, skip the (nontrivial)
    symbolic re-solve entirely. Otherwise solves symbolically and
    propagates."""
    if target_kind(model, target_name) != "equation":
        return None
    uncertainties = {sp.Symbol(v.symbol): v.uncertainty for v in model.variables
                      if v.uncertainty is not None and v.uncertainty > 0}
    if not uncertainties:
        return None
    expr = solve_symbolic_for_target(model, target_name)
    if expr is None:
        return None
    return propagate_uncertainty(expr, known_values, uncertainties)
