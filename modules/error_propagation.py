"""
Closed-form (analytic) error propagation: the standard first-order
"propagation of uncertainty" formula taught in every intro physics/chem
lab --

    sigma_f^2 = sum_i (df/dx_i)^2 * sigma_i^2

-- for a target f that depends on one or more independent uncertain
inputs x_i, each with its own standard deviation sigma_i. Exact for a
linear f, and an excellent local approximation for a mildly nonlinear
one; this is the textbook-standard alternative to monte_carlo.py's
sampling-based approach.

The two modules deliberately coexist rather than one replacing the
other: this one is instant (no sampling), produces an actual FORMULA
the student can see and learn from (not just a histogram), and matches
what an intro course grades against by name -- but it's only a good
approximation when the target is reasonably smooth and the input
uncertainties are small relative to the curvature of f. monte_carlo.py
is the right tool when that assumption is shaky (highly nonlinear f,
large uncertainties, or a target with real domain boundaries the linear
approximation would happily step over).

Solves the system symbolically ONCE to get a closed-form expression for
the target (mirroring monte_carlo.py's own "solve once, evaluate many
times" approach, since the same expression is reused here to compute
each partial derivative), then evaluates every partial derivative at the
given central (mean) values -- no sampling, no numerical solving.
"""
from dataclasses import dataclass, field

import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError


@dataclass
class UncertainVariable:
    symbol: str
    mean: float
    std: float    # standard deviation; must be > 0 -- a variable with no real
                   # uncertainty shouldn't be in this list


@dataclass
class ErrorPropagationResult:
    target: str
    value: float                        # f evaluated at the central (mean) values
    std: float                          # sigma_f, the propagated standard deviation
    contributions: dict = field(default_factory=dict)  # symbol -> fraction of the
                                          # total VARIANCE contributed by that input
                                          # (sums to 1.0) -- a tornado-chart-style
                                          # breakdown of which input's uncertainty
                                          # actually matters
    partials: dict = field(default_factory=dict)       # symbol -> df/dx_i, evaluated
                                          # at the central values (the raw sensitivity,
                                          # before being squared/weighted by sigma_i)
    formula_latex: str | None = None    # LaTeX of the closed-form expression for `target`


def propagate_error(model: ProblemModel, target: str,
                     uncertain_vars: list[UncertainVariable]) -> ErrorPropagationResult:
    """Propagates independent (uncorrelated) input uncertainties through
    to `target` via the first-order formula above. Assumes the
    uncertain variables are statistically independent -- covariance
    terms aren't modeled, the same simplifying assumption most intro
    treatments of this formula make."""
    if target not in model.solve_for or target_kind(model, target) != "equation":
        raise ValueError(f"'{target}' isn't an algebraic solve_for target of this model.")
    if not uncertain_vars:
        raise ValueError("Need at least one uncertain variable to propagate.")
    for uv in uncertain_vars:
        if uv.std <= 0:
            raise ValueError(f"'{uv.symbol}' has a non-positive std ({uv.std}) -- not actually uncertain.")

    uncertain_symbols = {uv.symbol for uv in uncertain_vars}
    fixed_subs = {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables
        if v.known_value is not None and v.symbol not in uncertain_symbols
    }
    eqs = [e.sympy_eq.subs(fixed_subs) for e in model.equations
           if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        raise ValueError("This model has no algebraic equations to solve.")

    algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    target_syms = [sp.Symbol(t) for t in algebraic_targets]
    try:
        sol = run_with_timeout(sp.solve, eqs, target_syms, dict=True, label="error propagation solve")
    except ComputationTimeoutError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't symbolically solve for '{target}': {e}") from e
    if not sol:
        raise ValueError(f"Couldn't symbolically solve for '{target}' given the fixed inputs.")

    target_expr = sol[0].get(sp.Symbol(target))
    if target_expr is None:
        raise ValueError(f"'{target}' didn't appear in the symbolic solution.")

    mean_subs = {sp.Symbol(uv.symbol): uv.mean for uv in uncertain_vars}

    variance = 0.0
    contributions_raw: dict[str, float] = {}
    partials: dict[str, float] = {}
    for uv in uncertain_vars:
        sym = sp.Symbol(uv.symbol)
        if sym not in target_expr.free_symbols:
            # the target doesn't actually depend on this input -- zero
            # sensitivity, not an error; still reported (as 0) so the
            # caller can see it was considered
            partials[uv.symbol] = 0.0
            contributions_raw[uv.symbol] = 0.0
            continue
        try:
            deriv = sp.diff(target_expr, sym)
            deriv_at_mean = float(deriv.subs(mean_subs))
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Couldn't differentiate the target with respect to "
                              f"'{uv.symbol}': {e}") from e
        partials[uv.symbol] = deriv_at_mean
        term_variance = (deriv_at_mean * uv.std) ** 2
        contributions_raw[uv.symbol] = term_variance
        variance += term_variance

    try:
        central_value = float(target_expr.subs(mean_subs))
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't evaluate the target at the given central values: {e}") from e

    total = sum(contributions_raw.values())
    contributions = ({k: v / total for k, v in contributions_raw.items()} if total > 0
                      else {k: 0.0 for k in contributions_raw})

    return ErrorPropagationResult(
        target=target, value=central_value, std=variance ** 0.5,
        contributions=contributions, partials=partials, formula_latex=sp.latex(target_expr),
    )
