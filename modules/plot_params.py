"""
Shared parameter handling for every plot that can either (a) solve an
equation for a chosen target variable and plot that curve/surface, or (b)
fall back to plotting the equation's residual (lhs - rhs).

Used by both modules/plotter.py (interactive Plotly) and
modules/plot_snapshot.py (static matplotlib export), so the live plot and
its exported snapshot can never disagree about which of the two they drew.

Two real bugs this exists to prevent:

1. A target value in `param_values` silently defeated the solve. The UI
   builds a slider for EVERY free symbol except the plotted axes -- the
   Y/Z target included, since the target selectbox comes after the sliders.
   Each plot function then substituted all of `param_values` into the
   equation BEFORE calling sp.solve(..., target). With the target already
   replaced by a number the equation is a constant truth value, sp.solve
   returns [], and the plot quietly fell back to the residual -- so the
   "Y-axis target" / "Z-axis target" selector never did anything for a
   target that appears in the equation. The target must be solved FOR, so
   it is excluded from the substitution used for solving
   (solve_for_target below).

2. The residual fallback with a missing value failed obscurely. If the
   solve genuinely failed and `param_values` had no entry for some symbol,
   the residual kept that symbol free; lambdify then produced a function
   with an unbound name and the caller crashed with an unrelated-looking
   TypeError/NameError ("Cannot convert expression to float"). Now the
   missing symbols are named up front in a ValueError (residual_expression
   below).
"""
import sympy as sp

from modules.equation_engine import Equation


def solve_for_target(eq: Equation, target_name: str, param_values: dict[str, float]) -> sp.Expr | None:
    """Solves `eq` for `target_name` with every OTHER parameter substituted,
    and returns the solved expression (first solution containing the
    target), or None if sympy can't solve it / the target isn't in the
    equation. `param_values` may include the target itself (the UI always
    supplies a slider for it); that entry is ignored here on purpose --
    see the module docstring."""
    assert eq.sympy_eq is not None  # every caller checks this first and raises a friendlier error
    target = sp.Symbol(target_name)
    subs = {sp.Symbol(k): v for k, v in param_values.items() if k != target_name}
    try:
        solved = sp.solve(eq.sympy_eq.subs(subs), target, dict=True)
    except Exception:  # noqa: BLE001
        return None
    for solution in solved:
        if target in solution:
            return solution[target]
    return None


def residual_expression(eq: Equation, param_values: dict[str, float],
                         plotted_symbols: set[str]) -> sp.Expr:
    """lhs - rhs of `eq` with every parameter substituted. Raises a
    ValueError naming any symbol that is neither plotted on an axis nor
    given a value, rather than letting the unresolved symbol surface later
    as an obscure lambdify/float-conversion failure."""
    assert eq.sympy_eq is not None
    subs = {sp.Symbol(k): v for k, v in param_values.items()}
    residual = (eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs)
    missing = sorted(s.name for s in residual.free_symbols if s.name not in plotted_symbols)
    if missing:
        raise ValueError(
            f"Can't plot {eq.name!r}: no value for {', '.join(missing)}. The equation couldn't be "
            f"solved for the chosen target, so its residual is plotted instead, and that needs a "
            f"value for every symbol other than the axes."
        )
    return residual
