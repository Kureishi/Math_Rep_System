"""
N-dimensional parameter sweep: chains.sweep_step_binding() sweeps ONE
input across a chain; the interactive plot section sweeps one input for
a 2D line. Neither lets someone grid-sweep TWO OR MORE inputs of a
single problem's own equation at once and get a results TABLE (or a
heatmap, for exactly two swept inputs) out -- the actual shape of a
real sensitivity study ("how does the answer vary across every
combination of these 5 masses and these 5 forces"), not just a single
line read one point at a time.

Solves the model's target symbolically ONCE -- the same "solve once,
evaluate the closed form many times" pattern monte_carlo.py,
error_propagation.py, and interval_arithmetic.py all use -- and
evaluates that one expression, vectorized via numpy's own meshgrid, over
the full cartesian-product grid. A 10x10 sweep is one lambdify call and
one vectorized array evaluation, not 100 separate sp.solve() calls.
"""
from dataclasses import dataclass, field

import numpy as np
import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

MAX_GRID_POINTS = 10000  # a sanity cap on the TOTAL number of grid points (the product of
                           # every swept variable's own point count) -- generous for an
                           # interactive sweep, but this is a full cartesian product, so it
                           # grows fast: 4 variables x 10 points each is already 10,000


@dataclass
class SweepResult:
    target: str
    swept_symbols: list           # in the fixed order used throughout `rows`
    rows: list = field(default_factory=list)   # one dict per grid point: {symbol: value, ..., target: value}


def sweep_parameters(model: ProblemModel, target: str, sweep_values: dict) -> SweepResult:
    """`sweep_values` maps a symbol name -> the list of values to sweep
    it across (2+ entries in the dict is a genuine N-dimensional grid;
    a single entry degenerates to a plain 1D sweep, same result shape).
    Every OTHER known value in the model is held fixed at whatever it's
    currently set to. A value the target doesn't actually depend on
    still appears as its own column in every row (held at each of its
    own swept values) even though it has no effect on the answer --
    dropping it silently would make the output table's row count not
    match what was actually asked to be swept."""
    if target not in model.solve_for or target_kind(model, target) != "equation":
        raise ValueError(f"'{target}' isn't an algebraic solve_for target of this model.")
    if not sweep_values:
        raise ValueError("Need at least one variable to sweep.")
    for sym, values in sweep_values.items():
        if not values:
            raise ValueError(f"'{sym}' has no values to sweep over.")

    swept_symbols = list(sweep_values.keys())
    total_points = 1
    for values in sweep_values.values():
        total_points *= len(values)
    if total_points > MAX_GRID_POINTS:
        raise ValueError(f"This grid has {total_points} points, over the cap of "
                          f"{MAX_GRID_POINTS} -- reduce the number of swept variables or the "
                          "number of values per variable.")

    fixed_subs = {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables
        if v.known_value is not None and v.symbol not in swept_symbols
    }
    eqs = [e.sympy_eq.subs(fixed_subs) for e in model.equations
           if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        raise ValueError("This model has no algebraic equations to solve.")

    algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    target_syms = [sp.Symbol(t) for t in algebraic_targets]
    try:
        sol = run_with_timeout(sp.solve, eqs, target_syms, dict=True, label="parameter sweep solve")
    except ComputationTimeoutError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ValueError(f"Couldn't symbolically solve for '{target}': {e}") from e
    if not sol:
        raise ValueError(f"Couldn't symbolically solve for '{target}' given the fixed inputs.")

    target_expr = sol[0].get(sp.Symbol(target))
    if target_expr is None:
        raise ValueError(f"'{target}' didn't appear in the symbolic solution.")

    grids = np.meshgrid(*[np.array(sweep_values[s], dtype=float) for s in swept_symbols], indexing="ij")
    grid_by_symbol = dict(zip(swept_symbols, grids))

    used_symbols = [s for s in swept_symbols if sp.Symbol(s) in target_expr.free_symbols]
    if not used_symbols:
        # the target doesn't depend on ANY of the swept variables -- constant across the grid
        value = float(target_expr)
        result_grid = np.full(grids[0].shape, value)
    else:
        f = sp.lambdify([sp.Symbol(s) for s in used_symbols], target_expr, "numpy")
        args = [grid_by_symbol[s] for s in used_symbols]
        try:
            raw = np.broadcast_to(np.asarray(f(*args), dtype=complex), grids[0].shape)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Couldn't evaluate the target over this grid: {e}") from e
        is_real = np.abs(raw.imag) < 1e-9
        real_vals = raw.real
        result_grid = np.where(is_real & np.isfinite(real_vals), real_vals, np.nan)

    rows: list[dict[str, float | None]] = []
    it = np.nditer(result_grid, flags=["multi_index"])
    for val in it:
        val_scalar = val.item()  # type: ignore[attr-defined]  # numpy's nditer stub mistypes
        # the per-element loop variable as a tuple-of-arrays; at runtime, with just the
        # multi_index flag (no per-operand flags), each `val` is genuinely a 0-d array
        # element and .item() works exactly as expected -- a stub inaccuracy, not a real issue
        idx = it.multi_index
        row: dict[str, float | None] = {s: float(sweep_values[s][idx[i]]) for i, s in enumerate(swept_symbols)}
        row[target] = None if np.isnan(val_scalar) else float(val_scalar)
        rows.append(row)

    return SweepResult(target=target, swept_symbols=swept_symbols, rows=rows)


def sweep_result_to_grid(result: SweepResult):
    """Reconstructs (x_values, y_values, z_matrix) from a 2-variable
    SweepResult's flat `rows`, for a heatmap/contour plot -- the row
    order is already the 'ij'-indexed meshgrid order sweep_parameters()
    produced, so this is a plain reshape, not a re-derivation. Raises
    ValueError if `result` doesn't have exactly two swept symbols."""
    if len(result.swept_symbols) != 2:
        raise ValueError(f"Need exactly 2 swept variables for a grid, got {len(result.swept_symbols)}.")
    x_symbol, y_symbol = result.swept_symbols
    x_values = sorted({row[x_symbol] for row in result.rows})
    y_values = sorted({row[y_symbol] for row in result.rows})
    x_index = {v: i for i, v in enumerate(x_values)}
    y_index = {v: i for i, v in enumerate(y_values)}
    z_matrix = np.full((len(y_values), len(x_values)), np.nan)
    for row in result.rows:
        if row[result.target] is not None:
            z_matrix[y_index[row[y_symbol]], x_index[row[x_symbol]]] = row[result.target]
    return x_values, y_values, z_matrix
