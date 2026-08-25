"""
Solves optimization problems extracted from a word problem: an objective
to minimize/maximize, optionally subject to equation-kind constraints
(handled via direct substitution/elimination where possible, falling back
to Lagrange multipliers when elimination isn't clean) and inequality-kind
constraints.

Scope, stated honestly: inequality-kind constraints are NOT used to derive
the optimum -- doing that properly requires full KKT / boundary analysis,
which is out of scope here. Instead, once a candidate optimum is found via
calculus, it's checked against any inequality constraints and flagged if
it falls outside the feasible region, so the result is upfront about what
it does and doesn't guarantee rather than silently presenting an
unconstrained answer as if it were the constrained one.

Imports _known_substitutions from verifier.py at module level -- this is
safe (no circular import) as long as verifier.py only imports FROM this
module inside a function body (as it already does for ode_utils.py),
never at its own module level.
"""
from dataclasses import dataclass, field
import sympy as sp

from modules.equation_engine import ProblemModel


@dataclass
class OptimizationResult:
    critical_points: list[dict[str, sp.Expr]] = field(default_factory=list)
    classifications: list[str] = field(default_factory=list)  # parallel to critical_points
    used_lagrange: bool = False
    multiplier_values: list[dict[str, sp.Expr]] = field(default_factory=list)  # parallel, if Lagrange
    reduced_objective: sp.Expr | None = None  # after elimination, if that path was used
    eliminated_vars: dict[str, sp.Expr] = field(default_factory=dict)
    feasibility_notes: list[str] = field(default_factory=list)
    error: str | None = None


def _eliminate_greedy(objective_expr: sp.Expr, optimize_over: list[str],
                        constraint_exprs: list[sp.Expr]) -> tuple[sp.Expr | None, dict[str, sp.Expr], list[sp.Symbol]]:
    """Eliminates as many of the objective's free variables as possible
    using available equation-kind constraints (given as expr, meaning
    expr == 0), while always keeping at least one variable free to
    differentiate with respect to.

    Deliberately does NOT protect optimize_over variables from being
    eliminated: if a constraint directly links two requested variables
    (e.g. "maximize xy subject to x+y=10" with optimize_over=["x","y"]
    both), one of them still needs to be eliminated to reduce the problem
    to something differentiable -- the eliminated one's value gets
    recovered afterward via _backfill_point. Only variables that are
    genuinely NOT eliminable and NOT requested cause this to fail (return
    an empty free-variable list), signaling the caller to fall back to
    Lagrange multipliers instead.

    Returns (reduced_objective, eliminated_map, free_vars). free_vars is
    empty on failure; eliminated_map maps eliminated variable NAMES to
    their solved expression (possibly referencing other still-free or
    still-eliminated variables, resolved later by _backfill_point).
    """
    obj_free = {s.name for s in objective_expr.free_symbols}
    if not obj_free:
        return objective_expr, {}, []

    reduced = objective_expr
    eliminated: dict[str, sp.Expr] = {}
    remaining = list(constraint_exprs)
    eligible = set(obj_free)
    optimize_set = set(optimize_over)

    def _ordered_candidates(c_free: set) -> list[str]:
        # deterministic order: prefer eliminating "helper" variables (not
        # explicitly requested) before ever touching a requested
        # optimize_over variable -- eliminating h from "V = pi*r^2*h" to
        # leave r free is always preferable to the reverse when h isn't
        # itself wanted, both for determinism (set iteration order is NOT
        # guaranteed and previously caused the wrong variable to get
        # eliminated at random) and because it keeps the "reduced" problem
        # expressed in terms of what was actually asked for.
        pool = c_free & eligible
        helpers = sorted(pool - optimize_set)
        requested = sorted(pool & optimize_set)
        return helpers + requested

    changed = True
    while changed and len(eligible) > 1:
        changed = False
        for c in remaining:
            c_free = {s.name for s in c.free_symbols}
            for cand in _ordered_candidates(c_free):
                try:
                    sol = sp.solve(c, sp.Symbol(cand))
                except Exception:  # noqa: BLE001
                    sol = []
                if sol:
                    # a constraint like r**2 = ... yields multiple branches;
                    # prefer one that isn't manifestly non-real (e.g. avoid
                    # picking a spurious negative-radius branch arbitrarily)
                    chosen = next((s for s in sol if s.is_real is not False), sol[0])
                    subs_map = {sp.Symbol(cand): chosen}
                    reduced = reduced.subs(subs_map)
                    eliminated[cand] = chosen
                    eligible.discard(cand)
                    remaining.remove(c)
                    changed = True
                    break
            if changed:
                break

    still_free_in_reduced = {s.name for s in reduced.free_symbols}
    # anything left that ISN'T an eligible/free optimize_over var and wasn't
    # eliminated is a genuinely stuck helper variable -- fall back to Lagrange
    truly_stuck = {name for name in still_free_in_reduced
                    if name not in optimize_over and name not in eligible}
    if truly_stuck:
        return None, {}, []

    free_vars = [sp.Symbol(name) for name in eligible if name in still_free_in_reduced]
    if not free_vars:
        return None, {}, []
    return reduced, eliminated, free_vars


def _backfill_point(free_point: dict[sp.Symbol, sp.Expr], eliminated: dict[str, sp.Expr]) -> dict[str, sp.Expr]:
    """Given the solved values for the "kept free" variables, resolves
    every eliminated variable's value by substituting known values in
    repeatedly (handles chains where one eliminated variable's formula
    references another). Returns a flat {name: value} dict covering both
    the free and the eliminated variables."""
    result = {str(k): v for k, v in free_point.items()}
    pending = dict(eliminated)
    for _ in range(len(pending) + 1):
        if not pending:
            break
        progressed = False
        for name, expr in list(pending.items()):
            subs_now = {sp.Symbol(k): v for k, v in result.items()}
            resolved = expr.subs(subs_now)
            if not resolved.free_symbols:
                result[name] = resolved
                del pending[name]
                progressed = True
        if not progressed:
            break
    for name, expr in pending.items():  # best-effort for anything still unresolved
        subs_now = {sp.Symbol(k): v for k, v in result.items()}
        result[name] = expr.subs(subs_now)
    return result


def _classify_critical_point(objective_expr: sp.Expr, variables: list[sp.Symbol],
                               point: dict) -> str:
    """'minimum' | 'maximum' | 'saddle point' | 'inconclusive', via the
    second-derivative test (1 variable) or the Hessian's eigenvalue signs
    (multivariable: all positive -> minimum, all negative -> maximum,
    mixed signs -> saddle point)."""
    if len(variables) == 1:
        v = variables[0]
        second = sp.diff(objective_expr, v, 2)
        try:
            val = float(second.subs(point))
        except (TypeError, ValueError):
            return "inconclusive"
        if val > 1e-12:
            return "minimum"
        if val < -1e-12:
            return "maximum"
        return "inconclusive"

    H = sp.hessian(objective_expr, variables)
    H_at = H.subs(point)
    try:
        eigenvals = [complex(e) for e in H_at.eigenvals().keys()]
    except Exception:  # noqa: BLE001
        return "inconclusive"
    reals = [e.real for e in eigenvals if abs(e.imag) < 1e-9]
    if len(reals) != len(eigenvals):
        return "inconclusive"
    if all(r > 1e-12 for r in reals):
        return "minimum"
    if all(r < -1e-12 for r in reals):
        return "maximum"
    if any(r > 1e-12 for r in reals) and any(r < -1e-12 for r in reals):
        return "saddle point"
    return "inconclusive"


def solve_optimization(model: ProblemModel) -> OptimizationResult | None:
    if model.objective is None or model.objective.sympy_expr is None:
        return None

    from modules.verifier import _known_substitutions  # local import: see module docstring

    obj = model.objective
    optimize_vars = [sp.Symbol(name) for name in obj.optimize_over]
    if not optimize_vars:
        return OptimizationResult(error="No optimize_over variable(s) specified.")

    equation_constraints = [e.sympy_eq for e in model.equations
                             if e.kind == "equation" and e.sympy_eq is not None]
    constraint_exprs = [(eq.lhs - eq.rhs) for eq in equation_constraints]

    knowns = _known_substitutions(model)
    objective_expr = obj.sympy_expr.subs(knowns)
    constraint_exprs = [c.subs(knowns) for c in constraint_exprs]

    obj_free_after_knowns = {s.name for s in objective_expr.free_symbols}
    # try elimination whenever there ARE constraints touching the
    # objective's variables, even if optimize_over already nominally
    # "covers" every free variable -- a constraint can still link two
    # requested variables together (e.g. x+y=10 with both x and y
    # requested), see _eliminate_greedy's docstring
    should_try_elimination = bool(constraint_exprs) and bool(
        obj_free_after_knowns & {s.name for c in constraint_exprs for s in c.free_symbols}
    )

    used_lagrange = False
    reduced_objective = None
    eliminated_vars: dict[str, sp.Expr] = {}
    working_objective = objective_expr
    working_free_vars = optimize_vars

    if should_try_elimination:
        reduced, eliminated, free_vars = _eliminate_greedy(objective_expr, obj.optimize_over, constraint_exprs)
        if reduced is not None and free_vars:
            reduced_objective = reduced
            eliminated_vars = eliminated
            working_objective = reduced
            working_free_vars = free_vars
        else:
            leftover = obj_free_after_knowns - set(obj.optimize_over)
            if leftover and not eliminated:
                return OptimizationResult(
                    error=f"Objective depends on {', '.join(sorted(leftover))}, which is/are "
                           "neither known, in optimize_over, nor eliminable via a constraint."
                )
            used_lagrange = True

    critical_points: list[dict[str, sp.Expr]] = []
    classifications: list[str] = []
    multiplier_values: list[dict[str, sp.Expr]] = []

    if used_lagrange:
        lambdas = [sp.Dummy(f"lambda{i+1}") for i in range(len(constraint_exprs))]
        L = objective_expr - sum(lam * c for lam, c in zip(lambdas, constraint_exprs))
        helper_syms = [sp.Symbol(h) for h in sorted(obj_free_after_knowns - set(obj.optimize_over))]
        all_vars = optimize_vars + helper_syms
        grad_eqs = [sp.diff(L, v) for v in all_vars] + constraint_exprs
        all_unknowns = all_vars + lambdas
        try:
            solutions = sp.solve(grad_eqs, all_unknowns, dict=True)
        except Exception as e:  # noqa: BLE001
            return OptimizationResult(used_lagrange=True, error=f"Could not solve the Lagrange system: {e}")
        solutions = [s for s in solutions if all(v.is_real is not False for v in s.values())]
        if not solutions:
            return OptimizationResult(used_lagrange=True,
                                        error="The Lagrange system had no real solution SymPy could find.")
        for sol in solutions:
            point = {str(k): v for k, v in sol.items() if k in all_vars}
            mult = {str(k): v for k, v in sol.items() if k in lambdas}
            critical_points.append(point)
            multiplier_values.append(mult)
            classifications.append("critical point (constrained -- not second-order classified)")
    else:
        try:
            solutions = sp.solve([sp.diff(working_objective, v) for v in working_free_vars],
                                   working_free_vars, dict=True)
        except Exception as e:  # noqa: BLE001
            return OptimizationResult(reduced_objective=reduced_objective, eliminated_vars=eliminated_vars,
                                        error=f"Could not solve for critical points: {e}")
        # filter out spurious complex roots (e.g. from an odd-degree
        # polynomial critical-point equation) -- only real solutions are
        # physically meaningful optima
        real_solutions = [s for s in solutions if all(v.is_real is not False for v in s.values())]
        if not real_solutions:
            return OptimizationResult(reduced_objective=reduced_objective, eliminated_vars=eliminated_vars,
                                        error="No real-valued critical points found -- the objective may "
                                              "have no interior optimum over the given variable(s).")
        for sol in real_solutions:
            full_point = _backfill_point(sol, eliminated_vars)
            critical_points.append(full_point)
            classifications.append(_classify_critical_point(working_objective, working_free_vars, sol))

    feasibility_notes = []
    inequality_constraints = [e for e in model.equations if e.kind == "inequality" and e.sympy_eq is not None]
    if inequality_constraints and critical_points:
        for point in critical_points:
            full_point = {sp.Symbol(k): v for k, v in point.items()}
            full_point.update(knowns)
            for ineq in inequality_constraints:
                try:
                    truth = bool(ineq.sympy_eq.subs(full_point))
                except Exception:  # noqa: BLE001
                    continue
                if not truth:
                    feasibility_notes.append(
                        f"Critical point {point} violates constraint '{ineq.name}' -- the true "
                        "constrained optimum may lie on the constraint boundary instead, which "
                        "this solver does not search for."
                    )

    return OptimizationResult(
        critical_points=critical_points,
        classifications=classifications,
        used_lagrange=used_lagrange,
        multiplier_values=multiplier_values,
        reduced_objective=reduced_objective,
        eliminated_vars=eliminated_vars,
        feasibility_notes=feasibility_notes,
    )
