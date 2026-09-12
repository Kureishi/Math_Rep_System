"""
Shared ODE-solving helper used by both solver.py (to build step-by-step
output) and verifier.py (to symbolically verify solutions via
checkodesol). Centralized here rather than in either of those two modules
because solver.py already imports from verifier.py, and verifier.py
needs this too -- putting it in either one would create a circular import.

Handles both single ODEs (dsolve) and coupled SYSTEMS of ODEs
(dsolve_system) -- e.g. a decay chain A -> B where B's rate depends on A,
or a predator-prey pair. Equations are grouped into independent "coupling
groups" by which function names they actually share; each group is solved
together so cross-coupling is respected, but unrelated ODEs in the same
problem don't force each other into one (possibly unsolvable) joint system.
"""
import sympy as sp
import numpy as np
from scipy.integrate import solve_ivp
from sympy.core.function import AppliedUndef
from sympy.solvers.ode.systems import dsolve_system
from sympy.solvers.deutils import ode_order
from dataclasses import dataclass, field

from modules.equation_engine import ProblemModel, Equation
from modules.timeout_utils import run_with_timeout


def _funcs_used(eq: sp.Eq) -> set[str]:
    return {str(f.func) for f in eq.atoms(AppliedUndef)}


def group_coupled_odes(ode_equations: list[Equation]) -> list[list[Equation]]:
    """Groups ode-kind equations into independent coupling groups: two
    equations land in the same group iff they share at least one function
    name (directly, or transitively through a chain of shared equations).
    A group of size 1 is just a normal standalone ODE."""
    groups: list[dict] = []
    for eq in ode_equations:
        if eq.sympy_eq is None:
            continue
        names = _funcs_used(eq.sympy_eq)
        merged = next((g for g in groups if g["names"] & names), None)
        if merged:
            merged["eqs"].append(eq)
            merged["names"] |= names
        else:
            groups.append({"eqs": [eq], "names": set(names)})

    # second pass: merge any groups that turned out to overlap once all
    # equations were seen (handles A-B and B-C being defined in either order)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                if groups[i]["names"] & groups[j]["names"]:
                    groups[i]["eqs"] += groups[j]["eqs"]
                    groups[i]["names"] |= groups[j]["names"]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break

    return [g["eqs"] for g in groups]


def _ics_for_group(model: ProblemModel, func_names: set[str]) -> dict[sp.Basic, float]:
    ics = {}
    for ic in model.initial_conditions:
        if ic.sympy_eq is None:
            continue
        lhs_funcs = ic.sympy_eq.lhs.atoms(AppliedUndef)
        if lhs_funcs and str(next(iter(lhs_funcs)).func) in func_names:
            ics[ic.sympy_eq.lhs] = ic.sympy_eq.rhs
    return ics


def solve_ode(model: ProblemModel) -> dict[str, sp.Eq]:
    """Solves every ode-kind equation, applying any initial conditions that
    match. Coupled equations (sharing a function across equations) are
    solved together via dsolve_system; standalone ODEs use plain dsolve.
    Returns {function_name: solution_Eq}, flattened across all groups."""
    ode_equations = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    if not ode_equations:
        return {}

    result: dict[str, sp.Eq] = {}

    for group in group_coupled_odes(ode_equations):
        sympy_eqs = [e.sympy_eq for e in group if e.sympy_eq is not None]  # redundant filter,
        # narrows the type for mypy -- ode_equations (and therefore every group drawn from it)
        # was already filtered on this above
        func_names = set()
        for e in sympy_eqs:
            func_names |= _funcs_used(e)
        ics = _ics_for_group(model, func_names)

        if len(group) == 1:
            func_applied = next(iter(sympy_eqs[0].atoms(AppliedUndef)))
            try:
                if ics:
                    sol = run_with_timeout(sp.dsolve, sympy_eqs[0], func_applied, ics=ics, label="dsolve")
                else:
                    sol = run_with_timeout(sp.dsolve, sympy_eqs[0], func_applied, label="dsolve")
                result[str(func_applied.func)] = sol
            except Exception:  # noqa: BLE001 -- includes ComputationTimeoutError; this
                continue        # ODE is simply skipped from the result dict, same as any
                                 # other dsolve failure (e.g. no closed form found)
        else:
            # figure out t and the ordered list of Function applications
            # dsolve_system wants (e.g. A(t), B(t)), not just names
            applied_funcs = []
            seen = set()
            for e in sympy_eqs:
                for f in e.atoms(AppliedUndef):
                    name = str(f.func)
                    if name not in seen:
                        seen.add(name)
                        applied_funcs.append(f)
            t = next(iter(applied_funcs[0].args))  # shared independent variable
            try:
                sol_sets = run_with_timeout(dsolve_system, sympy_eqs, funcs=applied_funcs, t=t,
                                              ics=ics or None, label="dsolve_system")
                for sol_eq in sol_sets[0]:
                    result[str(sol_eq.lhs.func)] = sol_eq
            except Exception:  # noqa: BLE001
                continue

    return result


def verify_coupled_solution(group: list[Equation], solutions: dict[str, sp.Eq]) -> tuple[bool, sp.Basic]:
    """Verifies a coupled system's solution the correct way: substitutes
    ALL functions' solutions into EACH original equation simultaneously and
    checks the residual is zero. checkodesol alone can't be used here since
    it only knows about one function/equation at a time and doesn't
    understand cross-coupling between equations."""
    sol_map = {}
    for eq in group:
        assert eq.sympy_eq is not None  # callers only ever pass groups drawn from a
        # kind == "ode" and sympy_eq is not None filtered list -- see solve_ode/group_coupled_odes
        for f in eq.sympy_eq.atoms(AppliedUndef):
            name = str(f.func)
            if name in solutions:
                sol_map[f] = solutions[name].rhs

    worst_residual = sp.Integer(0)
    for eq in group:
        assert eq.sympy_eq is not None
        lhs = eq.sympy_eq.lhs.subs(sol_map).doit()
        rhs = eq.sympy_eq.rhs.subs(sol_map)
        try:
            residual = sp.simplify(lhs - rhs)
        except Exception:  # noqa: BLE001
            residual = lhs - rhs
        if residual != 0:
            return False, residual
    return True, worst_residual


# --------------------------------------------------------- independent numerical cross-check


@dataclass
class NumericalCrossCheckResult:
    applicable: bool                     # False if this ODE/group doesn't meet the scope below
    ok: bool | None = None               # None whenever applicable is False
    max_relative_error: float | None = None
    sample_points: list[float] = field(default_factory=list)
    reason: str = ""                     # why not applicable, OR a summary when it is


# tolerance for "the two independent solve paths agree" -- looser than the
# symbolic checks' effectively-exact-zero standard, since this is comparing
# a symbolic closed form against a NUMERICAL integration (RK45, adaptive
# step), which itself carries integration error on the order of the
# solver's rtol/atol; 1e-4 relative is comfortably above that noise floor
# while still catching a genuinely wrong solution (which was on the order
# of 100-1000x off in testing, not a marginal few percent -- see this
# module's test suite for the deliberately-wrong-sign case this catches)
_CROSS_CHECK_RELATIVE_TOLERANCE = 1e-4


def _defined_function(eq_sympy: sp.Eq):
    """The function this equation actually DEFINES the derivative of --
    i.e. whatever's inside the Derivative(...) it contains -- NOT just
    any AppliedUndef atom the equation happens to mention.

    This distinction is a real, previously-shipped bug: a coupled
    equation like Eq(Derivative(B(t), t), k1*A(t) - k2*B(t)) contains
    BOTH A(t) and B(t) as AppliedUndef atoms (A(t) appears
    undifferentiated on the right), so `next(iter(eq.atoms(AppliedUndef)))`
    picks an ARBITRARY one of the two -- Python set iteration order
    depends on hash values, and Python randomizes string hashing (and
    therefore, transitively, sympy Symbol/Function hashing) per process
    by default, so this silently grabbed the wrong function in SOME
    process runs and the right one in others, corrupting the derived
    ODE order and numeric right-hand side intermittently. That's what
    test_correct_coupled_system_confirmed_by_independent_integration
    caught: the same test, unchanged, failed in one interpreter
    invocation and passed in the next, purely from hash-seed
    randomization -- the actual signature of this exact bug class."""
    derivatives = eq_sympy.atoms(sp.Derivative)
    funcs = {d.expr for d in derivatives if isinstance(d.expr, AppliedUndef)}
    if len(funcs) != 1:
        return None
    return next(iter(funcs))


def numerical_cross_check(model: ProblemModel, group: list[Equation],
                            solutions: dict[str, sp.Eq]) -> NumericalCrossCheckResult:
    """A SECOND, INDEPENDENT solve path for an initial-value ODE problem,
    alongside the symbolic checkodesol/verify_coupled_solution check
    _ode_checks already does -- this integrates the ORIGINAL differential
    equation(s) numerically (scipy's adaptive RK45, via solve_ivp) from
    the same initial condition, then compares the numerical trajectory
    against the symbolic closed-form solution at several sample points.

    This exists because checkodesol-style verification has a real,
    narrow blind spot: it confirms the closed-form solution satisfies
    the DIFFERENTIAL EQUATION (a true statement about the whole solution
    family), but does NOT independently re-confirm that dsolve's
    ics=-driven constant-solving actually landed on the constant
    matching the SPECIFIC given initial condition -- if dsolve picked a
    wrong root while solving for an integration constant (plausible for
    equations with sign ambiguity, e.g. from a square root), checkodesol
    would still report success, since the resulting expression genuinely
    does satisfy the ODE -- just not the one matching the stated initial
    value. A numerical integration started from the SAME initial
    condition has no such blind spot: it has no algebraic constant-
    solving step to get wrong in the first place, so a mismatch here is
    a strong, independent signal something is actually wrong, not a
    restatement of the same computation the symbolic check already did.

    SCOPED to first-order initial-value problems (every equation in the
    group has ode_order 1, an initial condition exists for every
    function, and every OTHER symbol appearing in the equations has a
    known numeric value in the model) -- a real, documented limitation,
    not a silent gap: higher-order ODEs would need converting to a
    first-order companion system with correctly-ordered derivative
    initial conditions (y'(0), y''(0), ...), which the current
    initial-condition representation doesn't reliably distinguish from
    plain y(0); extending to that is future work, not attempted here
    rather than risking a wrong companion-state mapping.
    """
    orders = []
    for eq in group:
        func = _defined_function(eq.sympy_eq)
        if func is None:
            return NumericalCrossCheckResult(applicable=False,
                                               reason="Could not identify the function in this equation.")
        try:
            orders.append(ode_order(eq.sympy_eq, func))
        except Exception:  # noqa: BLE001
            return NumericalCrossCheckResult(applicable=False, reason="Could not determine ODE order.")
    if any(o != 1 for o in orders):
        return NumericalCrossCheckResult(
            applicable=False,
            reason="Numerical cross-check is scoped to first-order equations; this group includes "
                   "a higher-order ODE, which isn't attempted (see numerical_cross_check's docstring).")

    func_names = set()
    for eq in group:
        func_names |= _funcs_used(eq.sympy_eq)
    if any(name not in solutions for name in func_names):
        return NumericalCrossCheckResult(applicable=False,
                                           reason="Not every function in this group has a closed-form solution.")

    ics = _ics_for_group(model, func_names)
    # every function needs its OWN initial condition, all at the SAME t0,
    # for this to be a well-posed initial-value problem to integrate
    func_order = sorted(func_names)  # fixed order for the state vector
    ic_by_func = {}
    t0 = None
    for applied, value in ics.items():
        name = str(applied.func)
        if name not in func_order:
            continue
        arg = applied.args[0]
        if not arg.is_number:
            continue  # not a plain y(t0)-style IC (e.g. a derivative IC) -- out of scope, see docstring
        if t0 is None:
            t0 = float(arg)
        elif float(arg) != t0:
            return NumericalCrossCheckResult(applicable=False,
                                               reason="Initial conditions for this group are given at "
                                                      "different points -- not a standard IVP to integrate.")
        ic_by_func[name] = float(value)
    if t0 is None or any(name not in ic_by_func for name in func_order):
        return NumericalCrossCheckResult(applicable=False,
                                           reason="Not every function in this group has a plain "
                                                  "y(t0)=value initial condition to integrate from.")

    first_eq = group[0].sympy_eq
    assert first_eq is not None  # guaranteed: group drawn from an already-filtered equation list
    t = next(iter(first_eq.atoms(AppliedUndef))).args[0]
    known_values = {sp.Symbol(v.symbol): v.known_value for v in model.variables
                    if v.known_value is not None}
    applied_by_name = {name: sp.Function(name)(t) for name in func_order}

    rhs_exprs = []
    for eq in group:
        func = _defined_function(eq.sympy_eq)
        if func is None:
            return NumericalCrossCheckResult(applicable=False,
                                               reason=f"Could not identify the function in {eq.name}.")
        deriv = func.diff(t)
        try:
            solved = sp.solve(eq.sympy_eq, deriv)
        except Exception:  # noqa: BLE001
            return NumericalCrossCheckResult(applicable=False,
                                               reason=f"Could not isolate the derivative in {eq.name}.")
        if not solved:
            return NumericalCrossCheckResult(applicable=False,
                                               reason=f"Could not isolate the derivative in {eq.name}.")
        expr = solved[0].subs(known_values)
        remaining = expr.free_symbols - set(applied_by_name.values()) - {t}
        if remaining:
            return NumericalCrossCheckResult(
                applicable=False,
                reason=f"{eq.name} has symbol(s) {sorted(str(s) for s in remaining)} with no known "
                        "numeric value -- can't build a numeric right-hand side without them.")
        rhs_exprs.append(expr)

    try:
        rhs_funcs = [sp.lambdify((t, *[applied_by_name[n] for n in func_order]), expr, "numpy")
                     for expr in rhs_exprs]
        sol_funcs = [sp.lambdify(t, solutions[n].rhs.subs(known_values), "numpy") for n in func_order]
    except Exception as exc:  # noqa: BLE001
        return NumericalCrossCheckResult(applicable=False, reason=f"Could not build a numeric form: {exc}")

    y0 = [ic_by_func[n] for n in func_order]

    def rhs(_t, y):
        return [f(_t, *y) for f in rhs_funcs]

    try:
        rate0 = float(np.linalg.norm(rhs(t0, y0)))
        scale0 = max(float(np.linalg.norm(y0)), 1e-9)
        window = 3.0 / (rate0 / scale0) if rate0 / scale0 > 1e-9 else 5.0
        window = min(max(window, 1e-6), 1e6)  # guard against a pathological/degenerate estimate
    except Exception:  # noqa: BLE001
        window = 5.0

    try:
        ivp = run_with_timeout(solve_ivp, rhs, (t0, t0 + window), y0, label="ode_numerical_cross_check",
                                dense_output=True, rtol=1e-8, atol=1e-10, method="RK45")
    except Exception as exc:  # noqa: BLE001
        return NumericalCrossCheckResult(applicable=False, reason=f"Numerical integration failed: {exc}")
    if not ivp.success:
        return NumericalCrossCheckResult(applicable=False,
                                           reason=f"Numerical integration did not converge: {ivp.message}")

    sample_ts = list(np.linspace(t0, t0 + window, 5))
    try:
        numeric_vals = ivp.sol(sample_ts)  # shape (n_funcs, n_samples)
        symbolic_vals = np.array([sf(np.array(sample_ts)) for sf in sol_funcs])
    except Exception as exc:  # noqa: BLE001
        return NumericalCrossCheckResult(applicable=False, reason=f"Could not evaluate for comparison: {exc}")

    denom = np.maximum(np.abs(symbolic_vals), 1e-9)
    rel_errors = np.abs(numeric_vals - symbolic_vals) / denom
    max_rel_error = float(np.max(rel_errors))
    ok = max_rel_error < _CROSS_CHECK_RELATIVE_TOLERANCE

    detail = (f"Independent numerical integration (scipy RK45) from the same initial condition(s) "
              f"{'agrees' if ok else 'DISAGREES'} with the symbolic closed-form solution "
              f"(max relative difference {max_rel_error:.2e} across {len(sample_ts)} sample points "
              f"over t in [{t0:g}, {t0 + window:g}]).")
    return NumericalCrossCheckResult(applicable=True, ok=ok, max_relative_error=max_rel_error,
                                       sample_points=sample_ts, reason=detail)
