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
from sympy.core.function import AppliedUndef
from sympy.solvers.ode.systems import dsolve_system

from modules.equation_engine import ProblemModel, Equation


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
        sympy_eqs = [e.sympy_eq for e in group]
        func_names = set()
        for e in sympy_eqs:
            func_names |= _funcs_used(e)
        ics = _ics_for_group(model, func_names)

        if len(group) == 1:
            func_applied = next(iter(sympy_eqs[0].atoms(AppliedUndef)))
            try:
                sol = sp.dsolve(sympy_eqs[0], func_applied, ics=ics) if ics else sp.dsolve(sympy_eqs[0], func_applied)
                result[str(func_applied.func)] = sol
            except Exception:  # noqa: BLE001
                continue
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
                sol_sets = dsolve_system(sympy_eqs, funcs=applied_funcs, t=t, ics=ics or None)
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
        for f in eq.sympy_eq.atoms(AppliedUndef):
            name = str(f.func)
            if name in solutions:
                sol_map[f] = solutions[name].rhs

    worst_residual = sp.Integer(0)
    for eq in group:
        lhs = eq.sympy_eq.lhs.subs(sol_map).doit()
        rhs = eq.sympy_eq.rhs.subs(sol_map)
        try:
            residual = sp.simplify(lhs - rhs)
        except Exception:  # noqa: BLE001
            residual = lhs - rhs
        if residual != 0:
            return False, residual
    return True, worst_residual
