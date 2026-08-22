"""
Shared ODE-solving helper used by both solver.py (to build step-by-step
output) and verifier.py (to symbolically verify solutions via
checkodesol). Centralized here rather than in either of those two modules
because solver.py already imports from verifier.py, and verifier.py
needs this too -- putting it in either one would create a circular import.
"""
import sympy as sp
from sympy.core.function import AppliedUndef

from modules.equation_engine import ProblemModel


def solve_ode(model: ProblemModel) -> dict[str, sp.Eq]:
    """dsolve()s every ode-kind equation, applying any initial conditions
    that match its function. Returns {function_name: solution_Eq}. The
    independent variable is read off the parsed function application
    itself (func.args[0]) rather than model.independent_variable, so this
    is robust even if that field was left blank or mismatched."""
    result: dict[str, sp.Eq] = {}
    for e in model.equations:
        if e.kind != "ode" or e.sympy_eq is None:
            continue
        funcs = e.sympy_eq.atoms(AppliedUndef)
        if not funcs:
            continue
        func_applied = next(iter(funcs))  # e.g. y(t)
        func_name = str(func_applied.func)

        ics = {}
        for ic in model.initial_conditions:
            if ic.sympy_eq is None:
                continue
            lhs_funcs = ic.sympy_eq.lhs.atoms(AppliedUndef)
            if lhs_funcs and str(next(iter(lhs_funcs)).func) == func_name:
                ics[ic.sympy_eq.lhs] = ic.sympy_eq.rhs

        try:
            sol = sp.dsolve(e.sympy_eq, func_applied, ics=ics) if ics else sp.dsolve(e.sympy_eq, func_applied)
            result[func_name] = sol
        except Exception:  # noqa: BLE001
            continue
    return result
