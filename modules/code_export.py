"""
Exports a derived formula as small, runnable Python SOURCE TEXT -- via
sp.pycode, not sp.lambdify. lambdify builds a live, compiled Python
closure that's useful for evaluating a formula in-process (this app uses
it elsewhere, e.g. plotter.py), but a closure can't be saved to a file,
read by a person, or dropped into someone else's codebase. pycode
renders the same math as actual readable source.

Scoped to the three kinds of target that reduce to "one formula, some
named inputs, evaluate it": algebraic targets (re-solved symbolically --
see uncertainty.solve_symbolic_for_target, reused here for exactly the
same reason it exists there), ODE closed-form solutions (a function of
the independent variable), and recurrence closed-form solutions (same,
discrete case). Optimization results are a single numeric critical
point, not a general-purpose formula in terms of arbitrary inputs, so
exporting "the answer is 42" as a function wouldn't be a useful function
-- left out of scope rather than faked.
"""
from dataclasses import dataclass
import sympy as sp

from modules.equation_engine import ProblemModel, target_kind
from modules.uncertainty import solve_symbolic_for_target


@dataclass
class ExportableFormula:
    target_name: str
    expr: sp.Expr
    arg_names: list[str]           # ordered (sorted), the function's parameters
    kind: str                      # "algebraic" | "ode" | "recurrence"
    independent_var: str | None = None  # e.g. "t" or "n", for ode/recurrence


def formula_for_target(model: ProblemModel, target_name: str) -> ExportableFormula | None:
    """Finds a single formula for target_name suitable for exporting as a
    plain Python function, or None if this target's kind isn't
    supported (optimization) or no closed form exists (e.g. an ODE
    SymPy couldn't solve symbolically)."""
    kind = target_kind(model, target_name)

    if kind == "equation":
        expr = solve_symbolic_for_target(model, target_name)
        if expr is None:
            return None
        arg_names = sorted(s.name for s in expr.free_symbols)
        return ExportableFormula(target_name, expr, arg_names, "algebraic")

    if kind == "ode":
        from modules.ode_utils import solve_ode
        sol_eq = solve_ode(model).get(target_name)
        if sol_eq is None:
            return None
        rhs = sol_eq.rhs
        arg_names = sorted(s.name for s in rhs.free_symbols)
        return ExportableFormula(target_name, rhs, arg_names, "ode",
                                   independent_var=model.independent_variable)

    if kind == "recurrence":
        from modules.recurrence_utils import solve_recurrence
        rhs = solve_recurrence(model).get(target_name)
        if rhs is None:
            return None
        arg_names = sorted(s.name for s in rhs.free_symbols)
        return ExportableFormula(target_name, rhs, arg_names, "recurrence",
                                   independent_var=model.independent_variable)

    return None  # optimization, or anything else without a general formula


def generate_python_function(formula: ExportableFormula,
                               variable_meanings: dict[str, str] | None = None,
                               unit: str | None = None, include_import: bool = True) -> str:
    """Renders `formula` as standalone Python function source text
    (not a live callable). Parameter order matches formula.arg_names.
    `include_import` is set False by generate_python_module(), which
    already emits a single shared "import math" once at the top of the
    bundled file rather than repeating it before every function."""
    variable_meanings = variable_meanings or {}
    body = sp.pycode(formula.expr)
    needs_math = "math." in body

    lines: list[str] = []
    if needs_math and include_import:
        lines.append("import math")
        lines.append("")
        lines.append("")

    lines.append(f"def {formula.target_name}({', '.join(formula.arg_names)}):")
    lines.append(f'    """Computes {formula.target_name}' + (f' ({unit})' if unit else '') + '.')
    if formula.independent_var:
        lines.append("")
        lines.append(f"    Closed-form {formula.kind} solution, as a function of "
                       f"{formula.independent_var}.")
    if formula.arg_names:
        lines.append("")
        lines.append("    Args:")
        for a in formula.arg_names:
            meaning = variable_meanings.get(a)
            lines.append(f"        {a}: {meaning}" if meaning else f"        {a}")
    lines.append('    """')
    lines.append(f"    return {body}")
    return "\n".join(lines) + "\n"


def generate_python_module(model: ProblemModel, target_names: list[str] | None = None) -> str:
    """Bundles every exportable target's formula (algebraic/ODE/
    recurrence -- see formula_for_target) into one standalone .py file,
    with a __main__ demo block that calls each function using the
    problem's own known values wherever every required argument is
    available. target_names defaults to model.solve_for."""
    target_names = target_names or model.solve_for
    variable_meanings = {v.symbol: v.meaning for v in model.variables}
    known_values = {v.symbol: v.known_value for v in model.variables if v.known_value is not None}
    unit_by_symbol = {v.symbol: v.unit for v in model.variables}

    functions: list[ExportableFormula] = []
    for t in target_names:
        f = formula_for_target(model, t)
        if f is not None:
            functions.append(f)

    if not functions:
        return ('"""No exportable closed-form formula was found for this problem\'s '
                'target(s) (this happens for optimization results, which are a single '
                'numeric answer rather than a general formula, or for an ODE/recurrence '
                'SymPy could not solve symbolically)."""\n')

    header = [
        f'"""Auto-generated from the Math Representation System.',
        f"Domain: {model.problem_domain}",
        '"""',
        "",
    ]
    if any("math." in sp.pycode(f.expr) for f in functions):
        header.append("import math")
        header.append("")
    header.append("")

    body_blocks = []
    for f in functions:
        body_blocks.append(generate_python_function(f, variable_meanings, unit_by_symbol.get(f.target_name),
                                                       include_import=False))

    demo_lines = ['if __name__ == "__main__":']
    any_demo = False
    for f in functions:
        if all(a in known_values for a in f.arg_names):
            call_args = ", ".join(f"{a}={known_values[a]!r}" for a in f.arg_names)
            demo_lines.append(f"    print({f.target_name}({call_args}))")
            any_demo = True
    if not any_demo:
        demo_lines.append("    pass  # fill in real values for the arguments above and call the function(s)")

    parts = header + ["\n\n".join(body_blocks), "", "\n".join(demo_lines), ""]
    return "\n".join(parts)
