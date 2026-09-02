"""
Jupyter notebook export: bundles a solved problem's step-by-step
narrative (as markdown cells) with its runnable Python formula(s) (as
executable code cells, reusing code_export.py's exact same
sp.pycode-rendered functions) into a single .ipynb file -- a more
natural deliverable than a bare .py script for anyone doing further
work in a notebook environment.

Built by hand-constructing the nbformat v4 JSON structure directly
rather than adding a dependency on the `nbformat` package: the schema
needed here (a flat list of markdown/code cells, no stored outputs, no
execution metadata) is small and stable enough that a new dependency
for it isn't worth it. Kept honest about that scope: this produces a
notebook JUPYTER CAN OPEN AND RUN, not a notebook that's been executed
and has real outputs baked in.
"""
import json

from modules.equation_engine import ProblemModel
from modules.solver import SolutionStep
from modules.code_export import formula_for_target, generate_python_function


def _markdown_cell(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def _code_cell(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
             "source": text.splitlines(keepends=True)}


def build_notebook(problem_text: str, model: ProblemModel,
                    steps_by_target: dict[str, list[SolutionStep]]) -> str:
    """Returns the .ipynb file content as a JSON string. One markdown
    cell of narrative + one code cell (the runnable function) + one
    demo call cell per target that has BOTH a closed-form formula
    (see code_export.formula_for_target -- algebraic/ODE/recurrence
    only, same scope as the plain .py export) AND every argument
    available as a known value to actually call it with."""
    cells = [_markdown_cell(f"# {model.problem_domain}\n\n**Problem:**\n\n{problem_text.strip()}\n")]

    variable_meanings = {v.symbol: v.meaning for v in model.variables}
    unit_by_symbol = {v.symbol: v.unit for v in model.variables}
    known_values = {v.symbol: v.known_value for v in model.variables if v.known_value is not None}

    for target_name, steps in steps_by_target.items():
        cells.append(_markdown_cell(f"## Solving for `{target_name}`\n"))
        if steps:
            step_lines = "\n".join(f"- **{s.description}**: {s.expression}" for s in steps)
            cells.append(_markdown_cell(step_lines + "\n"))

        formula = formula_for_target(model, target_name)
        if formula is None:
            continue
        src = generate_python_function(formula, variable_meanings, unit_by_symbol.get(target_name))
        cells.append(_code_cell(src))

        if all(a in known_values for a in formula.arg_names):
            call_args = ", ".join(f"{a}={known_values[a]!r}" for a in formula.arg_names)
            cells.append(_code_cell(f"{target_name}({call_args})"))

    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    return json.dumps(notebook, indent=1)
