"""
The same result in the forms people paste it in -- plain text, with units, LaTeX, Python -- Streamlit-free.

Each target's answer, and each Monte Carlo summary, gets a "Copy" popover offering these; the UI shows the
chosen form in a code block, whose built-in copy icon does the copying (Streamlit has no other clipboard
access). Everything is derived from numbers the app has already verified; nothing is recomputed here.

Precision: the plain-text and LaTeX forms use six significant figures, like the rest of the app. The Python
form uses the value's full precision, because someone pasting a number into code wants the number, not a
rounded display of it.
"""
from modules.code_export import formula_for_target, generate_python_function
from modules.equation_engine import ProblemModel
from modules.share_text import _number_tex, _unit_tex
from modules.verifier import VerificationReport

import sympy as sp


def _unit(model: ProblemModel, target: str) -> str | None:
    return next((v.unit for v in model.variables if v.symbol == target), None)


def _clean_unit(unit: str | None) -> str:
    return "" if not unit or unit.strip().lower() in ("unitless", "dimensionless", "none", "1") else unit.strip()


def copy_forms(model: ProblemModel, report: VerificationReport, target: str) -> dict[str, tuple[str, str | None]]:
    """{form name: (text, code-block language)} for `target`'s verified numeric answer; empty if it has none."""
    value = report.sympy_numeric_answers.get(target)
    if value is None:
        return {}
    unit = _clean_unit(_unit(model, target))
    forms: dict[str, tuple[str, str | None]] = {
        "Plain text": (f"{target} = {value:.6g}", None),
        "With units": (f"{target} = {value:.6g}" + (f" {unit}" if unit else ""), None),
        "LaTeX": (f"{sp.latex(sp.Symbol(target))} = {_number_tex(value)}{_unit_tex(_unit(model, target))}", "latex"),
        "Python value": (f"{target} = {float(value)!r}" + (f"  # {unit}" if unit else ""), "python"),
    }
    formula = formula_for_target(model, target)
    if formula is not None:
        forms["Python function"] = (generate_python_function(
            formula, {v.symbol: v.meaning for v in model.variables}, _unit(model, target)), "python")
    return forms


def monte_carlo_copy_forms(target: str, mean: float, std: float, p5: float | None, p95: float | None,
                           n: int, seed: int, unit: str | None) -> dict[str, tuple[str, str | None]]:
    """The summary of a Monte Carlo run, with enough about the run (samples, seed) to reproduce it."""
    u = _clean_unit(unit)
    band = f"; 5th-95th percentile {p5:.6g} to {p95:.6g}" if p5 is not None and p95 is not None else ""
    run = f" ({n:,} samples, seed {seed})"
    return {
        "Plain text": (f"{target} = {mean:.6g} ± {std:.6g}" + (f" {u}" if u else "") + band + run, None),
        "LaTeX": (f"{sp.latex(sp.Symbol(target))} = {_number_tex(mean)} \\pm {_number_tex(std)}{_unit_tex(unit)}", "latex"),
        "Python value": (f"{target}_mean, {target}_std = {mean!r}, {std!r}" + (f"  # {u}" if u else "")
                         + f"  # Monte Carlo, n={n}, seed={seed}", "python"),
    }
