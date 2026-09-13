"""
Guided/tutor mode: turns a solved problem's step-by-step derivation
(already computed by solver.py) into an interactive predict-then-reveal
exercise, rather than a fixed wall of text -- and grades a numeric
final-answer guess against the app's own rigorous verification
(VerificationReport.sympy_numeric_answers), so "am I right" gets a real
answer, not a vague self-assessment.

Deliberately does NOT attempt to grade intermediate algebraic steps
against a free-form symbolic guess: SolutionStep only stores each
step's expression as an already-rendered LaTeX string (see solver.py),
not the underlying sympy object, and reliably reconstructing a sympy
expression by parsing arbitrary LaTeX back out is a much harder and
more failure-prone problem than it looks -- exactly the kind of
brittle "looks like it works in the demo" feature this codebase's
verification-first philosophy exists to avoid. Progressive step reveal
(one click at a time, hint-before-full-reveal) still gets most of the
pedagogical value without that risk; only the FINAL numeric answer,
which the verification pipeline already computes and trusts, is
actually graded here.
"""
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

from modules.verifier import VerificationReport

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class AnswerCheckResult:
    correct: bool
    guess_value: float | None
    actual_value: float | None
    relative_error: float | None
    detail: str
    error: str | None = None


def check_final_answer_guess(guess_str: str, target_name: str, report: VerificationReport,
                              tolerance: float = 0.02) -> AnswerCheckResult:
    """Grades a person's numeric (or simple-expression, e.g. "3*4")
    guess for `target_name` against the verified numeric answer already
    computed for this problem. `tolerance` is a RELATIVE tolerance
    (0.02 = within 2%), not absolute -- appropriate here since answers
    span wildly different magnitudes across problems (3 vs 3,000,000),
    and an absolute tolerance that's sensible for one is meaningless
    for the other."""
    if target_name not in report.sympy_numeric_answers:
        return AnswerCheckResult(correct=False, guess_value=None, actual_value=None,
                                   relative_error=None, detail="",
                                   error=f"No verified numeric answer available for {target_name!r} "
                                         "to check against.")
    actual = report.sympy_numeric_answers[target_name]

    try:
        parsed = parse_expr(guess_str, transformations=_TRANSFORMS)
        guess_value = float(parsed)
    except Exception as exc:  # noqa: BLE001
        return AnswerCheckResult(correct=False, guess_value=None, actual_value=actual,
                                   relative_error=None, detail="",
                                   error=f"Could not read that as a number: {exc}")

    denom = max(abs(actual), 1e-9)
    relative_error = abs(guess_value - actual) / denom
    correct = relative_error < tolerance

    if correct:
        detail = f"Correct! {target_name} = {actual:g} (your guess: {guess_value:g})."
    else:
        detail = (f"Not quite -- you guessed {guess_value:g}, the actual value is {actual:g} "
                  f"(off by {relative_error:.1%}). Look through the steps below to see where "
                  f"the reasoning goes.")

    return AnswerCheckResult(correct=correct, guess_value=guess_value, actual_value=actual,
                               relative_error=relative_error, detail=detail)
