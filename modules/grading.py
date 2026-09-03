"""
"Grade my work": compares a student's own step-by-step attempt against
the system's independently-verified derivation for a target, using the
SAME equivalence-checking machinery as equivalence.py -- deliberately
NOT a literal line-by-line diff against the system's own derivation,
since two valid derivations of the same formula can look completely
different (isolating via a different variable, multiplying through
differently) and a literal diff would flag correct-but-differently-
ordered work as wrong.

Three-part diagnosis, since "wrong answer" alone doesn't say WHERE a
mistake happened:

1. Formula check: is the student's first written equation mathematically
   equivalent to one of the problem's own verified relations for this
   target? Checked by solving both for the target symbol and comparing
   the resulting expressions (not the raw equations) -- so a correctly
   rearranged equation (e.g. "a*t = v_f - v_i" instead of
   "a = (v_f - v_i)/t") is recognized as the same formula. A mismatch
   here means a conceptual/setup error, independent of whatever
   arithmetic follows.
2. Arithmetic check: for each subsequent equation-shaped line, does
   substituting the problem's own known values make both sides
   numerically equal? A failure here, even with a correct starting
   formula, means an arithmetic or substitution slip.
3. Final-answer check: does the student's last line's value match the
   system's own verified numeric answer within a normal tolerance?

Lines that don't fully substitute to numbers (e.g. they introduce an
intermediate variable of the student's own that isn't one of the
problem's declared variables) are marked as "not checkable" rather than
wrong -- there's nothing to fault a step for if there's no way to
evaluate it.
"""
from dataclasses import dataclass, field
import re
import sympy as sp

from modules.equation_engine import ProblemModel, TRANSFORMS, _local_dict, target_kind
from modules.equivalence import check_equivalence_exprs, EquivalenceResult
from modules.verifier import _known_substitutions
from sympy.parsing.sympy_parser import parse_expr

_REL_TOLERANCE = 0.01  # 1% -- matches the general "close enough" bar used elsewhere in the app


@dataclass
class LineResult:
    raw: str
    kind: str                      # "equation" | "expression" | "unparsed"
    arithmetic_ok: bool | None     # None = not checkable (e.g. involves an undeclared symbol)
    detail: str


@dataclass
class GradingResult:
    target: str
    formula_ok: bool | None        # None = undetermined (no equation line to check, or ambiguous)
    formula_detail: str
    line_results: list[LineResult] = field(default_factory=list)
    student_final_value: float | None = None
    correct_value: float | None = None
    final_answer_ok: bool | None = None
    summary: str = ""
    error: str | None = None


def _parse_line(line: str, local_dict: dict) -> sp.Basic | None:
    line = line.strip()
    if not line:
        return None
    if "=" in line and not any(op in line for op in ("==", "<=", ">=", "!=")):
        lhs_str, rhs_str = line.split("=", 1)
        expr_str = f"Eq({lhs_str.strip()}, {rhs_str.strip()})"
    else:
        expr_str = line
    try:
        return parse_expr(expr_str, local_dict=local_dict, transformations=TRANSFORMS)
    except Exception:  # noqa: BLE001
        return None


def _solve_for(eq: sp.Eq, symbol: sp.Symbol) -> sp.Expr | None:
    try:
        sols = sp.solve(eq, symbol)
        return sols[0] if sols else None
    except Exception:  # noqa: BLE001
        return None


def grade_work(model: ProblemModel, target_name: str, student_lines: list[str],
                correct_value: float | None) -> GradingResult:
    """`student_lines` is the student's own work, one attempted step per
    line, in ordinary "lhs = rhs" notation (or a bare final expression
    for the last line). `correct_value` is the system's own verified
    numeric answer for target_name (pass None if unavailable -- the
    formula/arithmetic checks still run, just not the final-answer check)."""
    if target_kind(model, target_name) != "equation":
        return GradingResult(target_name, None, "",
                              error="Grading is currently only supported for algebraic targets.")

    target = sp.Symbol(target_name)
    local_dict = _local_dict(model.variables)
    knowns = _known_substitutions(model)

    parsed_lines: list[tuple[str, sp.Basic | None]] = [
        (line, _parse_line(line, local_dict)) for line in student_lines if line.strip()
    ]
    if not parsed_lines:
        return GradingResult(target_name, None, "", error="No work was entered to grade.")

    # ---- 1. formula check: does the FIRST equation-shaped line match
    # (after solving both for the target) any of the model's own
    # verified equations for this target?
    formula_ok, formula_detail = None, "No equation line found to check against the verified formula."
    first_eq = next((p for _, p in parsed_lines if isinstance(p, sp.Eq)), None)
    if first_eq is not None:
        student_solved = _solve_for(first_eq, target)
        if student_solved is None:
            formula_detail = "Couldn't isolate the target from your first equation to compare it."
        else:
            candidates = [e for e in model.equations if e.kind == "equation" and e.sympy_eq is not None
                          and target in e.sympy_eq.free_symbols]
            best: EquivalenceResult | None = None
            for cand in candidates:
                model_solved = _solve_for(cand.sympy_eq, target)
                if model_solved is None:
                    continue
                result = check_equivalence_exprs(student_solved, model_solved)
                if result.equivalent is True:
                    best = result
                    break
                if best is None or (best.equivalent is False and result.equivalent is None):
                    best = result
            if best is not None:
                formula_ok = best.equivalent
                formula_detail = best.detail
            else:
                formula_detail = "No verified equation for this target could be compared against."

    # ---- 2. arithmetic check: for every equation-shaped line, do both
    # sides agree numerically once the problem's known values are subbed in?
    # A line whose LHS is literally the bare target symbol (e.g. "a = ...")
    # is a DEFINITION, not an independent claim to verify against anything
    # -- so it only needs its RHS to evaluate cleanly to a number, not for
    # both sides to already be numeric (the target itself is, by
    # definition, still unknown at that point in the derivation).
    line_results: list[LineResult] = []
    for raw, parsed in parsed_lines:
        if parsed is None:
            line_results.append(LineResult(raw, "unparsed", None, "Couldn't parse this line."))
            continue

        # sp.Eq() of two pure numeric literals auto-evaluates straight to
        # a BooleanTrue/BooleanFalse rather than staying an Eq object
        # (e.g. Eq(12/6, 3) -> False) -- that IS the arithmetic check
        # result directly, so handle it before the isinstance(Eq) branch.
        if parsed is sp.true or parsed is sp.false:
            ok = bool(parsed)
            line_results.append(LineResult(
                raw, "equation", ok,
                "Both sides agree numerically." if ok else "Both sides do NOT agree.",
            ))
            continue

        if not isinstance(parsed, sp.Eq):
            line_results.append(LineResult(raw, "expression", None, "Not an equation -- not checked."))
            continue

        lhs_is_bare_target = (parsed.lhs == target)
        rhs_sub = parsed.rhs.subs(knowns)
        if rhs_sub.free_symbols:
            line_results.append(LineResult(
                raw, "equation", None,
                "Involves a symbol not in the problem's known values -- not checkable.",
            ))
            continue
        try:
            rhs_val = float(rhs_sub)
        except (TypeError, ValueError):
            line_results.append(LineResult(raw, "equation", None, "Couldn't evaluate the right-hand side."))
            continue

        if lhs_is_bare_target:
            line_results.append(LineResult(
                raw, "equation", True,
                f"Right-hand side evaluates cleanly to {rhs_val:.6g}.",
            ))
            continue

        lhs_sub = parsed.lhs.subs(knowns)
        if lhs_sub.free_symbols:
            line_results.append(LineResult(
                raw, "equation", None,
                "Involves a symbol not in the problem's known values -- not checkable.",
            ))
            continue
        try:
            lhs_val = float(lhs_sub)
        except (TypeError, ValueError):
            line_results.append(LineResult(raw, "equation", None, "Couldn't evaluate the left-hand side."))
            continue

        ok = abs(lhs_val - rhs_val) <= max(abs(lhs_val), abs(rhs_val), 1e-9) * _REL_TOLERANCE
        detail = ("Both sides agree numerically." if ok else
                   f"Both sides do NOT agree: {lhs_val:.6g} vs {rhs_val:.6g}.")
        line_results.append(LineResult(raw, "equation", ok, detail))

    # ---- 3. final-answer check
    last_raw, last_parsed = parsed_lines[-1]
    student_final = None
    if isinstance(last_parsed, sp.Eq):
        candidate = last_parsed.rhs.subs(knowns)
        if not candidate.free_symbols:
            try:
                student_final = float(candidate)
            except (TypeError, ValueError):
                pass
    elif last_parsed is not None:
        candidate = last_parsed.subs(knowns)
        if not candidate.free_symbols:
            try:
                student_final = float(candidate)
            except (TypeError, ValueError):
                pass

    final_answer_ok = None
    if student_final is not None and correct_value is not None:
        tol = max(abs(correct_value), 1e-9) * _REL_TOLERANCE
        final_answer_ok = abs(student_final - correct_value) <= tol

    # ---- overall summary, distinguishing failure modes
    if final_answer_ok is True:
        summary = "✅ Correct final answer."
        if formula_ok is False:
            summary += " (Your starting formula didn't match a verified relation, but you still landed on the right number -- double check this wasn't a coincidence.)"
    elif final_answer_ok is False:
        if formula_ok is False:
            summary = "❌ Incorrect -- the starting formula/setup doesn't match a verified relation for this target. This looks like a conceptual error, not an arithmetic slip."
        elif any(lr.arithmetic_ok is False for lr in line_results):
            bad_lines = [i + 1 for i, lr in enumerate(line_results) if lr.arithmetic_ok is False]
            summary = f"❌ Incorrect -- your formula was right, but line {bad_lines[0]} has an arithmetic/substitution error."
        else:
            summary = "❌ Incorrect -- the formula and shown arithmetic check out individually, but the final value doesn't match. Check rounding, or a step may be missing."
    else:
        summary = "⚠️ Couldn't fully verify the final answer (missing a comparison value or the last line isn't a checkable number)."

    return GradingResult(
        target=target_name, formula_ok=formula_ok, formula_detail=formula_detail,
        line_results=line_results, student_final_value=student_final,
        correct_value=correct_value, final_answer_ok=final_answer_ok, summary=summary,
    )


# ---------------------------------------------------------------- error classification
#
# Feeds a learning loop: history.py persists this classification per
# submission, history.summarize_error_patterns() looks for a category/
# subtype that keeps recurring, and worksheet.py can bias new practice
# problems toward whatever's actually going wrong -- rather than every
# graded submission being a one-off with no memory of past mistakes.

_ARITHMETIC_LINE_RE = re.compile(r"(-?[\d.]+)\s+vs\s+(-?[\d.]+)")


@dataclass
class ErrorClassification:
    category: str          # "correct" | "formula" | "arithmetic" | "unverified"
    subtype: str | None    # only set for category=="arithmetic": "sign_error" |
                            # "addition" | "subtraction" | "multiplication" | "division" | None
    detail: str


def _arithmetic_subtype(line_result: LineResult) -> str | None:
    """Best-effort guess at WHAT KIND of arithmetic slip a failing line
    represents. LineResult only carries the raw text and a rendered
    detail string by this point (not the parsed expression), so this is
    a light heuristic, not a proof: if the two sides its own detail
    reports are equal in magnitude but opposite in sign, that's very
    likely a sign error; otherwise falls back to whichever operator
    appears in the raw line, checked in an order that favors the
    operator most likely to be the actual point of failure (a line with
    both + and - is far more often a subtraction slip than an addition
    one, since "+" is usually just holding two already-correct terms
    together)."""
    m = _ARITHMETIC_LINE_RE.search(line_result.detail)
    if m:
        a, b = float(m.group(1)), float(m.group(2))
        if abs(a + b) < max(abs(a), abs(b), 1e-9) * 0.01:
            return "sign_error"
    raw = line_result.raw
    if "-" in raw:
        return "subtraction"
    if "/" in raw:
        return "division"
    if "*" in raw:
        return "multiplication"
    if "+" in raw:
        return "addition"
    return None


def classify_mistake(result: GradingResult) -> ErrorClassification:
    """Turns a GradingResult's three-part diagnosis into a single,
    compact classification suitable for persisting and counting over
    time -- see history.record_grading / summarize_error_patterns."""
    if result.error:
        return ErrorClassification("unverified", None, result.error)
    if result.final_answer_ok is True:
        return ErrorClassification("correct", None, "Correct final answer.")
    if result.formula_ok is False:
        return ErrorClassification("formula", None, result.formula_detail)
    bad_line = next((lr for lr in result.line_results if lr.arithmetic_ok is False), None)
    if bad_line is not None:
        return ErrorClassification("arithmetic", _arithmetic_subtype(bad_line), bad_line.detail)
    # formula checked out, no line was flagged wrong, but the final
    # answer still didn't match (or couldn't be compared) -- not
    # confidently attributable to either category
    return ErrorClassification("unverified", None, result.summary)
