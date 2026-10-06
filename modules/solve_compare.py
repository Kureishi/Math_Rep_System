"""
Compare two solved problems side by side -- Streamlit-free, and no LLM calls.

The two things being compared are `SolveSnapshot`s: a problem text, its ProblemModel, its
VerificationReport and its steps. They come from the history database (`snapshot_from_history`), from the
solve on screen, or from a what-if variant of either (`what_if_snapshot`): the same equations with some
known inputs changed, re-solved by SymPy. A what-if is verified with the DETERMINISTIC checks only
(structure, numeric balance, units, domain, the ODE/recurrence/optimization checks); the independent
LLM re-solve is not repeated, and a snapshot made that way says so (`deterministic_only`), because
presenting a variant as cross-checked by a second model when it was not would overstate it.

`compare()` lines the two up on everything a reader would want to diff:

  answers     per target: both values, absolute and relative difference, same / changed / only in one
  variables   per symbol: known value and unit, and whether it changed role (given vs solved)
  equations   paired as identical, mathematically equivalent (rearranged or rescaled -- decided by
              SymPy, not by string comparison), changed, or present on one side only
  confidence  the overall score and label, each category's passed/total, and each individual check

Nothing here ranks the two solves or says which is "right" -- it reports what differs, and the summary
lines are plain statements of those differences.
"""
import math
from collections.abc import Iterable
from dataclasses import dataclass, field, replace
from typing import Any

import sympy as sp

from modules.equation_engine import Equation, ProblemModel
from modules.solver import SolutionStep
from modules.timeout_utils import ComputationTimeoutError, run_with_timeout
from modules.verifier import VerificationReport

# answers closer than this (relative to their size, with a floor of 1) count as the same number
SAME_TOLERANCE = 1e-9


@dataclass
class SolveSnapshot:
    label: str
    problem_text: str
    model: ProblemModel
    report: VerificationReport
    steps: dict[str, list[SolutionStep]] = field(default_factory=dict)
    deterministic_only: bool = False      # True for a what-if variant -- see the module docstring


@dataclass
class AnswerDiff:
    target: str
    a: float | None
    b: float | None
    unit: str | None
    abs_diff: float | None
    rel_diff: float | None                # |a - b| / max(|a|, |b|), None when either side is missing or both 0
    status: str                           # "same" | "changed" | "only_a" | "only_b"


@dataclass
class VariableDiff:
    symbol: str
    meaning: str
    a_value: float | None
    b_value: float | None
    a_unit: str | None
    b_unit: str | None
    status: str                           # "same" | "changed" | "role_changed" | "only_a" | "only_b"


@dataclass
class EquationDiff:
    a_name: str | None
    b_name: str | None
    a_latex: str | None
    b_latex: str | None
    status: str                           # "identical" | "equivalent" | "changed" | "only_a" | "only_b"


@dataclass
class CategoryDiff:
    category: str
    a_passed: int | None
    a_total: int | None
    b_passed: int | None
    b_total: int | None


@dataclass
class CheckDiff:
    label: str
    a_passed: bool | None                 # None = this check did not run on that side
    b_passed: bool | None
    a_detail: str
    b_detail: str


@dataclass
class SolveComparison:
    a_label: str
    b_label: str
    a_domain: str
    b_domain: str
    a_score: float
    b_score: float
    a_confidence: str
    b_confidence: str
    a_passed: bool
    b_passed: bool
    answers: list[AnswerDiff]
    variables: list[VariableDiff]
    equations: list[EquationDiff]
    categories: list[CategoryDiff]
    checks: list[CheckDiff]
    summary: list[str]
    caveats: list[str] = field(default_factory=list)


# --------------------------------------------------------------------- building snapshots
def snapshot_from_history(entry_id: int, label: str | None = None) -> SolveSnapshot | None:
    """A saved problem as a SolveSnapshot (no LLM call), or None if that id doesn't exist."""
    from modules import history          # imported here: it opens a database module-level path lazily
    loaded = history.load(entry_id)
    if loaded is None:
        return None
    problem_text, model, report, steps, _scenarios = loaded
    first_line = problem_text.strip().splitlines()[0][:50] if problem_text.strip() else f"#{entry_id}"
    return SolveSnapshot(label=label or first_line, problem_text=problem_text, model=model,
                         report=report, steps=steps)


def what_if_snapshot(base: SolveSnapshot, overrides: dict[str, float], label: str | None = None) -> SolveSnapshot:
    """`base` with some KNOWN inputs changed, re-solved by SymPy and re-checked deterministically.

    Only a variable that already has a known value can be overridden; naming anything else raises
    ValueError rather than quietly turning an unknown into a given."""
    known = {v.symbol: v for v in base.model.variables if v.known_value is not None}
    bad = sorted(set(overrides) - set(known))
    if bad:
        raise ValueError(f"Can only change inputs that have a known value; not: {', '.join(bad)}.")
    new_vars = [replace(v, known_value=float(overrides[v.symbol])) if v.symbol in overrides else v
                for v in base.model.variables]
    model = replace(base.model, variables=new_vars)
    report = deterministic_report(model)
    changed = ", ".join(f"{k} = {overrides[k]:g}" for k in sorted(overrides))
    return SolveSnapshot(label=label or f"{base.label} (what if {changed})", problem_text=base.problem_text,
                         model=model, report=report, steps={}, deterministic_only=True)


def deterministic_report(model: ProblemModel) -> VerificationReport:
    """verifier.verify() minus its one LLM call: every check that is computed rather than asked of a model."""
    from modules import verifier as v
    from modules.plausibility import check_plausibility

    report = VerificationReport()
    for check in (v._structural_checks, v._numeric_balance_check, v._dimensional_checks,
                  v._inequality_checks, v._ode_checks, v._recurrence_checks, v._optimization_checks,
                  v._matrix_system_checks, v._domain_checks):
        check(model, report)
    try:
        answers = v._solve_sympy(model)
    except ComputationTimeoutError as e:
        report.add("Symbolic solve", False, str(e))
        answers = {}
    report.sympy_numeric_answers = answers
    known = {x.symbol: x.known_value for x in model.variables if x.known_value is not None}
    report.plausibility_notes = check_plausibility(model, {**known, **answers})
    if not report.passed:
        report.failure_reason = " | ".join(f"{c.label}: {c.detail}" for c in report.checks if not c.passed)
    return report


# --------------------------------------------------------------------- comparing
def _close(a: float, b: float) -> bool:
    return abs(a - b) <= SAME_TOLERANCE * max(1.0, abs(a), abs(b))


def _unit_of(model: ProblemModel, symbol: str) -> str | None:
    return next((v.unit for v in model.variables if v.symbol == symbol), None)


def compare_answers(a: SolveSnapshot, b: SolveSnapshot) -> list[AnswerDiff]:
    av, bv = a.report.sympy_numeric_answers, b.report.sympy_numeric_answers
    order = list(av) + [t for t in bv if t not in av]
    out: list[AnswerDiff] = []
    for t in order:
        x, y = av.get(t), bv.get(t)
        unit = _unit_of(a.model, t) or _unit_of(b.model, t)
        if x is None:
            out.append(AnswerDiff(t, None, y, unit, None, None, "only_b"))
        elif y is None:
            out.append(AnswerDiff(t, x, None, unit, None, None, "only_a"))
        else:
            diff = abs(x - y)
            scale = max(abs(x), abs(y))
            rel = diff / scale if scale > 0 else None
            out.append(AnswerDiff(t, x, y, unit, diff, rel, "same" if _close(x, y) else "changed"))
    return out


def compare_variables(a: SolveSnapshot, b: SolveSnapshot) -> list[VariableDiff]:
    av = {v.symbol: v for v in a.model.variables}
    bv = {v.symbol: v for v in b.model.variables}
    out: list[VariableDiff] = []
    for sym in list(av) + [s for s in bv if s not in av]:
        x, y = av.get(sym), bv.get(sym)
        if x is None and y is not None:
            out.append(VariableDiff(sym, y.meaning, None, y.known_value, None, y.unit, "only_b"))
        elif y is None and x is not None:
            out.append(VariableDiff(sym, x.meaning, x.known_value, None, x.unit, None, "only_a"))
        elif x is not None and y is not None:
            if (x.known_value is None) != (y.known_value is None):
                status = "role_changed"
            elif x.known_value is not None and y.known_value is not None and not _close(x.known_value, y.known_value):
                status = "changed"
            elif x.unit != y.unit:
                status = "changed"
            else:
                status = "same"
            out.append(VariableDiff(sym, x.meaning or y.meaning, x.known_value, y.known_value, x.unit, y.unit, status))
    return out


def _relation_expr(eq: Equation) -> sp.Expr | None:
    """lhs - rhs of an equation-kind relation, or None if it has no such form."""
    e = eq.sympy_eq
    if isinstance(e, sp.Equality):
        return e.lhs - e.rhs
    return None


def _norm(raw: str) -> str:
    return "".join(raw.split()).lower()


def _same_relation(x: Equation, y: Equation) -> bool:
    """Whether two equations say the same thing: equal up to rearrangement, sign, or a constant factor.
    A relation that cannot be decided in time is reported as different rather than guessed at."""
    dx, dy = _relation_expr(x), _relation_expr(y)
    if dx is None or dy is None:
        return False
    try:
        if sp.expand(dx - dy) == 0 or sp.expand(dx + dy) == 0:
            return True
        ratio = run_with_timeout(lambda: sp.simplify(dx / dy), label="equation comparison")
    except ComputationTimeoutError:
        return False
    except Exception:  # noqa: BLE001 -- anything SymPy cannot do with these is "not shown equivalent"
        return False
    return bool(ratio.is_number and ratio.is_finite and ratio != 0)


def _symbols_of(eq: Equation) -> set[str]:
    e = eq.sympy_eq
    return {str(s) for s in e.free_symbols} if e is not None else set()


def _latex(eq: Equation) -> str:
    return sp.latex(eq.sympy_eq) if eq.sympy_eq is not None else eq.raw_expression


def compare_equations(a: SolveSnapshot, b: SolveSnapshot) -> list[EquationDiff]:
    left, right = list(a.model.equations), list(b.model.equations)
    out: list[EquationDiff] = []
    used_b: set[int] = set()
    remaining_a: list[Equation] = []

    # 1) identical text, then 2) mathematically equivalent
    for pass_name in ("identical", "equivalent"):
        still: list[Equation] = []
        for x in (left if pass_name == "identical" else remaining_a):
            match = None
            for j, y in enumerate(right):
                if j in used_b:
                    continue
                if (_norm(x.raw_expression) == _norm(y.raw_expression)) if pass_name == "identical" \
                        else _same_relation(x, y):
                    match = j
                    break
            if match is None:
                still.append(x)
            else:
                used_b.add(match)
                y = right[match]
                out.append(EquationDiff(x.name, y.name, _latex(x), _latex(y), pass_name))
        remaining_a = still

    # 3) what is left: pair by shared symbols (a changed equation), else report as one-sided
    leftover_b = [j for j in range(len(right)) if j not in used_b]
    for x in remaining_a:
        best, best_score = None, 0.0
        sx = _symbols_of(x)
        for j in leftover_b:
            sy = _symbols_of(right[j])
            union = sx | sy
            score = len(sx & sy) / len(union) if union else 0.0
            if score > best_score:
                best, best_score = j, score
        if best is not None and best_score >= 0.5:
            leftover_b.remove(best)
            y = right[best]
            out.append(EquationDiff(x.name, y.name, _latex(x), _latex(y), "changed"))
        else:
            out.append(EquationDiff(x.name, None, _latex(x), None, "only_a"))
    for j in leftover_b:
        y = right[j]
        out.append(EquationDiff(None, y.name, None, _latex(y), "only_b"))
    return out


def compare_confidence(a: SolveSnapshot, b: SolveSnapshot) -> tuple[list[CategoryDiff], list[CheckDiff]]:
    ca, cb = a.report.confidence_report(), b.report.confidence_report()
    cats: list[CategoryDiff] = []
    for name in list(ca.categories) + [c for c in cb.categories if c not in ca.categories]:
        x, y = ca.categories.get(name), cb.categories.get(name)
        cats.append(CategoryDiff(name, x.passed if x else None, x.total if x else None,
                                 y.passed if y else None, y.total if y else None))
    a_checks = {c.label: c for c in a.report.checks}
    b_checks = {c.label: c for c in b.report.checks}
    checks = [CheckDiff(lbl,
                        a_checks[lbl].passed if lbl in a_checks else None,
                        b_checks[lbl].passed if lbl in b_checks else None,
                        a_checks[lbl].detail if lbl in a_checks else "",
                        b_checks[lbl].detail if lbl in b_checks else "")
              for lbl in list(a_checks) + [c for c in b_checks if c not in a_checks]]
    return cats, checks


def _fmt(x: float | None) -> str:
    return "—" if x is None else f"{x:.6g}"


def compare(a: SolveSnapshot, b: SolveSnapshot) -> SolveComparison:
    answers = compare_answers(a, b)
    variables = compare_variables(a, b)
    equations = compare_equations(a, b)
    categories, checks = compare_confidence(a, b)
    ca, cb = a.report.confidence_report(), b.report.confidence_report()

    summary: list[str] = []
    changed = [d for d in answers if d.status == "changed"]
    one_sided = [d for d in answers if d.status in ("only_a", "only_b")]
    if answers and not changed and not one_sided:
        summary.append(f"All {len(answers)} answer(s) are the same.")
    for d in changed:
        pct = f" ({d.rel_diff:.2%} apart)" if d.rel_diff is not None else ""
        summary.append(f"{d.target}: {_fmt(d.a)} → {_fmt(d.b)}{pct}.")
    for d in one_sided:
        side = "A" if d.status == "only_a" else "B"
        summary.append(f"{d.target} is only solved in {side}.")
    changed_vars = [v for v in variables if v.status != "same"]
    if changed_vars:
        summary.append("Inputs that differ: " + ", ".join(v.symbol for v in changed_vars) + ".")
    n_eq_diff = sum(1 for e in equations if e.status in ("changed", "only_a", "only_b"))
    if n_eq_diff:
        summary.append(f"{n_eq_diff} equation(s) differ" +
                       (f"; {sum(1 for e in equations if e.status == 'equivalent')} more are the same "
                        "relation written differently." if any(e.status == "equivalent" for e in equations) else "."))
    elif equations and any(e.status == "equivalent" for e in equations):
        summary.append("The equations are mathematically the same, written differently in places.")
    if ca.passed != cb.passed:
        summary.append(f"Verification: A {'passed' if ca.passed else 'failed'}, B {'passed' if cb.passed else 'failed'}.")
    elif abs(ca.score - cb.score) >= 0.05:
        summary.append(f"Confidence: A {ca.score:.0%} ({ca.label}), B {cb.score:.0%} ({cb.label}).")
    if not summary:
        summary.append("Nothing differs between the two solves.")

    caveats = [f"{s.label!r} was checked with the deterministic checks only -- the independent LLM re-solve "
               "is not repeated for a what-if." for s in (a, b) if s.deterministic_only]
    return SolveComparison(
        a_label=a.label, b_label=b.label, a_domain=a.model.problem_domain, b_domain=b.model.problem_domain,
        a_score=ca.score, b_score=cb.score, a_confidence=ca.label, b_confidence=cb.label,
        a_passed=ca.passed, b_passed=cb.passed, answers=answers, variables=variables,
        equations=equations, categories=categories, checks=checks, summary=summary, caveats=caveats)


# --------------------------------------------------------------------- export
def comparison_markdown(c: SolveComparison) -> str:
    """The comparison as Markdown, for download."""
    L = [f"# Comparison: {c.a_label} vs {c.b_label}", ""]
    L += ["## Summary", ""] + [f"- {s}" for s in c.summary] + [""]
    L += [f"- A: {c.a_domain}, confidence {c.a_score:.0%} ({c.a_confidence})",
          f"- B: {c.b_domain}, confidence {c.b_score:.0%} ({c.b_confidence})", ""]
    if c.caveats:
        L += [f"> {t}" for t in c.caveats] + [""]
    L += ["## Answers", "", "| Target | A | B | Difference | Relative |", "|---|---|---|---|---|"]
    for d in c.answers:
        unit = f" {d.unit}" if d.unit else ""
        rel = f"{d.rel_diff:.3%}" if d.rel_diff is not None else "—"
        L.append(f"| {d.target} | {_fmt(d.a)}{unit} | {_fmt(d.b)}{unit} | {_fmt(d.abs_diff)} | {rel} |")
    L += ["", "## Variables", "", "| Symbol | A | B | Status |", "|---|---|---|---|"]
    for v in c.variables:
        L.append(f"| {v.symbol} | {_fmt(v.a_value)} {v.a_unit or ''} | {_fmt(v.b_value)} {v.b_unit or ''} | {v.status} |")
    L += ["", "## Equations", ""]
    for e in c.equations:
        L.append(f"- **{e.status}**: A `{e.a_name or '—'}`: ${e.a_latex or ''}$ — B `{e.b_name or '—'}`: ${e.b_latex or ''}$")
    L += ["", "## Verification", "", "| Check | A | B |", "|---|---|---|"]
    mark = {True: "pass", False: "FAIL", None: "—"}
    for k in c.checks:
        L.append(f"| {k.label} | {mark[k.a_passed]} | {mark[k.b_passed]} |")
    return "\n".join(L) + "\n"


def chart_rows(c: SolveComparison) -> dict[str, Any]:
    """The numbers the two comparison charts are drawn from (kept here so they are testable)."""
    rel = [(d.target, (d.b - d.a) / abs(d.a) * 100.0) for d in c.answers
           if d.a is not None and d.b is not None and d.a != 0
           and math.isfinite(d.a) and math.isfinite(d.b)]
    cats = [(k.category,
             (k.a_passed / k.a_total) if k.a_passed is not None and k.a_total else None,
             (k.b_passed / k.b_total) if k.b_passed is not None and k.b_total else None) for k in c.categories]
    return {"relative_change_pct": rel, "category_pass_fraction": cats}


def targets_in_common(a: SolveSnapshot, b: SolveSnapshot) -> Iterable[str]:
    return [t for t in a.report.sympy_numeric_answers if t in b.report.sympy_numeric_answers]
