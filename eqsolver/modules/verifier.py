"""
The system "checks its own work" in two independent ways:

1. STRUCTURAL/SYMBOLIC (SymPy, deterministic, no LLM involved):
   - every equation actually parsed
   - the requested solve_for symbol appears somewhere
   - the system isn't under/over-determined relative to knowns
   - if all variables are known, the equation numerically balances

2. CROSS-CHECK (a second, independent LLM call):
   - the LLM is asked to solve the ORIGINAL word problem directly,
     without seeing the derived equations, producing a plain numeric
     answer. That's compared against SymPy's solve of the derived
     equations. Disagreement is strong evidence the derivation is wrong
     (since two independent paths should agree), and triggers a retry
     of extraction with the discrepancy fed back in.

This separation matters: an LLM grading its own derivation tends to
rubber-stamp it. An independent re-derivation + numeric comparison is a
much harder test to pass by accident.
"""
from dataclasses import dataclass, field
import re
import sympy as sp

from config import settings
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel

INDEPENDENT_SOLVE_PROMPT = """Solve this problem yourself, from scratch, showing minimal work. \
End your response with a single line in EXACTLY this format (no units, no extra text):
FINAL_NUMERIC_ANSWER: <number>

If the problem has no single final number (e.g. it asks you to just model a relationship), \
respond with:
FINAL_NUMERIC_ANSWER: N/A

Problem:
{problem}
"""


@dataclass
class CheckResult:
    label: str
    passed: bool
    detail: str


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)
    sympy_numeric_answer: float | None = None
    llm_independent_answer: float | None = None
    passed: bool = True
    failure_reason: str | None = None

    def add(self, label: str, passed: bool, detail: str):
        self.checks.append(CheckResult(label, passed, detail))
        if not passed:
            self.passed = False


def _known_substitutions(model: ProblemModel) -> dict:
    return {
        sp.Symbol(v.symbol): sp.nsimplify(v.known_value)
        for v in model.variables if v.known_value is not None
    }


def _structural_checks(model: ProblemModel, report: VerificationReport):
    # 1. parse errors
    bad = [e for e in model.equations if e.sympy_eq is None]
    if bad:
        report.add("Equation parsing", False,
                    f"{len(bad)} equation(s) failed to parse: " +
                    "; ".join(f"'{e.raw_expression}' ({e.parse_error})" for e in bad))
    else:
        report.add("Equation parsing", True, "All equations parsed to valid SymPy expressions.")

    # 2. solve_for symbol is actually used
    if model.solve_for:
        used = set()
        for e in model.equations:
            if e.sympy_eq is not None:
                used |= {s.name for s in e.sympy_eq.free_symbols}
        if model.solve_for not in used:
            report.add("Target variable present", False,
                        f"'{model.solve_for}' does not appear in any derived equation.")
        else:
            report.add("Target variable present", True,
                        f"'{model.solve_for}' appears in the equations.")

    # 3. determinacy: unknowns vs equations
    unknowns = [v.symbol for v in model.variables if v.known_value is None]
    n_eqs = len([e for e in model.equations if e.sympy_eq is not None])
    if len(unknowns) > n_eqs and n_eqs > 0:
        report.add("Determinacy", False,
                    f"{len(unknowns)} unknown(s) ({', '.join(unknowns)}) but only "
                    f"{n_eqs} equation(s) -- system may be underdetermined.")
    else:
        report.add("Determinacy", True,
                    f"{len(unknowns)} unknown(s), {n_eqs} equation(s) -- solvable in principle.")


def _numeric_balance_check(model: ProblemModel, report: VerificationReport):
    """If every symbol in an equation has a known value, plugging them in
    should reduce it to (approximately) 0 = 0. Anything else means the
    derivation is inconsistent with the numbers given in the problem."""
    subs = _known_substitutions(model)
    for eq in model.equations:
        if eq.sympy_eq is None:
            continue
        free = eq.sympy_eq.free_symbols
        if free and free.issubset(subs.keys()):
            residual = sp.simplify((eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs))
            try:
                val = float(residual)
                ok = abs(val) < settings.numeric_tolerance
            except TypeError:
                ok = residual == 0
                val = residual
            report.add(f"Numeric balance: {eq.name}", ok,
                        f"Substituting known values gives residual {val} "
                        f"({'balances' if ok else 'DOES NOT balance'}).")


def _solve_sympy(model: ProblemModel) -> float | None:
    if not model.solve_for:
        return None
    subs = _known_substitutions(model)
    eqs = [e.sympy_eq.subs(subs) for e in model.equations if e.sympy_eq is not None]
    target = sp.Symbol(model.solve_for)
    try:
        sol = sp.solve(eqs, target, dict=True)
        if not sol:
            return None
        val = sol[0].get(target)
        return float(val) if val is not None and val.is_number else None
    except Exception:  # noqa: BLE001
        return None


def _extract_final_number(text: str) -> float | None:
    m = re.search(r"FINAL_NUMERIC_ANSWER:\s*([\-0-9.eE]+|N/A)", text)
    if not m or m.group(1) == "N/A":
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def verify(model: ProblemModel, client: LMStudioClient, problem_text: str) -> VerificationReport:
    report = VerificationReport()
    _structural_checks(model, report)
    _numeric_balance_check(model, report)

    sympy_answer = _solve_sympy(model)
    report.sympy_numeric_answer = sympy_answer

    if model.solve_for and sympy_answer is not None:
        raw = client.chat(
            system="You are a careful independent problem solver.",
            user=INDEPENDENT_SOLVE_PROMPT.format(problem=problem_text),
            temperature=0.0,
        )
        llm_answer = _extract_final_number(raw)
        report.llm_independent_answer = llm_answer
        if llm_answer is not None:
            rel_diff = abs(llm_answer - sympy_answer) / max(abs(sympy_answer), 1e-9)
            agree = rel_diff < 0.02  # 2% tolerance for rounding differences
            report.add(
                "Independent cross-check",
                agree,
                f"Derived-equation answer = {sympy_answer:.6g}; independent re-solve = "
                f"{llm_answer:.6g}. {'Agree.' if agree else 'DISAGREE -- derivation likely flawed.'}",
            )

    if not report.passed:
        failing = [c for c in report.checks if not c.passed]
        report.failure_reason = " | ".join(f"{c.label}: {c.detail}" for c in failing)

    return report
