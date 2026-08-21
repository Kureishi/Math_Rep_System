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
The problem asks for these quantities: {targets}. \
End your response with one line per quantity, EXACTLY in this format (no units, no extra text), \
one per line:
FINAL_NUMERIC_ANSWER[<symbol>]: <number>

If a quantity can't be reduced to a single number, use N/A for its value.

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
    sympy_numeric_answers: dict[str, float] = field(default_factory=dict)
    llm_independent_answers: dict[str, float] = field(default_factory=dict)
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

    # 2. every solve_for symbol is actually used somewhere
    if model.solve_for:
        used = set()
        for e in model.equations:
            if e.sympy_eq is not None:
                used |= {s.name for s in e.sympy_eq.free_symbols}
        missing = [t for t in model.solve_for if t not in used]
        if missing:
            report.add("Target variable present", False,
                        f"{', '.join(missing)} does not appear in any derived equation.")
        else:
            report.add("Target variable present", True,
                        f"{', '.join(model.solve_for)} all appear in the equations.")

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


def _solve_sympy(model: ProblemModel) -> dict[str, float]:
    """Solves the whole equation system at once for every requested target --
    important because targets can be coupled (e.g. displacement 'd' may
    depend on an also-unknown acceleration 'a' solved from another equation
    in the same system)."""
    if not model.solve_for:
        return {}
    subs = _known_substitutions(model)
    eqs = [e.sympy_eq.subs(subs) for e in model.equations if e.sympy_eq is not None]
    targets = [sp.Symbol(t) for t in model.solve_for]
    try:
        sol = sp.solve(eqs, targets, dict=True)
        if not sol:
            return {}
        result = {}
        for t, sym in zip(model.solve_for, targets):
            val = sol[0].get(sym)
            if val is not None and val.is_number:
                result[t] = float(val)
        return result
    except Exception:  # noqa: BLE001
        return {}


def _extract_final_numbers(text: str) -> dict[str, float]:
    """Parses one or more `FINAL_NUMERIC_ANSWER[symbol]: value` lines."""
    results = {}
    for m in re.finditer(r"FINAL_NUMERIC_ANSWER\[(\w+)\]:\s*([\-0-9.eE]+|N/A)", text):
        symbol, value = m.group(1), m.group(2)
        if value != "N/A":
            try:
                results[symbol] = float(value)
            except ValueError:
                pass
    return results


def verify(model: ProblemModel, client: LMStudioClient, problem_text: str) -> VerificationReport:
    report = VerificationReport()
    _structural_checks(model, report)
    _numeric_balance_check(model, report)

    sympy_answers = _solve_sympy(model)
    report.sympy_numeric_answers = sympy_answers

    if model.solve_for and sympy_answers:
        raw = client.chat(
            system="You are a careful independent problem solver.",
            user=INDEPENDENT_SOLVE_PROMPT.format(
                targets=", ".join(model.solve_for), problem=problem_text),
            temperature=0.0,
        )
        llm_answers = _extract_final_numbers(raw)
        report.llm_independent_answers = llm_answers

        for target in model.solve_for:
            sympy_val = sympy_answers.get(target)
            llm_val = llm_answers.get(target)
            if sympy_val is None or llm_val is None:
                continue
            rel_diff = abs(llm_val - sympy_val) / max(abs(sympy_val), 1e-9)
            agree = rel_diff < 0.02  # 2% tolerance for rounding differences
            report.add(
                f"Independent cross-check: {target}",
                agree,
                f"Derived-equation answer = {sympy_val:.6g}; independent re-solve = "
                f"{llm_val:.6g}. {'Agree.' if agree else 'DISAGREE -- derivation likely flawed.'}",
            )

    if not report.passed:
        failing = [c for c in report.checks if not c.passed]
        report.failure_reason = " | ".join(f"{c.label}: {c.detail}" for c in failing)

    return report
