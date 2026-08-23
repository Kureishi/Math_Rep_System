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
from sympy.core.function import AppliedUndef

from config import settings
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel, target_kind, symbols_and_functions_used
from modules.units_checker import parse_unit, dimension_of, dims_equivalent, UnitParseError, make_dimension_placeholder

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
    # How close this check came to its own pass/fail boundary, as a ratio
    # in [0, 1): 0 means "exact match, essentially zero difference", values
    # near 1 mean "passed, but only just." None for checks that are
    # inherently binary (parsing succeeded/failed, dimensions match/don't)
    # rather than measured against a numeric threshold.
    margin_ratio: float | None = None


def confidence_label(ratio: float) -> str:
    """Qualitative description of a margin_ratio for a passing check."""
    if ratio <= 0.05:
        return "essentially exact"
    if ratio <= 0.25:
        return "comfortable margin"
    if ratio <= 0.6:
        return "adequate margin"
    return "borderline -- close to the tolerance limit"


# kept as an alias since verifier.py's internals were written against this
# name before it was made public for use in app.py
_confidence_label = confidence_label


@dataclass
class VerificationReport:
    checks: list[CheckResult] = field(default_factory=list)
    sympy_numeric_answers: dict[str, float] = field(default_factory=dict)
    llm_independent_answers: dict[str, float] = field(default_factory=dict)
    passed: bool = True
    failure_reason: str | None = None

    def add(self, label: str, passed: bool, detail: str, margin_ratio: float | None = None):
        self.checks.append(CheckResult(label, passed, detail, margin_ratio))
        if not passed:
            self.passed = False

    def confidence(self) -> tuple[str, float | None]:
        """Overall confidence across all passing, margin-bearing checks --
        driven by the WORST (highest) margin_ratio, since one borderline
        check should make the whole verification look borderline even if
        everything else passed with room to spare. Returns (label, worst_ratio).
        """
        if not self.passed:
            return "failed", None
        ratios = [c.margin_ratio for c in self.checks if c.margin_ratio is not None]
        if not ratios:
            return "essentially exact", None  # only binary checks ran, and all passed
        worst = max(ratios)
        return _confidence_label(worst), worst


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
        report.add("Equation parsing", True, "All equations parsed to valid SymPy expressions.",
                    margin_ratio=0.0)

    # 2. every solve_for symbol/function is actually used somewhere
    if model.solve_for:
        used = set()
        for e in model.equations:
            used |= symbols_and_functions_used(e)
        missing = [t for t in model.solve_for if t not in used]
        if missing:
            report.add("Target variable present", False,
                        f"{', '.join(missing)} does not appear in any derived equation.")
        else:
            report.add("Target variable present", True,
                        f"{', '.join(model.solve_for)} all appear in the equations.",
                        margin_ratio=0.0)

    # 3. determinacy: unknowns vs equations (algebraic relations only --
    # inequalities and ODEs have their own solvability semantics, checked
    # separately in _inequality_checks / the ODE solution check)
    unknowns = [v.symbol for v in model.variables
                if v.known_value is None and not v.is_function
                and target_kind(model, v.symbol) != "inequality"]
    n_eqs = len([e for e in model.equations if e.kind == "equation" and e.sympy_eq is not None])
    if len(unknowns) > n_eqs and n_eqs > 0:
        report.add("Determinacy", False,
                    f"{len(unknowns)} unknown(s) ({', '.join(unknowns)}) but only "
                    f"{n_eqs} equation(s) -- system may be underdetermined.")
    else:
        report.add("Determinacy", True,
                    f"{len(unknowns)} unknown(s), {n_eqs} equation(s) -- solvable in principle.",
                    margin_ratio=0.0)


def _numeric_balance_check(model: ProblemModel, report: VerificationReport):
    """If every symbol in an equation has a known value, plugging them in
    should reduce it to (approximately) 0 = 0. Anything else means the
    derivation is inconsistent with the numbers given in the problem.

    Reports not just pass/fail but how close the residual came to the
    tolerance -- a residual of 1e-12 and one of 0.0009 (just under a 1e-3
    tolerance) both "pass," but the second is worth a second look.
    """
    subs = _known_substitutions(model)
    for eq in model.equations:
        if eq.kind != "equation" or eq.sympy_eq is None:
            continue
        free = eq.sympy_eq.free_symbols
        if free and free.issubset(subs.keys()):
            residual = sp.simplify((eq.sympy_eq.lhs - eq.sympy_eq.rhs).subs(subs))
            try:
                val = float(residual)
                tol = settings.numeric_tolerance
                ratio = abs(val) / tol if tol > 0 else (0.0 if val == 0 else float("inf"))
                ok = ratio < 1.0
                if ok:
                    conf = _confidence_label(ratio)
                    detail = (f"Residual = {val:.3g} against a tolerance of {tol:.0e} "
                               f"({ratio:.1%} of the allowed tolerance) -- {conf}.")
                else:
                    detail = (f"Residual = {val:.3g} exceeds the tolerance of {tol:.0e} "
                               f"by {ratio:.1f}x -- DOES NOT balance.")
                report.add(f"Numeric balance: {eq.name}", ok, detail,
                           margin_ratio=ratio if ok else None)
            except TypeError:
                ok = residual == 0
                report.add(f"Numeric balance: {eq.name}", ok,
                            f"Substituting known values gives residual {residual} "
                            f"({'balances exactly' if ok else 'DOES NOT balance'}).",
                            margin_ratio=0.0 if ok else None)


def _dimensional_checks(model: ProblemModel, report: VerificationReport):
    """Checks that both sides of each equation carry the same physical
    dimension, using declared variable units -- catches errors that are
    numerically invisible, e.g. equating a distance to a velocity.

    This is deliberately best-effort: variables with no unit, an empty
    unit, or an unrecognized unit string are simply skipped rather than
    failing the whole check, since not every problem has (or needs)
    physical units.
    """
    units_map: dict[str, sp.Expr] = {}
    unresolved: list[str] = []
    for v in model.variables:
        if v.unit and v.unit.strip().lower() not in ("", "unitless", "dimensionless", "n/a"):
            try:
                units_map[v.symbol] = parse_unit(v.unit)
            except UnitParseError as e:
                unresolved.append(f"{v.symbol} ('{v.unit}')")

    if not units_map:
        return  # nothing declared units-wise -- nothing to check

    if unresolved:
        report.add(
            "Unit resolution", True,  # informational, not a failure
            f"Couldn't recognize unit(s) for: {', '.join(unresolved)} -- skipping dimensional "
            "check for any equation that uses them.",
            margin_ratio=0.0,
        )

    for eq in model.equations:
        if eq.sympy_eq is None:
            continue
        if eq.kind == "ode":
            continue  # handled separately by _ode_dimensional_checks below
        names = {s.name for s in eq.sympy_eq.free_symbols}
        if not names or not names.issubset(units_map.keys()):
            continue  # can't fully check this equation -- some symbol has no known unit
        # each DISTINCT symbol gets its OWN placeholder quantity (same
        # dimension, different object) -- see make_dimension_placeholder's
        # docstring for why reusing one canonical unit object across
        # different symbols would risk a false "a - b" cancellation
        subs = {sp.Symbol(name): make_dimension_placeholder(dimension_of(units_map[name]))
                for name in names}
        try:
            lhs_dim = dimension_of(eq.sympy_eq.lhs.subs(subs))
            rhs_dim = dimension_of(eq.sympy_eq.rhs.subs(subs))
            ok = dims_equivalent(lhs_dim, rhs_dim)
            report.add(
                f"Dimensional consistency: {eq.name}", ok,
                f"LHS dimension = {lhs_dim}, RHS dimension = {rhs_dim}."
                + ("" if ok else " NOT equivalent -- the equation is physically inconsistent "
                                  "regardless of what numbers are plugged in."),
                margin_ratio=0.0 if ok else None,  # dimension match is exact, not a matter of degree
            )
        except ValueError as e:
            # sympy raises this itself when a single side adds incompatible
            # dimensions together, e.g. "v + a" without multiplying a by t
            report.add(f"Dimensional consistency: {eq.name}", False,
                        f"Dimensionally invalid: {e}")

    _ode_dimensional_checks(model, report, units_map)


def _ode_dimensional_substitute(expr: sp.Basic, deriv_atoms: set,
                                  subs_funcs: dict, subs_symbols: dict) -> sp.Basic:
    """Replaces each Derivative(f(t), t, ...) node with unit(f)/unit(t)**n
    BEFORE substituting bare function applications and plain symbols --
    order matters, since once a Derivative node is replaced there's nothing
    left inside it to double-substitute. Operates on a single side (.lhs or
    .rhs) of an equation, never the wrapped Eq itself: substituting into a
    whole Eq can trigger premature auto-evaluation to a bare True/False
    once both sides become fully concrete (e.g. Eq(g/yr, -g/yr) auto-
    resolves to False before dimensions are ever compared), which loses
    the .lhs/.rhs structure the dimension check needs."""
    subs_derivs = {}
    for d in deriv_atoms:
        if d.expr not in subs_funcs:
            continue
        denom = 1
        for var, n in d.variable_count:
            denom *= subs_symbols.get(var, 1) ** n
        subs_derivs[d] = subs_funcs[d.expr] / denom
    return expr.subs(subs_derivs).subs(subs_funcs).subs(subs_symbols)


def _ode_dimensional_checks(model: ProblemModel, report: VerificationReport, units_map: dict[str, sp.Expr]):
    """Dimensional consistency for ODEs: the dimension of d^n(f)/dx^n is
    dim(f)/dim(x)**n. Every declared function (is_function=True variable)
    gets its own unit substituted in for bare applications; each Derivative
    node gets that unit divided by the independent variable's unit raised
    to the derivative's order. Skips (rather than fails) any equation
    referencing a function or symbol with no declared/recognized unit."""
    ode_eqs = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    if not ode_eqs:
        return

    func_units = {v.symbol: units_map[v.symbol] for v in model.variables
                  if v.is_function and v.symbol in units_map}

    for eq in ode_eqs:
        deriv_atoms = eq.sympy_eq.atoms(sp.Derivative)
        applied_funcs = eq.sympy_eq.atoms(AppliedUndef)
        func_names_used = {str(f.func) for f in applied_funcs}
        plain_symbol_names = {s.name for s in eq.sympy_eq.free_symbols}

        # every function AND every plain symbol involved needs a known unit
        if not func_names_used.issubset(func_units.keys()):
            continue
        if not plain_symbol_names.issubset(units_map.keys()):
            continue

        subs_funcs = {f: make_dimension_placeholder(dimension_of(func_units[str(f.func)]))
                       for f in applied_funcs}
        subs_symbols = {sp.Symbol(name): make_dimension_placeholder(dimension_of(units_map[name]))
                         for name in plain_symbol_names}

        try:
            lhs_sub = _ode_dimensional_substitute(eq.sympy_eq.lhs, deriv_atoms, subs_funcs, subs_symbols)
            rhs_sub = _ode_dimensional_substitute(eq.sympy_eq.rhs, deriv_atoms, subs_funcs, subs_symbols)
            lhs_dim = dimension_of(lhs_sub)
            rhs_dim = dimension_of(rhs_sub)
            ok = dims_equivalent(lhs_dim, rhs_dim)
            report.add(
                f"Dimensional consistency: {eq.name}", ok,
                f"LHS dimension = {lhs_dim}, RHS dimension = {rhs_dim}."
                + ("" if ok else " NOT equivalent -- the differential equation is physically "
                                  "inconsistent regardless of what numbers are plugged in."),
                margin_ratio=0.0 if ok else None,
            )
        except ValueError as e:
            report.add(f"Dimensional consistency: {eq.name}", False, f"Dimensionally invalid: {e}")
        except Exception as e:  # noqa: BLE001
            report.add(f"Dimensional consistency: {eq.name}", True,
                        f"Skipped -- could not evaluate dimensions for this ODE ({e}).",
                        margin_ratio=0.0)


def _inequality_checks(model: ProblemModel, report: VerificationReport):
    """When every symbol in an inequality-kind constraint has a known
    value, substitutes them in and checks whether the constraint actually
    holds -- e.g. flags a derived "v <= 25" constraint that's violated by
    the problem's own stated numbers."""
    subs = _known_substitutions(model)
    for eq in model.equations:
        if eq.kind != "inequality" or eq.sympy_eq is None:
            continue
        free = eq.sympy_eq.free_symbols
        if not free or not free.issubset(subs.keys()):
            continue
        substituted = eq.sympy_eq.subs(subs)
        try:
            truth = bool(substituted)
        except TypeError:
            continue  # couldn't reduce to a definite True/False -- skip silently
        report.add(
            f"Constraint satisfied: {eq.name}", truth,
            f"With known values substituted, the constraint becomes {substituted} "
            f"({'holds' if truth else 'DOES NOT hold -- inconsistent with the given numbers'}).",
            margin_ratio=0.0 if truth else None,
        )


def _ode_checks(model: ProblemModel, report: VerificationReport):
    """ODE-specific verification: rather than a numeric residual, this
    substitutes the solved solution back into the ORIGINAL differential
    equation(s) -- an exact symbolic check (not a numeric approximation)
    that the solution actually satisfies the equation(s) it was derived
    from. This replaces the numeric-balance check for ODEs, which doesn't
    apply to a relation between a function and its derivative.

    Coupled systems (e.g. a decay chain A -> B) are verified as a group:
    checkodesol alone only knows about one function/equation at a time and
    gives false negatives on a coupled system, since it doesn't substitute
    the OTHER function's solution in before checking. verify_coupled_solution
    substitutes every function's solution into every equation in its group
    simultaneously, which is the correct check for coupling.
    """
    from modules.ode_utils import solve_ode, group_coupled_odes, verify_coupled_solution

    ode_eqs = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    if not ode_eqs:
        return

    solutions = solve_ode(model)
    groups = group_coupled_odes(ode_eqs)

    for group in groups:
        names_in_group = ", ".join(sorted({
            str(next(iter(e.sympy_eq.atoms(AppliedUndef))).func) for e in group
        }))

        # if any equation in the group has no solution at all, nothing to verify
        missing = [e for e in group
                   if str(next(iter(e.sympy_eq.atoms(AppliedUndef))).func) not in solutions]
        if missing:
            for e in missing:
                report.add(f"ODE solved: {e.name}", False,
                            "SymPy's dsolve()/dsolve_system() could not find a closed-form "
                            "solution for this equation.")
            continue

        if len(group) == 1:
            eq = group[0]
            func_name = str(next(iter(eq.sympy_eq.atoms(AppliedUndef))).func)
            solution = solutions[func_name]
            try:
                ok, remainder = sp.checkodesol(eq.sympy_eq, solution)
                detail = (
                    f"Solution {solution} verified by substituting back into the original "
                    "differential equation (exact symbolic check, not a numeric approximation)."
                    if ok else
                    f"Substituting the solution back into the ODE leaves a nonzero remainder "
                    f"({remainder}) -- the solution does not actually satisfy the equation."
                )
                report.add(f"ODE solution check: {eq.name}", bool(ok), detail,
                            margin_ratio=0.0 if ok else None)
            except Exception as e:  # noqa: BLE001
                report.add(f"ODE solution check: {eq.name}", False, f"Could not verify: {e}")
        else:
            try:
                ok, residual = verify_coupled_solution(group, solutions)
                group_label = " & ".join(e.name for e in group)
                detail = (
                    f"Coupled solution for {names_in_group} verified by substituting all "
                    "functions' solutions into every equation in the group simultaneously "
                    "(exact symbolic check)."
                    if ok else
                    f"Substituting the coupled solutions back leaves a nonzero remainder "
                    f"({residual}) in at least one equation -- the system's solution is "
                    "inconsistent."
                )
                report.add(f"Coupled ODE system check: {group_label}", bool(ok), detail,
                            margin_ratio=0.0 if ok else None)
            except Exception as e:  # noqa: BLE001
                report.add(f"Coupled ODE system check: {names_in_group}", False,
                            f"Could not verify: {e}")


def _solve_sympy(model: ProblemModel) -> dict[str, float]:
    """Solves the whole EQUATION-kind system at once for every algebraic
    target -- important because targets can be coupled (e.g. displacement
    'd' may depend on an also-unknown acceleration 'a' solved from another
    equation in the same system). Inequality and ODE targets are handled
    by their own checks and are deliberately excluded here."""
    algebraic_targets = [t for t in model.solve_for if target_kind(model, t) == "equation"]
    if not algebraic_targets:
        return {}
    subs = _known_substitutions(model)
    eqs = [e.sympy_eq.subs(subs) for e in model.equations if e.kind == "equation" and e.sympy_eq is not None]
    if not eqs:
        return {}
    targets = [sp.Symbol(t) for t in algebraic_targets]
    try:
        sol = sp.solve(eqs, targets, dict=True)
        if not sol:
            return {}
        result = {}
        for t, sym in zip(algebraic_targets, targets):
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
    _dimensional_checks(model, report)
    _inequality_checks(model, report)
    _ode_checks(model, report)

    sympy_answers = _solve_sympy(model)
    report.sympy_numeric_answers = sympy_answers

    if sympy_answers:
        # only ask about targets we actually have an algebraic answer for --
        # inequality/ODE targets are verified separately above and would
        # just confuse this prompt (they often have no single number)
        algebraic_targets = list(sympy_answers.keys())
        raw = client.chat(
            system="You are a careful independent problem solver.",
            user=INDEPENDENT_SOLVE_PROMPT.format(
                targets=", ".join(algebraic_targets), problem=problem_text),
            temperature=0.0,
        )
        llm_answers = _extract_final_numbers(raw)
        report.llm_independent_answers = llm_answers

        for target in algebraic_targets:
            sympy_val = sympy_answers.get(target)
            llm_val = llm_answers.get(target)
            if sympy_val is None or llm_val is None:
                continue
            rel_diff = abs(llm_val - sympy_val) / max(abs(sympy_val), 1e-9)
            tol = settings.cross_check_tolerance
            ratio = rel_diff / tol
            agree = ratio < 1.0
            if agree:
                conf = _confidence_label(ratio)
                detail = (f"Derived-equation answer = {sympy_val:.6g}; independent re-solve = "
                           f"{llm_val:.6g} -- {rel_diff:.2%} apart, against a {tol:.0%} tolerance "
                           f"({conf}).")
            else:
                detail = (f"Derived-equation answer = {sympy_val:.6g}; independent re-solve = "
                           f"{llm_val:.6g} -- {rel_diff:.2%} apart, exceeding the {tol:.0%} "
                           "tolerance. DISAGREE -- derivation likely flawed.")
            report.add(f"Independent cross-check: {target}", agree, detail,
                       margin_ratio=ratio if agree else None)

    if not report.passed:
        failing = [c for c in report.checks if not c.passed]
        report.failure_reason = " | ".join(f"{c.label}: {c.detail}" for c in failing)

    return report
