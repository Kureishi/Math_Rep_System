"""
Symbolic proof mode: for "are these two expressions the same" questions
equivalence.py already confirmed symbolically equivalent, this renders
the ACTUAL sequence of SymPy simplifications that reduce their
difference to zero as readable proof steps -- not a fabricated
derivation, the real transformation SymPy applies at each stage, just
reported incrementally instead of only handing back the final True the
way equivalence.py does.

Applies a fixed, pedagogically-ordered sequence of named simplification
passes (expand, combine into a single fraction, trig identities,
combine powers, combine logs, simplify radicals, factor, general
simplification) to the difference of the two expressions, keeping only
the steps that actually change something structurally (so two already-
close expressions don't get padded with redundant identical-looking
lines), and stopping the moment the difference reaches exactly zero.

Scoped to cases equivalence.py already found symbolically equivalent
(method == "symbolic", equivalent is True) -- there's no proof to walk
through for something that's only equivalent by numeric-sampling
evidence (see equivalence.py's own docstring on why that's evidence,
not proof) or that isn't equivalent at all.
"""
import sympy as sp
from dataclasses import dataclass, field

from modules.equivalence import EquivalenceResult
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_STEPS: list[tuple[str, callable]] = [
    ("Expand", sp.expand),
    ("Combine into a single fraction", sp.together),
    ("Apply trigonometric identities", sp.trigsimp),
    ("Combine powers with matching bases", lambda e: sp.powsimp(e, force=True)),
    ("Combine logarithms", lambda e: sp.logcombine(e, force=True)),
    ("Simplify radicals", sp.radsimp),
    ("Factor", sp.factor),
    ("General simplification", sp.simplify),
]


def build_proof(equivalence_result: EquivalenceResult) -> list[tuple[str, str]] | None:
    """Returns [(technique_name, resulting_expression_latex), ...] for a
    symbolically-confirmed equivalence, starting from the difference of
    the two expressions and ending at 0. Returns None if there's
    nothing to prove (equivalence_result wasn't a confirmed symbolic
    equivalence in the first place)."""
    if equivalence_result.equivalent is not True or equivalence_result.method != "symbolic":
        return None
    diff = equivalence_result.raw_difference
    if diff is None:
        return None

    steps: list[tuple[str, str]] = [("Start from the difference of the two expressions", sp.latex(diff))]
    current = diff
    for name, transform in _STEPS:
        try:
            new_expr = run_with_timeout(transform, current, label=f"proof step: {name}")
        except ComputationTimeoutError:
            # a single pass in the chain ran long -- skip just this
            # technique and try the next one, rather than losing every
            # earlier step (or hanging the whole proof) over one slow pass
            continue
        except Exception:  # noqa: BLE001
            continue
        if new_expr == current:
            continue  # no structural change -- skip, don't pad the proof
        steps.append((name, sp.latex(new_expr)))
        current = new_expr
        if current == 0:
            break

    if current != 0:
        # The recorded equivalence check already confirmed True via
        # e1.equals(e2), which can succeed through an internal method
        # (e.g. numeric confirmation on a case this fixed step sequence
        # doesn't fully reduce) that this named sequence doesn't reach.
        # Rather than claim a false proof trail ending somewhere other
        # than zero, say so explicitly instead of overstating what was shown.
        steps.append((
            "SymPy's internal equality check confirms this is zero, though the named "
            "steps above didn't fully reduce it -- inspect the final expression directly",
            sp.latex(current),
        ))
    return steps


# ------------------------------------------------------------- induction proofs
@dataclass
class InductionStep:
    label: str
    detail: str
    verified: bool


@dataclass
class InductionProofResult:
    steps: list = field(default_factory=list)   # list[InductionStep]
    valid: bool = False
    conclusion: str = ""
    error: str | None = None


def build_recurrence_induction_proof(eq_sympy: sp.Eq, func_name: str, closed_form: sp.Expr,
                                       indep_var: sp.Symbol,
                                       base_cases: dict[int, float]) -> InductionProofResult:
    """A genuine proof by (strong) induction that closed_form(n) equals
    the sequence defined by eq_sympy + base_cases, for every n at or
    above the smallest base index -- not just a restatement of
    recurrence_utils.verify_recurrence_solution's identity check, but
    the actual two-part induction argument that check is one half of:

    1. BASE CASE(S): closed_form evaluated at each given base index
       must equal the stated base value -- checked here directly by
       substitution, independently of the recurrence relation itself.
       (This is the step recurrence_utils.solve_recurrence's rsolve()
       already relies on internally when it solves for the closed
       form's constants, but rsolve's own bookkeeping is exactly the
       kind of thing a genuinely independent re-check is for -- the
       same "don't just trust the first solve path" principle behind
       ode_utils.numerical_cross_check.)
    2. INDUCTIVE STEP: substituting closed_form for every shifted
       occurrence of the function in the ORIGINAL recurrence relation
       (a(n), a(n+1), a(n+2), ...) produces an identity that holds for
       GENERAL n -- delegated to
       recurrence_utils.verify_recurrence_solution, which is exactly
       this substitution. Because this holds for a symbolic n rather
       than one specific value, it establishes "IF the formula is
       correct up through the indices this recurrence references, THEN
       it's correct at the next index" for every n simultaneously --
       which is precisely the inductive step's logical content.

    Together, 1 and 2 constitute a complete induction proof: the base
    case(s) anchor the formula at the start, and the inductive step
    carries correctness forward from there to every subsequent index,
    covering all n from the base upward with no separate case-by-case
    checking needed."""
    from modules.recurrence_utils import verify_recurrence_solution

    if not base_cases:
        return InductionProofResult(error="No base case(s) given -- an induction proof needs at "
                                             "least one anchor point to start from.")

    steps = []
    base_ok = True
    for index in sorted(base_cases):
        expected = base_cases[index]
        try:
            actual = closed_form.subs(indep_var, index)
            actual_val = complex(actual)
            matches = abs(actual_val - complex(expected)) < 1e-6
        except (TypeError, ValueError) as exc:
            matches = False
            actual = f"could not evaluate ({exc})"
        base_ok = base_ok and matches
        steps.append(InductionStep(
            label=f"Base case: n = {index}",
            detail=(f"Closed form gives {func_name}({index}) = {sp.nsimplify(actual) if matches else actual}, "
                    f"matching the given value {expected}." if matches else
                    f"Closed form gives {func_name}({index}) = {actual}, which does NOT match the "
                    f"given value {expected}."),
            verified=matches))

    try:
        inductive_ok, residual = verify_recurrence_solution(eq_sympy, func_name, closed_form, indep_var)
    except Exception as exc:  # noqa: BLE001
        inductive_ok, residual = False, f"could not verify ({exc})"
    steps.append(InductionStep(
        label=f"Inductive step: assume the formula holds up to n = k, show it then holds at the "
              f"next index",
        detail=(f"Substituting the closed form into the recurrence relation for a general (symbolic) "
                f"n reduces the relation to an identity (residual = {residual}) -- so the formula "
                f"being correct at the index/indices this recurrence relation references is enough "
                f"to guarantee it's correct at the next one, for every n at once."
                if inductive_ok else
                f"Substituting the closed form into the recurrence relation for a general n leaves a "
                f"nonzero residual ({residual}) -- the formula does not actually satisfy the "
                f"recurrence relation, so the inductive step fails."),
        verified=bool(inductive_ok)))

    valid = base_ok and bool(inductive_ok)
    conclusion = (
        f"By induction: the base case(s) hold and the inductive step carries correctness forward "
        f"from them, so {func_name}(n) = {sp.latex(closed_form)} for every n at or above "
        f"{min(base_cases)}." if valid else
        "The induction proof does NOT go through -- see the failing step above."
    )
    return InductionProofResult(steps=steps, valid=valid, conclusion=conclusion)
