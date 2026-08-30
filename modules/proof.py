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
