"""
Multi-model cross-verification ("paranoid mode"): the existing
independent cross-check in verifier.py re-asks an LLM to solve the word
problem from scratch and compares its stated final answer against the
derived equations' answer -- but by default that call uses the SAME
model as extraction. A model can be wrong in a way that's entirely
self-consistent: it misreads the problem the same way whether asked to
extract equations or asked to just "solve it and give a number," and
single-model verification structurally can't catch that class of error.

This module re-runs the FULL extraction pipeline through a SECOND,
independently-configured model and compares the two derivations two
ways: do their equations have the same structural SHAPE (reusing
similarity.py's canonicalization -- the same trick "find similar past
problems" uses, just comparing two live derivations against each other
instead of one against history), and do their final numeric answers
agree within the normal cross-check tolerance? Disagreement between two
different models is a much stronger signal of a genuine problem than
either model's own self-reported confidence.

Off by default (config.settings.secondary_reasoning_model is blank
until a person configures a second loaded model) -- this doubles the
extraction cost of a problem, so it's opt-in, not something every
solve pays for.
"""
from dataclasses import dataclass, field

from config import settings
from modules.equation_engine import extract_model, ProblemModel
from modules.llm_client import LMStudioClient, LLMOutputError
from modules.similarity import problem_shape, jaccard_similarity
from modules.verifier import _solve_sympy


@dataclass
class ParanoidResult:
    ran: bool                                   # False if no secondary model was configured
    secondary_model: str | None = None
    secondary_problem_model: ProblemModel | None = None
    equations_match: float | None = None        # Jaccard similarity of the two derivations' equation shapes
    primary_answers: dict[str, float] = field(default_factory=dict)
    secondary_answers: dict[str, float] = field(default_factory=dict)
    disagreements: dict[str, tuple[float, float]] = field(default_factory=dict)  # target -> (primary, secondary)
    error: str | None = None


def run_paranoid_check(client: LMStudioClient, problem_text: str, primary_model: ProblemModel,
                        primary_answers: dict[str, float],
                        secondary_model_name: str | None = None) -> ParanoidResult:
    """Re-runs extraction + solving through secondary_model_name (falls
    back to settings.secondary_reasoning_model if not given) and
    compares against the already-verified primary derivation's
    equations (structural shape) and numeric answers. Never raises --
    an API error or malformed response on the secondary model's side is
    captured on the result rather than breaking the primary,
    already-verified result the caller has in hand."""
    secondary_model_name = secondary_model_name or settings.secondary_reasoning_model
    if not secondary_model_name:
        return ParanoidResult(ran=False)

    try:
        secondary = extract_model(client, problem_text, model=secondary_model_name)
    except LLMOutputError as e:
        return ParanoidResult(ran=True, secondary_model=secondary_model_name, error=str(e))
    except Exception as e:  # noqa: BLE001
        return ParanoidResult(ran=True, secondary_model=secondary_model_name,
                                error=f"Unexpected error: {e}")

    shape_sim = jaccard_similarity(problem_shape(primary_model), problem_shape(secondary))
    secondary_answers = _solve_sympy(secondary)

    disagreements = {}
    for target, primary_val in primary_answers.items():
        secondary_val = secondary_answers.get(target)
        if secondary_val is None:
            continue
        tol = max(abs(primary_val), 1e-9) * settings.cross_check_tolerance
        if abs(primary_val - secondary_val) > tol:
            disagreements[target] = (primary_val, secondary_val)

    return ParanoidResult(
        ran=True, secondary_model=secondary_model_name, secondary_problem_model=secondary,
        equations_match=shape_sim, primary_answers=dict(primary_answers),
        secondary_answers=secondary_answers, disagreements=disagreements,
    )
