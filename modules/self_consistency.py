"""
Self-consistency check: re-runs extraction on the SAME model 2-5 times
(at its normal, non-zero extraction temperature) and compares the
resulting derivations via similarity.py's equation-shape canonicalization
-- distinct from paranoid.py's cross-model check (which asks "would a
DIFFERENT model derive this differently"), this asks "does the SAME
model derive this the same way twice in a row."

Those are genuinely different signals. Two different models disagreeing
suggests one of them is specifically wrong. The SAME model disagreeing
with ITSELF across repeated runs of the identical prompt usually means
something else: the PROBLEM STATEMENT is ambiguous or underspecified
enough that even one model can't parse it the same way twice -- worth
surfacing regardless of which particular derivation ends up being used,
since it's a property of the input, not of any one model's competence.
"""
from dataclasses import dataclass, field

from modules.equation_engine import extract_model, ProblemModel
from modules.llm_client import LMStudioClient
from modules.similarity import problem_shape, jaccard_similarity


@dataclass
class SelfConsistencyResult:
    runs: int
    shapes_match: list[float] = field(default_factory=list)  # each subsequent run's similarity vs the first
    consistent: bool | None = None    # None if fewer than 2 runs produced a usable model to compare
    models: list[ProblemModel | None] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def run_self_consistency_check(client: LMStudioClient, problem_text: str,
                                runs: int = 3, min_similarity: float = 0.7) -> SelfConsistencyResult:
    """Runs extraction `runs` times independently (each a fresh call --
    no shared context between them) and compares every subsequent run's
    equation shape against the FIRST run's via Jaccard similarity.
    `consistent` is True if every run scored at or above min_similarity
    against the first, False if any scored below it, None if fewer than
    2 runs produced a usable model at all (can't compare with only one
    or zero). Never raises -- a failed individual run is recorded in
    `errors` and simply excluded from the comparison, same as any other
    graceful-degradation path in this app."""
    runs = max(2, min(runs, 5))
    models: list[ProblemModel | None] = []
    errors: list[str] = []
    for _ in range(runs):
        try:
            models.append(extract_model(client, problem_text))
        except Exception as e:  # noqa: BLE001
            models.append(None)
            errors.append(str(e))

    usable = [m for m in models if m is not None]
    if len(usable) < 2:
        return SelfConsistencyResult(runs=runs, models=models, errors=errors, consistent=None)

    base_shape = problem_shape(usable[0])
    shapes_match = [jaccard_similarity(base_shape, problem_shape(m)) for m in usable[1:]]
    consistent = all(s >= min_similarity for s in shapes_match)

    return SelfConsistencyResult(runs=runs, shapes_match=shapes_match, consistent=consistent,
                                   models=models, errors=errors)
