"""
The word-problem pipeline as one importable function: extract -> verify ->
(retry with the failure reason fed back) -> step trace -> optional
narration -> optional alternative scenarios.

Why this exists: this exact sequence used to live in two places -- inline
Streamlit script code in app.py, and a hand-mirrored copy in
batch_solver.solve_one() (mirrored precisely because the app's copy wasn't
importable). Two copies of the retry loop is how a fix or a behaviour
change lands in one and silently not the other. Now app.py and
batch_solver.py both call run_pipeline() here, so there is exactly one
place that decides "how many retries, what gets fed back, what order".

This module has no Streamlit import (same rule as everything else in
modules/). The one place the UI legitimately needs to hook in -- showing a
spinner around each stage -- is handled by the `stage` parameter: a
callable that takes a human-readable label and returns a context manager.
app.py passes `st.spinner` directly; everything else gets a no-op.

Deliberately NOT routed through here: cli.py's `solve` and the chains
tab's "Extract & verify" both run extract -> verify once with NO retry
loop (a different, deliberate behaviour: report the first attempt
honestly rather than silently re-prompting). cli.py's tests also patch
`cli.extract_model` directly, so both were left as they are rather than
changing what they do under a refactor whose point is "no behaviour
change". `extract_and_verify(..., max_retries=0)` is the drop-in
equivalent if either ever wants to converge on this module.
"""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager

from config import settings
from modules.equation_engine import ProblemModel, extract_model
from modules.llm_client import LMStudioClient
from modules.scenarios import generate_alternative_scenarios
from modules.solver import SolutionStep, compute_steps, narrate_steps
from modules.verifier import VerificationReport, verify

# label -> context manager. st.spinner satisfies this as-is.
StageFn = Callable[[str], ContextManager[Any]]


def _no_stage(label: str) -> ContextManager[Any]:
    return nullcontext()


@dataclass
class VerifiedDerivation:
    model: ProblemModel
    report: VerificationReport
    retries: int = 0


@dataclass
class PipelineResult:
    model: ProblemModel
    report: VerificationReport
    steps: dict[str, list[SolutionStep]]
    scenarios: list[dict] = field(default_factory=list)
    retries: int = 0


def extract_and_verify(client: LMStudioClient, problem_text: str, *,
                        known_context: str | None = None,
                        max_retries: int | None = None,
                        stage: StageFn | None = None) -> VerifiedDerivation:
    """Extraction + verification with the bounded retry loop.

    A failed verification re-runs extraction with `report.failure_reason`
    appended to the prompt, up to `max_retries` times (default: whatever
    settings.max_verification_retries is *at call time* -- the app's
    Advanced-settings panel mutates that live, so it must not be captured
    at import). The returned report is the LAST attempt's, pass or fail;
    a persistent failure is returned, not raised -- the caller decides how
    to present a derivation that never verified.

    Raises whatever extract_model()/verify() raise (LLMOutputError for an
    unusable model response, most commonly) -- callers own error display.
    """
    stage = stage or _no_stage
    limit = settings.max_verification_retries if max_retries is None else max_retries

    with stage("Deriving equations from the problem statement..."):
        model = extract_model(client, problem_text, known_context=known_context)

    with stage("Verifying the derivation..."):
        report = verify(model, client, problem_text)
        retries = 0
        while not report.passed and retries < limit:
            retries += 1
            with stage(f"Verification failed -- retrying derivation ({retries}/{limit})..."):
                model = extract_model(client, problem_text, retry_reason=report.failure_reason,
                                       known_context=known_context)
                report = verify(model, client, problem_text)

    return VerifiedDerivation(model=model, report=report, retries=retries)


def run_pipeline(client: LMStudioClient, problem_text: str, *,
                  known_context: str | None = None,
                  narrate: bool = True,
                  with_scenarios: bool = True,
                  max_retries: int | None = None,
                  stage: StageFn | None = None) -> PipelineResult:
    """The full single-problem pipeline. `narrate` / `with_scenarios`
    each cost one extra LLM round trip and never change whether the math
    is right, which is why batch mode turns them off by default."""
    stage = stage or _no_stage
    derived = extract_and_verify(client, problem_text, known_context=known_context,
                                  max_retries=max_retries, stage=stage)
    model = derived.model

    with stage("Computing step-by-step solution..."):
        steps = compute_steps(model)
        if narrate:
            steps = narrate_steps(client, model, steps)

    scenarios: list[dict] = []
    if with_scenarios:
        with stage("Generating alternative scenarios..."):
            scenarios = generate_alternative_scenarios(client, model)

    return PipelineResult(model=model, report=derived.report, steps=steps,
                           scenarios=scenarios, retries=derived.retries)
