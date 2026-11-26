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

from contextlib import contextmanager, nullcontext
from dataclasses import dataclass, field
from typing import Any, Callable, ContextManager

from config import settings
from modules.app_logging import logger
from modules.equation_engine import ProblemModel, extract_model
from modules.llm_client import LMStudioClient
from modules.scenarios import generate_alternative_scenarios
from modules.solver import SolutionStep, compute_steps, narrate_steps
from modules.verifier import VerificationReport, verify
from modules.workspace import bind_workspace_values

# label -> context manager. st.spinner satisfies this as-is.
StageFn = Callable[[str], ContextManager[Any]]


def _no_stage(label: str) -> ContextManager[Any]:
    return nullcontext()


@contextmanager
def _tagged(stage: StageFn, label: str):
    """Run `stage(label)` and, if the work inside raises, record WHICH stage it was on the exception (`exc.pipeline_stage`), so the failure can be explained
    as "while deriving the equations" rather than just "something failed". The exception itself is re-raised unchanged."""
    with stage(label):
        try:
            yield
        except Exception as exc:  # noqa: BLE001
            if not hasattr(exc, "pipeline_stage"):
                try:
                    exc.pipeline_stage = label                      # type: ignore[attr-defined]
                except Exception:  # noqa: BLE001 -- an exception type that refuses attributes is still re-raised
                    pass
            raise


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
    # stages AFTER the derivation that could not be completed (the step trace, narration, scenarios): the derivation and its verification are still good
    warnings: list[str] = field(default_factory=list)


def extract_and_verify(client: LMStudioClient, problem_text: str, *,
                        known_context: str | None = None,
                        workspace_entries: dict | None = None,
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

    with _tagged(stage, "Deriving equations from the problem statement..."):
        model = extract_model(client, problem_text, known_context=known_context)
        bind_workspace_values(model, workspace_entries, problem_text)         # the model may have ignored the values it was told

    with _tagged(stage, "Verifying the derivation..."):
        report = verify(model, client, problem_text)
        retries = 0
        while not report.passed and retries < limit:
            retries += 1
            with _tagged(stage, f"Verification failed -- retrying derivation ({retries}/{limit})..."):
                model = extract_model(client, problem_text, retry_reason=report.failure_reason,
                                       known_context=known_context)
                bind_workspace_values(model, workspace_entries, problem_text)
                report = verify(model, client, problem_text)

    return VerifiedDerivation(model=model, report=report, retries=retries)


def run_pipeline(client: LMStudioClient, problem_text: str, *,
                  known_context: str | None = None,
                  workspace_entries: dict | None = None,
                  narrate: bool = True,
                  with_scenarios: bool = True,
                  max_retries: int | None = None,
                  stage: StageFn | None = None) -> PipelineResult:
    """The full single-problem pipeline. `narrate` / `with_scenarios`
    each cost one extra LLM round trip and never change whether the math
    is right, which is why batch mode turns them off by default."""
    stage = stage or _no_stage
    derived = extract_and_verify(client, problem_text, known_context=known_context, workspace_entries=workspace_entries,
                                  max_retries=max_retries, stage=stage)
    model = derived.model

    # Everything below needs the derivation, not the other way round: if one of these stages fails, the equations and their verification are still
    # worth showing, so the failure becomes a warning on the result instead of an exception that throws the derivation away.
    warnings: list[str] = []
    steps: dict[str, list[SolutionStep]] = {}
    try:
        with _tagged(stage, "Computing step-by-step solution..."):
            steps = compute_steps(model)
            if narrate:
                steps = narrate_steps(client, model, steps)
    except Exception as exc:  # noqa: BLE001
        logger.warning("run_pipeline(): step trace failed: %s", exc)
        warnings.append(f"The step-by-step solution could not be computed ({type(exc).__name__}: {str(exc)[:160]}). "
                        "The derived equations and their verification are shown without it.")

    scenarios: list[dict] = []
    if with_scenarios:
        try:
            with _tagged(stage, "Generating alternative scenarios..."):
                scenarios = generate_alternative_scenarios(client, model)
        except Exception as exc:  # noqa: BLE001
            logger.warning("run_pipeline(): scenarios failed: %s", exc)
            warnings.append(f"Alternative scenarios could not be generated ({type(exc).__name__}: {str(exc)[:160]}).")

    return PipelineResult(model=model, report=derived.report, steps=steps,
                           scenarios=scenarios, retries=derived.retries, warnings=warnings)
