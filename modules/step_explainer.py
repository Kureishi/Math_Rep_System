"""
Step-level "explain just this" drill-down: a narrower, more surgical
sibling of followup.py's whole-problem grounded Q&A. Rather than asking
about the problem as a whole ("why this formula?"), lets someone point
at ONE specific step from the already-rendered step-by-step solution
and ask for a more scaffolded re-explanation of JUST that step, without
re-deriving or re-summarizing the whole problem around it.

Grounding is deliberately narrow: the prompt includes only the target
step's own description/expression/(existing) explanation, plus the
minimal surrounding context of which step number it is and what the
overall target is -- not the full equation list followup.py's
conceptual answers ground against. The point is a tighter, more
focused explanation of one arithmetic/algebraic move, not a second
whole-problem summary squeezed into a smaller box.

Unverified in the same sense followup.py's conceptual answers are
unverified: the step's own MATH was already verified upstream (it comes
straight from solver.compute_steps(), which only narrates an already-
confirmed derivation); what isn't independently re-checked here is the
PROSE explanation of that step -- the same honest limitation
followup.py's own docstring states for its conceptual branch.
"""
from dataclasses import dataclass

from modules.equation_engine import ProblemModel
from modules.llm_client import LMStudioClient
from modules.solver import SolutionStep

EXPLAIN_STEP_SYSTEM_PROMPT = """You are explaining ONE STEP of an already-solved and verified \
math problem to a student who followed everything up to this point but is stuck on THIS step \
specifically. Explain only this step -- don't re-derive or summarize the whole problem. Use plain, \
concrete language. Keep it to 2-5 sentences unless the step genuinely needs more."""

EXPLAIN_STEP_PROMPT = """Problem domain: {domain}
Overall target: {target}

This is step {step_number} of the derivation:
Description: {description}
Expression: {expression}
{prior_explanation}

The student is stuck on THIS step. {mode_instruction}
"""

_MODE_INSTRUCTIONS = {
    "default": "Explain what this step is doing and why.",
    "simpler": ("The student found the existing explanation too dense -- explain this step more "
                 "simply, breaking it into smaller pieces, as if to someone newer to this topic."),
    "example": ("Illustrate this step with a small worked mini-example using simple made-up "
                 "numbers, showing the same operation being applied."),
}


@dataclass
class StepExplanation:
    step_number: int
    mode: str
    text: str
    error: str | None = None


def explain_step(client: LMStudioClient, model: ProblemModel, steps: list[SolutionStep],
                  step_number: int, target_name: str = "", mode: str = "default") -> StepExplanation:
    """Explains ONE step (1-indexed `step_number` into `steps`) in
    isolation. `mode` selects the scaffolding style: "default" (a
    plain explanation), "simpler" (more broken-down/beginner-friendly),
    or "example" (a small worked mini-example of the same operation)."""
    if not steps:
        return StepExplanation(step_number, mode, "", error="No steps to explain.")
    if not (1 <= step_number <= len(steps)):
        return StepExplanation(step_number, mode, "",
                                 error=f"Step {step_number} doesn't exist (this derivation has "
                                       f"{len(steps)} step(s)).")

    step = steps[step_number - 1]
    mode_instruction = _MODE_INSTRUCTIONS.get(mode, _MODE_INSTRUCTIONS["default"])
    prior_explanation = f"Existing explanation: {step.explanation}" if step.explanation else ""

    try:
        raw = client.chat(
            system=EXPLAIN_STEP_SYSTEM_PROMPT,
            user=EXPLAIN_STEP_PROMPT.format(
                domain=model.problem_domain, target=target_name or "(unspecified)",
                step_number=step_number, description=step.description, expression=step.expression,
                prior_explanation=prior_explanation, mode_instruction=mode_instruction,
            ),
            temperature=0.3,
        )
        return StepExplanation(step_number, mode, raw.strip())
    except Exception as e:  # noqa: BLE001
        return StepExplanation(step_number, mode, "", error=f"Couldn't reach the model ({e}).")
