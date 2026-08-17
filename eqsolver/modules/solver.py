"""
Produces a step-by-step solution.

Approach: SymPy computes the actual solve steps we can trust (isolate,
substitute, simplify). The LLM is then given those verified intermediate
results and asked only to narrate/explain them in plain language --
it is not trusted to invent the math itself here, only to explain math
that has already been checked. This avoids the common failure mode of
LLM step-by-step solutions that "look right" but skip or fudge a step.
"""
from dataclasses import dataclass
import sympy as sp

from config import settings
from modules.llm_client import LMStudioClient
from modules.equation_engine import ProblemModel
from modules.verifier import _known_substitutions  # reuse the same substitution logic

NARRATION_PROMPT = """Given this verified sequence of algebraic steps solving for {target}, \
write a brief, clear explanation for each step in plain language (one short sentence per step). \
Respond as a JSON array of strings, same length and order as the steps, no other text.

Steps:
{steps}
"""


@dataclass
class SolutionStep:
    description: str
    expression: str
    explanation: str = ""


def compute_steps(model: ProblemModel) -> list[SolutionStep]:
    """Deterministic SymPy trace: substitute knowns -> isolate target -> simplify."""
    if not model.solve_for:
        return []

    target = sp.Symbol(model.solve_for)
    subs = _known_substitutions(model)
    eqs = [e.sympy_eq for e in model.equations if e.sympy_eq is not None]
    if not eqs:
        return []

    steps: list[SolutionStep] = []

    # Step 1: state the governing equation(s)
    for e, orig in zip(eqs, model.equations):
        steps.append(SolutionStep(
            description=f"Start from: {orig.name}",
            expression=sp.latex(e),
        ))

    # Step 2: substitute known values
    if subs:
        substituted = [e.subs(subs) for e in eqs]
        readable = ", ".join(f"{k} = {v}" for k, v in subs.items())
        for orig_eq, sub_eq in zip(eqs, substituted):
            if sub_eq != orig_eq:
                steps.append(SolutionStep(
                    description=f"Substitute known values ({readable})",
                    expression=sp.latex(sub_eq),
                ))
        eqs = substituted

    # Step 3: solve for the target
    try:
        solutions = sp.solve(eqs, target, dict=True)
    except Exception:  # noqa: BLE001
        solutions = []

    if solutions and target in solutions[0]:
        result = sp.simplify(solutions[0][target])
        steps.append(SolutionStep(
            description=f"Isolate and simplify to solve for {model.solve_for}",
            expression=f"{model.solve_for} = {sp.latex(result)}",
        ))
        if result.is_number:
            steps.append(SolutionStep(
                description="Numeric result",
                expression=f"{model.solve_for} = {sp.N(result, 6)}",
            ))

    return steps


def narrate_steps(client: LMStudioClient, model: ProblemModel, steps: list[SolutionStep]) -> list[SolutionStep]:
    """Ask the LLM to explain (not compute) the already-verified steps."""
    if not steps:
        return steps
    steps_text = "\n".join(f"{i+1}. {s.description}: {s.expression}" for i, s in enumerate(steps))
    raw = client.chat(
        system="You explain math steps clearly and concisely for a student.",
        user=NARRATION_PROMPT.format(target=model.solve_for, steps=steps_text),
        temperature=settings.temperature_narration,
        json_mode=False,
    )
    try:
        import json
        text = raw.strip().strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        start, end = text.find("["), text.rfind("]")
        explanations = json.loads(text[start:end + 1])
        for step, expl in zip(steps, explanations):
            step.explanation = expl
    except Exception:  # noqa: BLE001
        pass  # narration is a nice-to-have; steps remain valid without it
    return steps
