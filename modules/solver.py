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
from modules.llm_client import LMStudioClient, extract_json
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


def compute_steps(model: ProblemModel) -> dict[str, list[SolutionStep]]:
    """Deterministic SymPy trace per target: substitute knowns -> isolate ->
    simplify. Returns a dict keyed by target symbol since a problem may ask
    for more than one quantity (e.g. both acceleration and displacement)."""
    if not model.solve_for:
        return {}

    subs = _known_substitutions(model)
    orig_eqs = [e.sympy_eq for e in model.equations if e.sympy_eq is not None]
    if not orig_eqs:
        return {}

    all_steps: dict[str, list[SolutionStep]] = {}

    for target_name in model.solve_for:
        target = sp.Symbol(target_name)
        steps: list[SolutionStep] = []

        for e, orig in zip(orig_eqs, model.equations):
            steps.append(SolutionStep(
                description=f"Start from: {orig.name}",
                expression=sp.latex(e),
            ))

        eqs = orig_eqs
        if subs:
            substituted = [e.subs(subs) for e in orig_eqs]
            readable = ", ".join(f"{k} = {v}" for k, v in subs.items())
            for orig_eq, sub_eq in zip(orig_eqs, substituted):
                if sub_eq != orig_eq:
                    steps.append(SolutionStep(
                        description=f"Substitute known values ({readable})",
                        expression=sp.latex(sub_eq),
                    ))
            eqs = substituted

        # solve the whole (substituted) system simultaneously so that
        # coupled targets (e.g. 'd' depending on an also-unknown 'a' from
        # another equation) resolve correctly, not just this equation alone
        other_targets = [sp.Symbol(t) for t in model.solve_for if t != target_name]
        try:
            solutions = sp.solve(eqs, [target, *other_targets], dict=True)
        except Exception:  # noqa: BLE001
            solutions = []

        if solutions and target in solutions[0]:
            result = sp.simplify(solutions[0][target])
            steps.append(SolutionStep(
                description=f"Isolate and simplify to solve for {target_name}",
                expression=f"{target_name} = {sp.latex(result)}",
            ))
            if result.is_number:
                steps.append(SolutionStep(
                    description="Numeric result",
                    expression=f"{target_name} = {sp.N(result, 6)}",
                ))

        all_steps[target_name] = steps

    return all_steps


def narrate_steps(client: LMStudioClient, model: ProblemModel,
                   steps_by_target: dict[str, list[SolutionStep]]) -> dict[str, list[SolutionStep]]:
    """Ask the LLM to explain (not compute) each target's already-verified steps."""
    for target_name, steps in steps_by_target.items():
        if not steps:
            continue
        steps_text = "\n".join(f"{i+1}. {s.description}: {s.expression}" for i, s in enumerate(steps))
        raw = client.chat(
            system="You explain math steps clearly and concisely for a student.",
            user=NARRATION_PROMPT.format(target=target_name, steps=steps_text),
            temperature=settings.temperature_narration,
            json_mode=False,
        )
        try:
            explanations = extract_json(raw)
            if isinstance(explanations, list):
                for step, expl in zip(steps, explanations):
                    if isinstance(expl, str):
                        step.explanation = expl
        except Exception:  # noqa: BLE001
            pass  # narration is a nice-to-have; steps remain valid without it
    return steps_by_target
