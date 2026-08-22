"""
Produces a step-by-step solution, dispatching by equation kind:
  - "equation":   substitute knowns -> isolate -> simplify (existing logic)
  - "inequality": substitute knowns -> sp.reduce_inequalities -> solution set
  - "ode":        sp.dsolve for the general solution, then apply any initial
                   conditions to get the particular solution

Approach throughout: SymPy computes the actual solve steps we can trust.
The LLM is then given those verified intermediate results and asked only
to narrate/explain them in plain language -- it is not trusted to invent
the math itself here, only to explain math that has already been checked.
"""
from dataclasses import dataclass
import sympy as sp
from sympy.core.function import AppliedUndef

from config import settings
from modules.llm_client import LMStudioClient, extract_json
from modules.equation_engine import ProblemModel, Equation, target_kind, symbols_and_functions_used
from modules.verifier import _known_substitutions  # reuse the same substitution logic
from modules.ode_utils import solve_ode

NARRATION_PROMPT = """Given this verified sequence of steps solving for {target}, \
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


# ---------------------------------------------------------------- algebraic

def _algebraic_steps_for_target(model: ProblemModel, target_name: str, subs: dict) -> list[SolutionStep]:
    target = sp.Symbol(target_name)
    eq_objs = [e for e in model.equations if e.kind == "equation" and e.sympy_eq is not None]
    orig_eqs = [e.sympy_eq for e in eq_objs]
    steps: list[SolutionStep] = []
    if not orig_eqs:
        return steps

    for e, orig in zip(orig_eqs, eq_objs):
        steps.append(SolutionStep(description=f"Start from: {orig.name}", expression=sp.latex(e)))

    eqs = orig_eqs
    if subs:
        substituted = [e.subs(subs) for e in orig_eqs]
        readable = ", ".join(f"{k} = {v}" for k, v in subs.items())
        for orig_eq, sub_eq in zip(orig_eqs, substituted):
            if sub_eq != orig_eq:
                steps.append(SolutionStep(description=f"Substitute known values ({readable})",
                                            expression=sp.latex(sub_eq)))
        eqs = substituted

    # solve the whole (substituted) system simultaneously so that coupled
    # targets (e.g. 'd' depending on an also-unknown 'a') resolve correctly
    other_targets = [sp.Symbol(t) for t in model.solve_for
                      if t != target_name and target_kind(model, t) == "equation"]
    try:
        solutions = sp.solve(eqs, [target, *other_targets], dict=True)
    except Exception:  # noqa: BLE001
        solutions = []

    if solutions and target in solutions[0]:
        result = sp.simplify(solutions[0][target])
        steps.append(SolutionStep(description=f"Isolate and simplify to solve for {target_name}",
                                    expression=f"{target_name} = {sp.latex(result)}"))
        if result.is_number:
            steps.append(SolutionStep(description="Numeric result",
                                        expression=f"{target_name} = {sp.N(result, 6)}"))
    return steps


# ---------------------------------------------------------------- inequality

def _inequality_steps_for_target(model: ProblemModel, target_name: str, subs: dict) -> list[SolutionStep]:
    target = sp.Symbol(target_name)
    relevant = [e for e in model.equations
                if e.kind == "inequality" and e.sympy_eq is not None
                and target_name in symbols_and_functions_used(e)]
    steps: list[SolutionStep] = []
    if not relevant:
        return steps

    for e in relevant:
        steps.append(SolutionStep(description=f"Start from constraint: {e.name}",
                                    expression=sp.latex(e.sympy_eq)))

    substituted = [e.sympy_eq.subs(subs) for e in relevant]
    if subs:
        readable = ", ".join(f"{k} = {v}" for k, v in subs.items())
        for orig, sub in zip(relevant, substituted):
            if sub != orig.sympy_eq:
                steps.append(SolutionStep(description=f"Substitute known values ({readable})",
                                            expression=sp.latex(sub)))

    try:
        solution = sp.reduce_inequalities(substituted, [target])
        steps.append(SolutionStep(description=f"Solve the constraint(s) for {target_name}",
                                    expression=sp.latex(solution)))
    except Exception as e:  # noqa: BLE001
        steps.append(SolutionStep(
            description="Could not automatically solve this constraint",
            expression=str(e),
        ))
    return steps


# ---------------------------------------------------------------- ODE

def _ode_steps_for_target(model: ProblemModel, target_name: str,
                            particular_solution: sp.Eq) -> list[SolutionStep]:
    ode_eq = next((e for e in model.equations
                    if e.kind == "ode" and e.sympy_eq is not None
                    and target_name in symbols_and_functions_used(e)), None)
    steps: list[SolutionStep] = []
    if ode_eq is None:
        return steps

    steps.append(SolutionStep(description="State the differential equation",
                                expression=sp.latex(ode_eq.sympy_eq)))

    func_applied = next(iter(ode_eq.sympy_eq.atoms(AppliedUndef)))
    try:
        general_sol = sp.dsolve(ode_eq.sympy_eq, func_applied)
    except Exception:  # noqa: BLE001
        general_sol = None

    if general_sol is not None and general_sol != particular_solution:
        steps.append(SolutionStep(description="Solve for the general solution",
                                    expression=sp.latex(general_sol)))
        matching_ics = [
            ic for ic in model.initial_conditions
            if ic.sympy_eq is not None and ic.sympy_eq.lhs.atoms(AppliedUndef)
            and str(next(iter(ic.sympy_eq.lhs.atoms(AppliedUndef))).func) == target_name
        ]
        if matching_ics:
            # use the plain raw_expression (e.g. "N(0)") for the human-readable
            # description text, not sp.latex() -- that emits LaTeX control
            # sequences like "N{\left(0 \right)}" which look broken as plain text
            ic_text = ", ".join(f"{ic.raw_expression} = {ic.value:g}" for ic in matching_ics)
            steps.append(SolutionStep(description=f"Apply initial condition(s): {ic_text}",
                                        expression=sp.latex(particular_solution)))
    else:
        steps.append(SolutionStep(description="Solve the differential equation",
                                    expression=sp.latex(particular_solution)))
    return steps


# ---------------------------------------------------------------- dispatch

def compute_steps(model: ProblemModel) -> dict[str, list[SolutionStep]]:
    """Deterministic SymPy trace per target, dispatched by what kind of
    relation actually defines that target (equation / inequality / ode).
    Returns a dict keyed by target name since a problem may ask for more
    than one quantity."""
    if not model.solve_for:
        return {}

    subs = _known_substitutions(model)
    ode_solutions = solve_ode(model)
    all_steps: dict[str, list[SolutionStep]] = {}

    for target_name in model.solve_for:
        kind = target_kind(model, target_name)
        if kind == "ode" and target_name in ode_solutions:
            all_steps[target_name] = _ode_steps_for_target(model, target_name, ode_solutions[target_name])
        elif kind == "inequality":
            all_steps[target_name] = _inequality_steps_for_target(model, target_name, subs)
        else:
            all_steps[target_name] = _algebraic_steps_for_target(model, target_name, subs)

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
