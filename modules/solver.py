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
from modules.matrix_utils import linear_system_view
from modules.uncertainty import uncertainty_for_target
from modules.physical_validity import filter_physically_valid
from modules.algebra_rules import classify_isolation

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

    # if this is genuinely part of a coupled linear system (>=2 equations,
    # >=2 shared unknowns, and NOT solvable one-equation-at-a-time by
    # plain substitution -- see matrix_utils._is_sequentially_solvable),
    # show the explicit A x = b representation as its own step before
    # falling through to the ordinary sp.solve() below -- the numeric
    # answer is unaffected, this only makes the structure visible when
    # it's actually needed rather than whenever >=2 unknowns merely
    # coexist somewhere in the model
    matrix_result = linear_system_view(model, subs)
    if matrix_result is not None and target_name in matrix_result.symbols:
        A_latex = sp.latex(matrix_result.A)
        x_latex = sp.latex(sp.Matrix([sp.Symbol(s) for s in matrix_result.symbols]))
        b_latex = sp.latex(matrix_result.b)
        steps.append(SolutionStep(
            description="Represent the system as A x = b",
            expression=f"{A_latex} {x_latex} = {b_latex}",
        ))
        if matrix_result.is_square:
            steps.append(SolutionStep(
                description="Determinant of the coefficient matrix",
                expression=f"\\det(A) = {sp.latex(matrix_result.determinant)}",
            ))
        steps.append(SolutionStep(description="Classification", expression=matrix_result.classification))

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

    # sp.solve() returns however many branches the equation has (a
    # quadratic in the target gives two), in whatever order its internal
    # algorithm happens to produce -- NOT sorted by which one is
    # physically meaningful. Filtering by each unknown's declared domain
    # (Variable.domain) picks out the branch that's actually a sensible
    # answer instead of always taking branch [0] unconditionally, which
    # for e.g. a projectile time-of-flight equation can silently be a
    # negative time. See physical_validity.py.
    chosen = solutions[0] if solutions else None
    if len(solutions) > 1:
        filter_result = filter_physically_valid(model, solutions, [target, *other_targets])
        if filter_result.checked_any_domain and filter_result.discarded:
            for sol, reasons in filter_result.discarded:
                steps.append(SolutionStep(
                    description="Discard a non-physical root",
                    expression="; ".join(reasons),
                ))
            if len(filter_result.valid) == 1:
                chosen = filter_result.valid[0]
            elif filter_result.valid:
                # more than one branch still survives filtering -- can't
                # pick a single "the" answer without more info, so note
                # the ambiguity explicitly rather than silently guessing
                chosen = filter_result.valid[0]
                steps.append(SolutionStep(
                    description="Multiple physically valid solutions remain",
                    expression=f"{len(filter_result.valid)} branches still satisfy every declared "
                                "domain -- showing the first; the problem may be genuinely ambiguous.",
                ))
            # if filtering discarded EVERY branch, fall back to the
            # original solutions[0] (chosen already defaults to that) --
            # better to show a possibly-wrong-signed answer with the
            # discard reasoning visible than to show no answer at all

    if chosen is not None and target in chosen:
        # tag the isolation step with which algebraic TECHNIQUE it
        # required (linear/quadratic/root/inverse-function/etc.) --
        # classified structurally from whichever substituted equation
        # still contains the target, since sp.solve() doesn't expose an
        # internal step-by-step trace to read the technique back out of.
        target_eq = next((e for e in eqs if isinstance(e, sp.Eq) and target in e.free_symbols), None)
        if target_eq is not None:
            steps.append(SolutionStep(
                description="Technique",
                expression=classify_isolation(target_eq, target),
            ))

        result = sp.simplify(chosen[target])
        steps.append(SolutionStep(description=f"Isolate and simplify to solve for {target_name}",
                                    expression=f"{target_name} = {sp.latex(result)}"))
        if result.is_number:
            steps.append(SolutionStep(description="Numeric result",
                                        expression=f"{target_name} = {sp.N(result, 6)}"))

            # if any input this target depends on carries a stated
            # measurement uncertainty, propagate it (first-order error
            # propagation) and show it as one more step -- deliberately
            # only runs when the numeric result is already in hand, since
            # it re-solves the ORIGINAL (unsubstituted) system to get a
            # symbolic formula to differentiate; see uncertainty.py
            unc = uncertainty_for_target(model, target_name, subs)
            if unc is not None:
                rel_text = f" ({unc.relative_uncertainty:.2%})" if unc.relative_uncertainty is not None else ""
                steps.append(SolutionStep(
                    description="Propagate measurement uncertainty",
                    expression=f"{target_name} = {unc.nominal:.6g} \\pm {unc.uncertainty:.4g}{rel_text}",
                ))
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
    from modules.ode_utils import group_coupled_odes

    ode_eqs = [e for e in model.equations if e.kind == "ode" and e.sympy_eq is not None]
    ode_eq = next((e for e in ode_eqs if target_name in symbols_and_functions_used(e)), None)
    steps: list[SolutionStep] = []
    if ode_eq is None:
        return steps

    group = next((g for g in group_coupled_odes(ode_eqs) if ode_eq in g), [ode_eq])
    is_coupled = len(group) > 1

    if is_coupled:
        for e in group:
            steps.append(SolutionStep(description=f"State the differential equation: {e.name}",
                                        expression=sp.latex(e.sympy_eq)))
    else:
        steps.append(SolutionStep(description="State the differential equation",
                                    expression=sp.latex(ode_eq.sympy_eq)))

    func_applied = next(iter(ode_eq.sympy_eq.atoms(AppliedUndef)))

    if is_coupled:
        from sympy.solvers.ode.systems import dsolve_system
        applied_funcs = []
        seen = set()
        for e in group:
            for f in e.sympy_eq.atoms(AppliedUndef):
                if str(f.func) not in seen:
                    seen.add(str(f.func))
                    applied_funcs.append(f)
        t = next(iter(applied_funcs[0].args))
        try:
            general_sols = dsolve_system([e.sympy_eq for e in group], funcs=applied_funcs, t=t)[0]
        except Exception:  # noqa: BLE001
            general_sols = None

        if general_sols is not None:
            general_this_target = next((s for s in general_sols if str(s.lhs.func) == target_name), None)
            if general_this_target is not None:
                steps.append(SolutionStep(
                    description=f"Solve the coupled system for the general solution of {target_name}",
                    expression=sp.latex(general_this_target)))

        matching_ics = [ic for ic in model.initial_conditions
                         if ic.sympy_eq is not None and ic.sympy_eq.lhs.atoms(AppliedUndef)
                         and str(next(iter(ic.sympy_eq.lhs.atoms(AppliedUndef))).func) in seen]
        if matching_ics:
            ic_text = ", ".join(f"{ic.raw_expression} = {ic.value:g}" for ic in matching_ics)
            steps.append(SolutionStep(
                description=f"Apply initial condition(s) across the coupled system: {ic_text}",
                expression=sp.latex(particular_solution)))
        else:
            steps.append(SolutionStep(description=f"Particular solution for {target_name}",
                                        expression=sp.latex(particular_solution)))
        return steps

    # standalone (uncoupled) ODE -- unchanged from before
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

def _recurrence_steps_for_target(model: ProblemModel, target_name: str,
                                    closed_form: sp.Expr) -> list[SolutionStep]:
    from modules.recurrence_utils import _independent_variable

    rec_eq = next((e for e in model.equations
                    if e.kind == "recurrence" and e.sympy_eq is not None
                    and target_name in symbols_and_functions_used(e)), None)
    steps: list[SolutionStep] = []
    if rec_eq is None:
        return steps

    steps.append(SolutionStep(description="State the recurrence (difference equation)",
                                expression=sp.latex(rec_eq.sympy_eq)))

    matching_ics = [ic for ic in model.initial_conditions
                     if ic.sympy_eq is not None and ic.sympy_eq.lhs.atoms(AppliedUndef)
                     and str(next(iter(ic.sympy_eq.lhs.atoms(AppliedUndef))).func) == target_name]
    if matching_ics:
        ic_text = ", ".join(f"{ic.raw_expression} = {ic.value:g}" for ic in matching_ics)
        steps.append(SolutionStep(description=f"Solve via rsolve() with initial condition(s): {ic_text}",
                                    expression=f"{target_name}(n) = {sp.latex(closed_form)}"))
    else:
        steps.append(SolutionStep(description="Solve for the general closed-form solution",
                                    expression=f"{target_name}(n) = {sp.latex(closed_form)}"))
    return steps


def _optimization_steps_for_target(model: ProblemModel, target_name: str,
                                     opt_result) -> list[SolutionStep]:
    obj = model.objective
    steps: list[SolutionStep] = []
    if obj is None or not opt_result.critical_points:
        return steps

    direction_word = "Minimize" if obj.direction == "minimize" else "Maximize"
    steps.append(SolutionStep(description=f"{direction_word} the objective",
                                expression=sp.latex(obj.sympy_expr)))

    if opt_result.eliminated_vars:
        for var_name, expr in opt_result.eliminated_vars.items():
            steps.append(SolutionStep(
                description=f"Use a constraint to eliminate {var_name}",
                expression=f"{var_name} = {sp.latex(expr)}",
            ))
        if opt_result.reduced_objective is not None:
            steps.append(SolutionStep(description="Objective after substitution",
                                        expression=sp.latex(opt_result.reduced_objective)))

    if opt_result.used_lagrange:
        steps.append(SolutionStep(
            description="Form the Lagrangian and solve where its gradient is zero "
                        "(equality-constrained optimization)",
            expression=f"{target_name} = {sp.latex(opt_result.critical_points[0].get(target_name, '?'))}",
        ))
    else:
        steps.append(SolutionStep(
            description="Set the derivative(s) to zero and solve for the critical point",
            expression=f"{target_name} = {sp.latex(opt_result.critical_points[0].get(target_name, '?'))}",
        ))
        steps.append(SolutionStep(
            description="Classify via the second-derivative/Hessian test",
            expression=opt_result.classifications[0],
        ))

    return steps


def alternate_method_steps(model: ProblemModel, target_name: str, subs: dict) -> list[SolutionStep] | None:
    """A SECOND way to reach/check the same answer for target_name --
    shown only on explicit request (a "show me another way" toggle in
    the UI), never part of the default step list, since most problems
    don't need two derivations but seeing an independent path can help
    build confidence or teach a different technique.

    Always includes a back-substitution check: plug the already-solved
    numeric answer into each original equation referencing the target,
    using the SAME known values the primary path used, and confirm both
    sides agree. This works for any algebraic target.

    Additionally includes Cramer's rule -- x_i = det(A_i)/det(A), where
    A_i is the coefficient matrix with column i replaced by b -- when
    the target is part of a square linear system, EVEN if the default
    view decided plain substitution was simpler for it (build_linear_system
    is called with force=True here specifically to get that matrix
    regardless of the sequential-solvability heuristic that gates the
    default view).

    Returns None if target_name isn't an algebraic target, or no
    numeric answer for it could be established at all.
    """
    if target_kind(model, target_name) != "equation":
        return None

    target = sp.Symbol(target_name)
    orig_eqs = [e.sympy_eq for e in model.equations
                if e.kind == "equation" and e.sympy_eq is not None and target in e.sympy_eq.free_symbols]
    if not orig_eqs:
        return None

    all_eqs = [e.sympy_eq for e in model.equations if e.kind == "equation" and e.sympy_eq is not None]
    other_targets = [sp.Symbol(t) for t in model.solve_for
                      if t != target_name and target_kind(model, t) == "equation"]
    substituted = [e.subs(subs) for e in all_eqs]
    try:
        solutions = sp.solve(substituted, [target, *other_targets], dict=True)
    except Exception:  # noqa: BLE001
        return None
    if not solutions:
        return None

    chosen = solutions[0]
    if len(solutions) > 1:
        filter_result = filter_physically_valid(model, solutions, [target, *other_targets])
        if filter_result.checked_any_domain and filter_result.valid:
            chosen = filter_result.valid[0]
    if target not in chosen or not chosen[target].is_number:
        return None
    answer = sp.N(chosen[target], 6)

    steps: list[SolutionStep] = [SolutionStep(
        description="Alternate method: verify by back-substitution",
        expression=f"Plug {target_name} = {answer} back into the original equation(s).",
    )]
    for eq in orig_eqs:
        check_eq = eq.subs(subs).subs(target, answer)
        if check_eq is sp.true or check_eq is sp.false:
            # sp.Eq() of two pure numeric literals auto-evaluates straight
            # to a BooleanTrue/BooleanFalse (e.g. Eq(2, 2) -> True) rather
            # than staying an Eq object -- that IS the check result.
            eq_with_knowns = eq.subs(subs)
            steps.append(SolutionStep(
                description="Check -- matches" if bool(check_eq) else "Check -- MISMATCH",
                expression=f"{sp.latex(eq_with_knowns)} \\text{{ at }} {target_name} = {answer}",
            ))
            continue
        try:
            lhs_val, rhs_val = sp.N(check_eq.lhs, 6), sp.N(check_eq.rhs, 6)
            matches = abs(complex(lhs_val) - complex(rhs_val)) < 1e-4 * max(
                abs(complex(lhs_val)), abs(complex(rhs_val)), 1)
        except (TypeError, ValueError, AttributeError):
            continue
        steps.append(SolutionStep(
            description="Check -- matches" if matches else "Check -- MISMATCH",
            expression=f"{sp.latex(check_eq.lhs)} = {lhs_val}, \\quad {sp.latex(check_eq.rhs)} = {rhs_val}",
        ))

    matrix_result = linear_system_view(model, subs, force=True)
    if matrix_result is not None and matrix_result.is_square and target_name in matrix_result.symbols:
        idx = matrix_result.symbols.index(target_name)
        A_i = matrix_result.A.copy()
        A_i[:, idx] = matrix_result.b
        det_A = matrix_result.determinant
        det_Ai = A_i.det()
        steps.append(SolutionStep(
            description="Alternate method: Cramer's rule",
            expression=(f"{target_name} = \\dfrac{{\\det(A_{{{target_name}}})}}{{\\det(A)}} = "
                         f"\\dfrac{{{sp.latex(det_Ai)}}}{{{sp.latex(det_A)}}} = "
                         f"{sp.latex(sp.simplify(det_Ai / det_A)) if det_A != 0 else 'undefined (det(A) = 0)'}"),
        ))
    return steps


def compute_steps(model: ProblemModel) -> dict[str, list[SolutionStep]]:
    """Deterministic SymPy trace per target, dispatched by what kind of
    relation actually defines that target (equation / inequality / ode /
    recurrence / optimization). Returns a dict keyed by target name since
    a problem may ask for more than one quantity."""
    if not model.solve_for:
        return {}

    subs = _known_substitutions(model)
    ode_solutions = solve_ode(model)
    all_steps: dict[str, list[SolutionStep]] = {}

    recurrence_solutions = None
    opt_result = None

    for target_name in model.solve_for:
        kind = target_kind(model, target_name)
        if kind == "ode" and target_name in ode_solutions:
            all_steps[target_name] = _ode_steps_for_target(model, target_name, ode_solutions[target_name])
        elif kind == "recurrence":
            if recurrence_solutions is None:
                from modules.recurrence_utils import solve_recurrence
                recurrence_solutions = solve_recurrence(model)
            if target_name in recurrence_solutions:
                all_steps[target_name] = _recurrence_steps_for_target(
                    model, target_name, recurrence_solutions[target_name])
            else:
                all_steps[target_name] = []
        elif kind == "optimization":
            if opt_result is None:
                from modules.optimization_utils import solve_optimization
                opt_result = solve_optimization(model)
            if opt_result is not None and not opt_result.error:
                all_steps[target_name] = _optimization_steps_for_target(model, target_name, opt_result)
            else:
                all_steps[target_name] = []
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
        try:
            raw = client.chat(
                system="You explain math steps clearly and concisely for a student.",
                user=NARRATION_PROMPT.format(target=target_name, steps=steps_text),
                temperature=settings.temperature_narration,
                json_mode=False,
            )
            explanations = extract_json(raw)
            if isinstance(explanations, list):
                for step, expl in zip(steps, explanations):
                    if isinstance(expl, str):
                        step.explanation = expl
        except Exception:  # noqa: BLE001
            pass  # narration is a nice-to-have; steps remain valid without it (API errors included)
    return steps_by_target
