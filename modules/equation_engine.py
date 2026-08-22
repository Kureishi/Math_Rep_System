"""
Turns a natural-language problem statement into structured, symbolic
math: variables, equations/expressions/inequalities/ODEs, and a
derivation narrative.

The LLM proposes; SymPy disposes. This module only handles the
"propose" half -- structured extraction. Verification lives in verifier.py,
solving in solver.py.

Three kinds of relation are supported, each parsed differently:
  - "equation":   a normal algebraic equality, parsed via sp.Eq
  - "inequality": a constraint like "v <= 25", parsed as a raw SymPy
                   Relational (no Eq wrapping)
  - "ode":        a differential equation relating a declared function
                   (e.g. y(t)) to its derivative(s), parsed with the
                   function names bound to sp.Function(...) instead of
                   sp.Symbol(...) so "Derivative(y(t), t)" parses correctly
"""
from dataclasses import dataclass, field
import sympy as sp
from sympy.core.relational import Relational
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application,
    convert_xor,
)

from modules.llm_client import LMStudioClient, extract_json

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

EXTRACTION_SYSTEM_PROMPT = """You are a rigorous applied mathematician. Given a problem \
statement, you convert it into symbolic math. Respond ONLY with a JSON object, no prose, \
matching this exact schema:

{
  "problem_domain": "short label, e.g. 'kinematics', 'compound interest', 'population dynamics'",
  "problem_type": "'algebraic' for ordinary equations/inequalities (the common case), or 'ode' if the problem is fundamentally about a rate of change (population growth, cooling, decay, RC circuits, etc.) and needs a differential equation",
  "independent_variable": "only if problem_type is 'ode': the variable other quantities are changing with respect to, e.g. 't'. null otherwise",
  "variables": [
    {"symbol": "v", "meaning": "velocity", "known_value": "20", "unit": "m/s", "is_function": false}
  ],
  "equations": [
    {
      "name": "short label",
      "kind": "'equation' (default) | 'inequality' | 'ode'",
      "expression": "see rules below for exact syntax per kind",
      "derivation": "2-4 sentences explaining how this follows from the problem text"
    }
  ],
  "initial_conditions": [
    {"expression": "y(0)", "value": "1000", "note": "only used when problem_type is 'ode' -- initial/boundary values like y(0)=1000"}
  ],
  "solve_for": ["symbols/functions the problem asks for, e.g. [\\"a\\", \\"d\\"] or [\\"y\\"] for an ODE's unknown function"],
  "assumptions": ["list any assumptions you had to make"]
}

Rules by equation kind:
- "equation": use sympy.Eq(lhs, rhs) syntax, e.g. "Eq(F, m*a)".
- "inequality": write the raw comparison directly, e.g. "v <= 25" or "x**2 + y**2 < 100" -- do NOT wrap in Eq().
- "ode": mark the unknown function's variable entry with "is_function": true (its "unit" still applies to
  the function's output, e.g. y is population so unit="people"). Write the equation using
  Eq(Derivative(y(t), t), ...) where y is the function name and t is independent_variable. Every
  place y appears, write it as y(t) (applied), not bare y. Non-function variables (like a rate
  constant k) still use "is_function": false and are plain symbols.

General rules:
- known_value should be a plain number string if given in the problem, else null.
- Every symbol/function used in "equations" must appear in "variables".
- solve_for must be a JSON array of single names (never a comma-joined string like "a, d").
- If the problem has multiple valid equations/constraints (e.g. a system), include all of them,
  each with its own correct "kind".
- Do not solve anything here -- extraction only.
- Most problems are plain "equation" kind with problem_type "algebraic" -- only use "inequality"
  or "ode" when the problem genuinely calls for a constraint/threshold or a rate-of-change relationship.
"""

VERIFY_RETRY_SYSTEM_SUFFIX = """
Your previous attempt failed a consistency check:
{failure_reason}
Revise your JSON to fix this. Respond ONLY with the corrected JSON object.
"""

VALID_KINDS = {"equation", "inequality", "ode"}


@dataclass
class Variable:
    symbol: str
    meaning: str
    known_value: float | None
    unit: str | None
    is_function: bool = False


@dataclass
class InitialCondition:
    raw_expression: str
    value: float
    sympy_eq: sp.Eq | None = None
    parse_error: str | None = None


@dataclass
class Equation:
    name: str
    raw_expression: str
    derivation: str
    kind: str = "equation"
    sympy_eq: sp.Basic | None = None  # sp.Eq for "equation"/"ode", a Relational for "inequality"
    parse_error: str | None = None


@dataclass
class ProblemModel:
    problem_domain: str
    variables: list[Variable]
    equations: list[Equation]
    solve_for: list[str]
    assumptions: list[str]
    problem_type: str = "algebraic"  # "algebraic" | "ode"
    independent_variable: str | None = None
    initial_conditions: list[InitialCondition] = field(default_factory=list)
    raw_json: dict = field(default_factory=dict)


def _normalize_solve_for(raw) -> list[str]:
    """Tolerates a model returning a single string, a comma-joined string
    ("a, d"), a list, or null -- always returns a clean list of bare symbol
    names."""
    if raw is None:
        return []
    if isinstance(raw, str):
        return [s.strip() for s in raw.split(",") if s.strip()]
    if isinstance(raw, list):
        out = []
        for item in raw:
            out.extend(_normalize_solve_for(item))
        return out
    return []


def _local_dict(variables: list[Variable]) -> dict:
    """Functions declared is_function=True are bound to sp.Function(name)
    instead of sp.Symbol(name), so "y(t)" and "Derivative(y(t), t)" parse
    as function application/differentiation rather than raising."""
    local = {}
    for v in variables:
        local[v.symbol] = sp.Function(v.symbol) if v.is_function else sp.Symbol(v.symbol)
    local["Eq"] = sp.Eq
    local["Derivative"] = sp.Derivative
    local["diff"] = sp.Derivative
    return local


def _parse_equation(raw: str, kind: str, local_dict: dict) -> tuple[sp.Basic | None, str | None]:
    try:
        expr = parse_expr(raw, local_dict=local_dict, transformations=TRANSFORMS, evaluate=False)
    except Exception as e:  # noqa: BLE001
        return None, str(e)

    if kind == "inequality":
        if not isinstance(expr, Relational) or isinstance(expr, sp.Eq):
            return None, (f"expected a comparison like 'x <= 5' for an inequality, got: {raw}")
        return expr, None

    # "equation" and "ode" both want an sp.Eq
    if isinstance(expr, sp.Eq):
        return expr, None
    return sp.Eq(expr, 0), None  # bare expression means "expression = 0"


def _parse_initial_condition(raw_expr: str, value: float, local_dict: dict) -> InitialCondition:
    try:
        lhs = parse_expr(raw_expr, local_dict=local_dict, transformations=TRANSFORMS, evaluate=False)
        return InitialCondition(raw_expression=raw_expr, value=value, sympy_eq=sp.Eq(lhs, value))
    except Exception as e:  # noqa: BLE001
        return InitialCondition(raw_expression=raw_expr, value=value, parse_error=str(e))


def symbols_and_functions_used(eq: Equation) -> set[str]:
    """Names of both plain symbols AND applied-function names (e.g. 'y' for
    y(t)) referenced in an equation -- needed because solve_for may name
    either kind, and free_symbols alone misses function names."""
    if eq.sympy_eq is None:
        return set()
    names = {s.name for s in eq.sympy_eq.free_symbols}
    names |= {str(f.func) for f in eq.sympy_eq.atoms(AppliedUndef)}
    return names


def target_kind(model: ProblemModel, target_name: str) -> str:
    """Which kind of relation actually defines a solve_for target:
    'ode' if it's a declared function, 'equation' if it appears in any
    equation-kind relation, 'inequality' if it only appears in
    inequality-kind relations, otherwise falls back to 'equation'."""
    var = next((v for v in model.variables if v.symbol == target_name), None)
    if var and var.is_function:
        return "ode"
    in_equation = any(target_name in symbols_and_functions_used(e)
                       for e in model.equations if e.kind == "equation")
    if in_equation:
        return "equation"
    in_inequality = any(target_name in symbols_and_functions_used(e)
                         for e in model.equations if e.kind == "inequality")
    if in_inequality:
        return "inequality"
    return "equation"


def build_model(json_payload: dict) -> ProblemModel:
    variables = [
        Variable(
            symbol=v["symbol"],
            meaning=v.get("meaning", ""),
            known_value=(float(v["known_value"]) if v.get("known_value") not in (None, "") else None),
            unit=v.get("unit"),
            is_function=bool(v.get("is_function", False)),
        )
        for v in json_payload.get("variables", [])
    ]
    local_dict = _local_dict(variables)

    equations = []
    for eq in json_payload.get("equations", []):
        kind = eq.get("kind", "equation")
        if kind not in VALID_KINDS:
            kind = "equation"
        sympy_eq, err = _parse_equation(eq["expression"], kind, local_dict)
        equations.append(Equation(
            name=eq.get("name", "equation"),
            raw_expression=eq["expression"],
            derivation=eq.get("derivation", ""),
            kind=kind,
            sympy_eq=sympy_eq,
            parse_error=err,
        ))

    initial_conditions = [
        _parse_initial_condition(ic["expression"], float(ic["value"]), local_dict)
        for ic in json_payload.get("initial_conditions", []) or []
        if ic.get("expression") is not None and ic.get("value") not in (None, "")
    ]

    problem_type = json_payload.get("problem_type", "algebraic")
    if problem_type not in ("algebraic", "ode"):
        problem_type = "algebraic"

    return ProblemModel(
        problem_domain=json_payload.get("problem_domain", "unspecified"),
        variables=variables,
        equations=equations,
        solve_for=_normalize_solve_for(json_payload.get("solve_for")),
        assumptions=json_payload.get("assumptions", []),
        problem_type=problem_type,
        independent_variable=json_payload.get("independent_variable"),
        initial_conditions=initial_conditions,
        raw_json=json_payload,
    )


def extract_model(client: LMStudioClient, problem_text: str, retry_reason: str | None = None,
                   known_context: str | None = None) -> ProblemModel:
    """One LLM round trip: problem text -> structured symbolic model.

    known_context: optional text listing values already available from the
    variable workspace (previously solved results), so the model can treat
    e.g. "using d from the workspace" as a known numeric input rather than
    an undefined reference it has to guess at or ask about.
    """
    system = EXTRACTION_SYSTEM_PROMPT
    if retry_reason:
        system += VERIFY_RETRY_SYSTEM_SUFFIX.format(failure_reason=retry_reason)

    user = problem_text
    if known_context:
        user += (
            "\n\nThe following values are already known from earlier calculations. "
            "If the problem references one of them (by name or by clear description), "
            "treat it as a known_value input rather than an unknown:\n" + known_context
        )

    from config import settings
    raw = client.chat(system=system, user=user,
                       temperature=settings.temperature_extraction, json_mode=True)
    payload = extract_json(raw)
    return build_model(payload)
