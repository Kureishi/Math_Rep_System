"""
Turns a natural-language problem statement into structured, symbolic
math: variables, equations/expressions, and a derivation narrative.

The LLM proposes; SymPy disposes. This module only handles the
"propose" half -- structured extraction. Verification lives in verifier.py.
"""
from dataclasses import dataclass, field
import sympy as sp
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
  "problem_domain": "short label, e.g. 'kinematics', 'compound interest', 'related rates'",
  "variables": [
    {"symbol": "v", "meaning": "velocity", "known_value": "20", "unit": "m/s"}
  ],
  "equations": [
    {
      "name": "short label",
      "expression": "valid sympy-parseable string, e.g. 'v**2 - u**2 - 2*a*x' meaning 'v**2 = u**2 + 2*a*x' rearranged to equal 0, OR 'Eq(v, u + a*t)' using sympy's Eq()",
      "derivation": "2-4 sentences explaining how this follows from the problem text, referencing the specific quantities given"
    }
  ],
  "solve_for": ["list every symbol the problem actually asks for -- often just one, but include all of them if the problem asks multiple questions, e.g. [\\"a\\", \\"d\\"]. Empty list if it's only asking to model the situation, not compute a value."],
  "assumptions": ["list any assumptions you had to make, e.g. 'ignoring air resistance'"]
}

Rules:
- Use sympy.Eq(lhs, rhs) syntax for equations, e.g. "Eq(F, m*a)".
- known_value should be a plain number string if given in the problem, else null.
- Every symbol used in "equations" must appear in "variables".
- solve_for must be a JSON array of single symbol names (never a comma-joined string like "a, d" -- use ["a", "d"]).
- If the problem has multiple valid equations (e.g. a system), include all of them.
- Do not solve the equation here -- extraction only.
"""

VERIFY_RETRY_SYSTEM_SUFFIX = """
Your previous attempt failed a consistency check:
{failure_reason}
Revise your JSON to fix this. Respond ONLY with the corrected JSON object.
"""


@dataclass
class Variable:
    symbol: str
    meaning: str
    known_value: float | None
    unit: str | None


@dataclass
class Equation:
    name: str
    raw_expression: str
    derivation: str
    sympy_eq: sp.Eq | None = None
    parse_error: str | None = None


@dataclass
class ProblemModel:
    problem_domain: str
    variables: list[Variable]
    equations: list[Equation]
    solve_for: list[str]
    assumptions: list[str]
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
    return {v.symbol: sp.Symbol(v.symbol) for v in variables}


def _parse_equation(raw: str, local_dict: dict) -> tuple[sp.Eq | None, str | None]:
    try:
        expr = parse_expr(raw, local_dict=local_dict | {"Eq": sp.Eq},
                           transformations=TRANSFORMS, evaluate=False)
        if isinstance(expr, sp.Eq):
            return expr, None
        # bare expression meaning "expression = 0"
        return sp.Eq(expr, 0), None
    except Exception as e:  # noqa: BLE001
        return None, str(e)


def build_model(json_payload: dict) -> ProblemModel:
    variables = [
        Variable(
            symbol=v["symbol"],
            meaning=v.get("meaning", ""),
            known_value=(float(v["known_value"]) if v.get("known_value") not in (None, "") else None),
            unit=v.get("unit"),
        )
        for v in json_payload.get("variables", [])
    ]
    local_dict = _local_dict(variables)

    equations = []
    for eq in json_payload.get("equations", []):
        sympy_eq, err = _parse_equation(eq["expression"], local_dict)
        equations.append(Equation(
            name=eq.get("name", "equation"),
            raw_expression=eq["expression"],
            derivation=eq.get("derivation", ""),
            sympy_eq=sympy_eq,
            parse_error=err,
        ))

    return ProblemModel(
        problem_domain=json_payload.get("problem_domain", "unspecified"),
        variables=variables,
        equations=equations,
        solve_for=_normalize_solve_for(json_payload.get("solve_for")),
        assumptions=json_payload.get("assumptions", []),
        raw_json=json_payload,
    )


def extract_model(client: LMStudioClient, problem_text: str, retry_reason: str | None = None) -> ProblemModel:
    """One LLM round trip: problem text -> structured symbolic model."""
    system = EXTRACTION_SYSTEM_PROMPT
    if retry_reason:
        system += VERIFY_RETRY_SYSTEM_SUFFIX.format(failure_reason=retry_reason)

    from config import settings
    raw = client.chat(system=system, user=problem_text,
                       temperature=settings.temperature_extraction, json_mode=True)
    payload = extract_json(raw)
    return build_model(payload)
