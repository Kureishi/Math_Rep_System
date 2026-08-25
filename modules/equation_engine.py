"""
Turns a natural-language problem statement into structured, symbolic
math: variables, equations/expressions/inequalities/ODEs/recurrences, an
optional optimization objective, and a derivation narrative.

The LLM proposes; SymPy disposes. This module only handles the
"propose" half -- structured extraction. Verification lives in verifier.py,
solving in solver.py.

Kinds of relation supported, each parsed differently:
  - "equation":   a normal algebraic equality, parsed via sp.Eq. May
                   contain sp.Piecewise for conditional/tiered relations
                   (tax brackets, tiered pricing) -- no separate kind
                   needed since Piecewise composes with the normal
                   equation machinery.
  - "inequality": a constraint like "v <= 25", parsed as a raw SymPy
                   Relational (no Eq wrapping)
  - "ode":        a differential equation relating a declared function
                   (e.g. y(t)) to its derivative(s), parsed with the
                   function names bound to sp.Function(...) so
                   "Derivative(y(t), t)" parses correctly
  - "recurrence": a difference equation relating a declared function to
                   shifted versions of itself (e.g. a(n+1) = a(n) + 5),
                   using the same function-binding as "ode" but solved
                   via rsolve instead of dsolve

A problem can also carry an "objective" (see the Objective dataclass) for
optimization problems -- this is orthogonal to the equation kinds above,
since an optimization problem's constraints are just ordinary equation-
or inequality-kind relations.
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
from modules.vector_utils import vector_local_dict, make_vector

TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

EXTRACTION_SYSTEM_PROMPT = """You are a rigorous applied mathematician. Given a problem \
statement, you convert it into symbolic math. Respond ONLY with a JSON object, no prose, \
matching this exact schema:

{
  "problem_domain": "short label, e.g. 'kinematics', 'compound interest', 'population dynamics'",
  "problem_type": "'algebraic' (the common case), 'ode' if the problem needs a differential equation (rate of change), or 'recurrence' if it needs a difference equation (discrete step-by-step process)",
  "independent_variable": "only if problem_type is 'ode' or 'recurrence': the variable other quantities depend on, e.g. 't' or 'n'. null otherwise",
  "variables": [
    {"symbol": "v", "meaning": "velocity", "known_value": "20", "unit": "m/s", "is_function": false},
    {"symbol": "F", "meaning": "applied force", "known_value": null, "unit": "N",
     "is_vector": true, "components": ["Fx", "Fy"]}
  ],
  "equations": [
    {
      "name": "short label",
      "kind": "'equation' (default) | 'inequality' | 'ode' | 'recurrence'",
      "expression": "see rules below for exact syntax per kind",
      "derivation": "2-4 sentences explaining how this follows from the problem text"
    }
  ],
  "initial_conditions": [
    {"expression": "y(0)", "value": "1000", "note": "for 'ode'/'recurrence' problems -- initial values like y(0)=1000 or a(0)=100"}
  ],
  "objective": {
    "expression": "2*pi*r**2 + 2*pi*r*h",
    "direction": "'minimize' or 'maximize'",
    "optimize_over": ["which symbol(s) the problem wants the optimal value of, e.g. [\\"r\\"]"],
    "note": "ONLY include this field if the problem genuinely asks to minimize/maximize/optimize something. null/omit otherwise."
  },
  "solve_for": ["symbols/functions the problem actually asks for, e.g. [\\"a\\", \\"d\\"], [\\"y\\"] for an ODE, or the optimize_over variable(s) for an optimization problem"],
  "assumptions": ["list any assumptions you had to make"]
}

Rules by equation kind:
- "equation": use sympy.Eq(lhs, rhs) syntax, e.g. "Eq(F, m*a)". Can use Piecewise for tiered/
  conditional relations, e.g. "Eq(tax, Piecewise((0.1*x, x <= 10000), (0.2*x - 1000, True)))" --
  the LAST condition should usually be "True" to cover the remaining case (like an "else" branch).
- "inequality": write the raw comparison directly, e.g. "v <= 25" or "x**2 + y**2 < 100" -- do NOT wrap in Eq().
- "ode": mark the unknown function's variable entry with "is_function": true. Write using
  Eq(Derivative(y(t), t), ...) where y is the function name and t is independent_variable. Every
  place y appears, write it as y(t) (applied), not bare y.
- "recurrence": mark the unknown function's variable entry with "is_function": true. Write using
  the SAME applied-function style, e.g. Eq(a(n+1), a(n) + 5) or Eq(a(n+2), a(n+1) + a(n)) for a
  second-order recurrence -- no Derivative() involved, just shifted integer arguments.

<<<<<<< HEAD
=======
<<<<<<< HEAD
>>>>>>> e3ab651c15f0ca938979b7c9a06c3ea6fb7febce
Vector quantities (forces, displacements, velocities in 2D/3D, torque, work):
- Mark the variable "is_vector": true and give "components": ["Fx", "Fy"] (2D) or
  ["Fx", "Fy", "Fz"] (3D) -- each component name must ALSO be listed as its own ordinary
  (non-vector) variable entry with its own known_value/unit, exactly like any scalar.
- Do NOT pre-decompose a physical vector into "the x-component of the force" as a separate
  scalar concept from "the force" -- declare the vector itself (F) plus its components
  (Fx, Fy[, Fz]), then use F directly in equations via the helper functions below.
- In "equations", use these functions on vector variables (they always reduce to a scalar
  or another vector, never left symbolic as "Dot(...)"): dot(u, v) -> scalar dot product;
  cross(u, v) -> scalar (2D) or vector (3D) cross product; magnitude(v) / norm(v) -> scalar
  length; unit(v) -> unit vector; angle_between(u, v) / angle_between_deg(u, v) -> angle in
  radians/degrees; Vector(a, b) or Vector(a, b, c) -> build an inline vector from components;
  distance(a, b) -> Euclidean distance between two points or two vectors; Point(x, y[, z]) ->
  a geometry point for use with distance(). Example: work done, "Eq(W, dot(F, d))" where F and
  d are declared as vectors with components Fx, Fy, dx, dy.
- A vector variable's own "known_value" should be null -- only its components carry numbers.
- "solve_for" must never name a vector variable itself (e.g. "F") -- it can only be solved
  via sp.solve as a scalar. Instead solve for a scalar derived from it (a component like "Fx",
  or a scalar equation's own LHS symbol, e.g. "Eq(F_mag, magnitude(F))" then solve_for ["F_mag"]).

<<<<<<< HEAD
=======
=======
>>>>>>> 06e8a1bf8422d1758550ad4cd55c73cf5c90bff6
>>>>>>> e3ab651c15f0ca938979b7c9a06c3ea6fb7febce
Objective/optimization rules:
- Only include "objective" when the problem asks to minimize, maximize, or find an optimal value.
- "optimize_over" lists the variable(s) being solved for at the optimum -- other symbols in the
  objective expression should either have a known_value, or be eliminable via an "equation"-kind
  constraint also present in "equations" (e.g. a fixed-volume constraint used to eliminate height
  when minimizing surface area in terms of radius alone).
- Constraints for an optimization problem are just ordinary "equation"/"inequality" entries in
  "equations" -- don't duplicate them inside "objective".

General rules:
- known_value should be a plain number string if given in the problem, else null.
- Every symbol/function used in "equations" or "objective" must appear in "variables".
- solve_for must be a JSON array of single names (never a comma-joined string like "a, d").
- If the problem has multiple valid equations/constraints (e.g. a system), include all of them,
  each with its own correct "kind".
- Do not solve anything here -- extraction only.
- Most problems are plain "equation" kind with problem_type "algebraic" -- only use "inequality",
  "ode", "recurrence", or "objective" when the problem genuinely calls for that structure.
"""

VERIFY_RETRY_SYSTEM_SUFFIX = """
Your previous attempt failed a consistency check:
{failure_reason}
Revise your JSON to fix this. Respond ONLY with the corrected JSON object.
"""

VALID_KINDS = {"equation", "inequality", "ode", "recurrence"}


@dataclass
class Variable:
    symbol: str
    meaning: str
    known_value: float | None
    unit: str | None
    is_function: bool = False
    is_vector: bool = False
    components: list[str] = field(default_factory=list)  # only meaningful if is_vector


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
    sympy_eq: sp.Basic | None = None  # sp.Eq for "equation"/"ode"/"recurrence", a Relational for "inequality"
    parse_error: str | None = None


@dataclass
class Objective:
    raw_expression: str
    direction: str  # "minimize" | "maximize"
    optimize_over: list[str]
    sympy_expr: sp.Expr | None = None
    parse_error: str | None = None


@dataclass
class ProblemModel:
    problem_domain: str
    variables: list[Variable]
    equations: list[Equation]
    solve_for: list[str]
    assumptions: list[str]
    problem_type: str = "algebraic"  # "algebraic" | "ode" | "recurrence"
    independent_variable: str | None = None
    initial_conditions: list[InitialCondition] = field(default_factory=list)
    objective: Objective | None = None
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
    instead of sp.Symbol(name), so "y(t)" and "Derivative(y(t), t)" (or
    "a(n+1)" for a recurrence) parse as function application rather than
    raising. Piecewise is always available so tiered/conditional equations
    can be expressed without needing a separate equation kind."""
    local = {}
    for v in variables:
        if v.is_function:
            local[v.symbol] = sp.Function(v.symbol)
        elif v.is_vector and v.components:
            # bind the vector NAME to an actual column Matrix of its
            # component symbols, so dot(F, d) etc. operate on genuine
            # vectors rather than the LLM having to hand-decompose them
            local[v.symbol] = make_vector([sp.Symbol(c) for c in v.components])
        else:
            local[v.symbol] = sp.Symbol(v.symbol)
    local["Eq"] = sp.Eq
    local["Derivative"] = sp.Derivative
    local["diff"] = sp.Derivative
    local["Piecewise"] = sp.Piecewise
<<<<<<< HEAD
    local.update(vector_local_dict())
=======
<<<<<<< HEAD
    local.update(vector_local_dict())
=======
>>>>>>> 06e8a1bf8422d1758550ad4cd55c73cf5c90bff6
>>>>>>> e3ab651c15f0ca938979b7c9a06c3ea6fb7febce
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

    # "equation", "ode", and "recurrence" all want an sp.Eq
    if isinstance(expr, sp.Eq):
        return expr, None
    return sp.Eq(expr, 0), None  # bare expression means "expression = 0"


def _parse_initial_condition(raw_expr: str, value: float, local_dict: dict) -> InitialCondition:
    try:
        lhs = parse_expr(raw_expr, local_dict=local_dict, transformations=TRANSFORMS, evaluate=False)
        return InitialCondition(raw_expression=raw_expr, value=value, sympy_eq=sp.Eq(lhs, value))
    except Exception as e:  # noqa: BLE001
        return InitialCondition(raw_expression=raw_expr, value=value, parse_error=str(e))


def _parse_objective(raw_obj: dict | None, local_dict: dict) -> Objective | None:
    if not raw_obj or not raw_obj.get("expression"):
        return None
    direction = raw_obj.get("direction", "minimize")
    if direction not in ("minimize", "maximize"):
        direction = "minimize"
    optimize_over = raw_obj.get("optimize_over") or []
    if isinstance(optimize_over, str):
        optimize_over = [s.strip() for s in optimize_over.split(",") if s.strip()]
    try:
        expr = parse_expr(raw_obj["expression"], local_dict=local_dict,
                           transformations=TRANSFORMS, evaluate=False)
        return Objective(raw_expression=raw_obj["expression"], direction=direction,
                          optimize_over=optimize_over, sympy_expr=expr)
    except Exception as e:  # noqa: BLE001
        return Objective(raw_expression=raw_obj["expression"], direction=direction,
                          optimize_over=optimize_over, parse_error=str(e))


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
    'optimization' if it's an objective's optimize_over variable, 'ode' or
    'recurrence' if it's a declared function (disambiguated by which kind
    of equation actually uses it), 'equation' if it appears in any
    equation-kind relation, 'inequality' if it only appears in
    inequality-kind relations, otherwise falls back to 'equation'."""
    if model.objective and target_name in model.objective.optimize_over:
        return "optimization"

    var = next((v for v in model.variables if v.symbol == target_name), None)
    if var and var.is_function:
        if any(target_name in symbols_and_functions_used(e)
               for e in model.equations if e.kind == "recurrence"):
            return "recurrence"
        return "ode"  # default for a declared function, including the "ode" kind itself

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
            is_vector=bool(v.get("is_vector", False)),
            components=list(v.get("components") or []),
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
    if problem_type not in ("algebraic", "ode", "recurrence"):
        problem_type = "algebraic"

    objective = _parse_objective(json_payload.get("objective"), local_dict)

    return ProblemModel(
        problem_domain=json_payload.get("problem_domain", "unspecified"),
        variables=variables,
        equations=equations,
        solve_for=_normalize_solve_for(json_payload.get("solve_for")),
        assumptions=json_payload.get("assumptions", []),
        problem_type=problem_type,
        independent_variable=json_payload.get("independent_variable"),
        initial_conditions=initial_conditions,
        objective=objective,
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
