"""
Pydantic validation for the LLM's problem-extraction JSON, at the exact
point where equation_engine.extract_model() hands a raw dict (parsed
from the model's own JSON output by llm_client.extract_json()) to
build_model(). Before this, a malformed response -- a missing
"expression" field on an equation, say -- reached build_model() and
failed with a bare KeyError deep inside a list comprehension, with
nothing in the error pointing at WHICH equation or WHICH field was the
problem. Pydantic turns that into a field-level ValidationError
("equations.0.expression: Field required") at the one place in the
whole pipeline positioned to give a genuinely actionable message,
because it's checking a shape it actually knows in advance.

Deliberately permissive, not strict, everywhere it plausibly can be:
- Every model here sets `extra = "allow"`, so an LLM response with
  additional fields build_model() doesn't use isn't rejected outright
  for that alone -- only the fields build_model() actually reads are
  validated.
- known_value/uncertainty/value accept `float | str | None`, matching
  build_model()'s own existing tolerance for a numeric-looking string
  (some models emit "5" instead of 5) rather than tightening beyond
  what the rest of the pipeline already handles.
- solve_for and equations[].kind/derivation are similarly left loose
  where build_model()'s own normalization (_normalize_solve_for, the
  VALID_KINDS fallback) already does the real tolerance work -- this
  validates SHAPE (is it the right kind of JSON structure at all), not
  every value-level nuance build_model() was already handling fine.

What IS made strict, deliberately: variables[].symbol and
equations[].expression are REQUIRED. Those are the two fields
build_model() accesses with `v["symbol"]` / `eq["expression"]` --
direct dict indexing, not `.get()` -- specifically because there's no
sensible default for "which symbol is this variable" or "what is this
equation." A response missing either is genuinely malformed, and
should fail clearly right here instead of silently becoming a
mysterious downstream KeyError several stack frames later.
"""
from typing import Any

from pydantic import BaseModel, Field, ConfigDict


class VariablePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    symbol: str
    meaning: str = ""
    known_value: float | str | None = None
    unit: str | None = None
    is_function: bool = False
    is_vector: bool = False
    components: list = Field(default_factory=list)
    uncertainty: float | str | None = None
    domain: str | None = None


class EquationPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    name: str = "equation"
    expression: str
    derivation: str = ""
    kind: str = "equation"


class InitialConditionPayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    expression: str | None = None
    value: float | str | None = None


class ObjectivePayload(BaseModel):
    model_config = ConfigDict(extra="allow")

    expression: str | None = None
    direction: str = "minimize"
    optimize_over: Any = None


class ExtractionPayload(BaseModel):
    """The top-level shape build_model() expects. Every field has a
    default matching what build_model() itself falls back to via
    .get(key, default) today, so a response that omits an entire
    optional section (no initial_conditions, no objective, ...)
    validates the same way it always ran -- this tightens the fields
    that were UNSAFELY accessed, not the ones that were always tolerant."""
    model_config = ConfigDict(extra="allow")

    problem_domain: str = "unspecified"
    variables: list[VariablePayload] = Field(default_factory=list)
    equations: list[EquationPayload] = Field(default_factory=list)
    solve_for: Any = None
    assumptions: list = Field(default_factory=list)
    problem_type: str = "algebraic"
    independent_variable: str | None = None
    initial_conditions: list[InitialConditionPayload] = Field(default_factory=list)
    objective: ObjectivePayload | None = None


def validate_extraction_payload(payload: Any) -> dict:
    """Validates `payload` (already-parsed JSON, a dict expected) against
    ExtractionPayload and returns it back as a plain dict (via
    model_dump()) for build_model() to consume exactly as before --
    this is a validation GATE, not a rewrite of build_model()'s own
    parsing logic, so build_model() itself is untouched and this can be
    dropped in front of any dict source, not just the LLM's.

    Raises pydantic.ValidationError on a genuinely malformed payload;
    callers at the LLM boundary should catch that and re-raise as
    LLMOutputError, consistent with how every other malformed-response
    case in this pipeline is already surfaced (see extract_json())."""
    return ExtractionPayload.model_validate(payload).model_dump()
