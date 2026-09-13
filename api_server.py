"""
A thin REST API surface over this project's core verification-first
engine, wrapping the same modules the Streamlit app (app.py) itself
calls -- NOT a reimplementation of anything, a second front door onto
identical logic, so a result from this API and a result from the UI
for the same input are guaranteed to agree.

Why a separate process/entry point rather than adding routes to the
Streamlit app itself: Streamlit isn't built to also serve a JSON API
alongside its own UI, and the two have fundamentally different
consumers -- this exists specifically so the math engine can be used
from somewhere that ISN'T this Streamlit app at all (a lab notebook, a
grading pipeline, a second UI, a CI check that verifies a derivation)
without going through a browser session.

Run with:
    pip install -r requirements-api.txt
    uvicorn api_server:app --reload
Then see the auto-generated interactive docs at http://localhost:8000/docs

Deliberately NOT covering all 12 of the Streamlit app's modes -- these
four (structured problem solving, curve fitting, equivalence checking,
dimensional analysis) are the ones with the clearest "send me JSON, get
back JSON" shape and the least UI-specific state (no session_state,
no multi-step wizard). The others (PDE solver's many sub-tabs, tensor
calculus's interactive parameter sliders, the research journal) are
built around back-and-forth interactive exploration in a way a single
request/response doesn't fit naturally -- extending this to more
endpoints, following the same thin-wrapper pattern, is straightforward
if a concrete need for one shows up.

/solve here takes an already-STRUCTURED problem JSON (validated by the
same llm_schema.py gate extract_model() uses), not free-form problem
TEXT -- a caller that wants text-to-structure extraction can call
equation_engine.extract_model() directly in their own code instead.

NOTE: /solve still calls the same verify() the Streamlit UI does, and
verify() performs its own independent LLM re-solve as one of its
cross-checks (see verifier.py) -- so this endpoint DOES require LM
Studio reachable at the configured URL, exactly like the UI does,
despite skipping the free-text extraction step. This isn't a
limitation invented for the API; it's the real behavior of the
verification pipeline both front doors share. If LM Studio isn't
reachable, this returns a 503 with a clear message rather than a raw
connection-error traceback.
"""
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ValidationError

from modules.equation_engine import build_model
from modules.llm_schema import validate_extraction_payload
from modules.llm_client import LMStudioClient
from modules.verifier import verify
from modules.solver import compute_steps
from modules.curve_fitting import fit_curve, best_fit, BUILTIN_FAMILIES
from modules.equivalence import check_equivalence
from modules.dimensional_analysis import analyze_dimensions

app = FastAPI(
    title="Math Representation System API",
    description=__doc__,
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    """Liveness check -- returns immediately, touches no external
    service (specifically not LM Studio), so this succeeding says
    "the API process itself is up," not "the whole pipeline is."""
    return {"status": "ok"}


# --------------------------------------------------------------------- /solve
class SolveRequest(BaseModel):
    problem: dict[str, Any]


class SolveResponse(BaseModel):
    passed: bool
    confidence: str
    equations: list[str]
    numeric_answers: dict[str, float]
    check_summary: list[dict]
    steps: dict[str, list[dict]]


@app.post("/solve", response_model=SolveResponse)
def solve(request: SolveRequest) -> SolveResponse:
    """Builds, verifies, and solves an already-structured problem (the
    same JSON shape extract_model() validates -- see llm_schema.py).
    Returns the same verification report and step-by-step derivation
    the Streamlit UI would show for identical input."""
    try:
        validated = validate_extraction_payload(request.problem)
    except ValidationError as exc:
        problems = "; ".join(f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors())
        raise HTTPException(status_code=422, detail=f"Malformed problem JSON: {problems}") from exc

    model = build_model(validated)
    problem_text = request.problem.get("problem_text_hint", "") if isinstance(request.problem, dict) else ""
    try:
        client = LMStudioClient()
        report = verify(model, client, problem_text)
    except Exception as exc:  # noqa: BLE001
        # verify() performs its own independent LLM re-solve -- if LM
        # Studio isn't reachable, that's a service-availability problem
        # (503), not a malformed-request problem (422/400) or an
        # unexpected server bug (500)
        raise HTTPException(status_code=503,
                              detail=f"Could not complete verification (LM Studio unreachable?): {exc}") from exc
    steps_by_target = compute_steps(model)

    return SolveResponse(
        passed=report.passed,
        confidence=confidence_label_safe(report),
        equations=[str(eq.sympy_eq) for eq in model.equations if eq.sympy_eq is not None],
        numeric_answers=report.sympy_numeric_answers,
        check_summary=[{"label": c.label, "passed": c.passed, "detail": c.detail} for c in report.checks],
        steps={name: [{"description": s.description, "expression": s.expression,
                        "explanation": s.explanation} for s in steps]
               for name, steps in steps_by_target.items()},
    )


def confidence_label_safe(report) -> str:
    from modules.verifier import confidence_label
    try:
        return confidence_label(report)
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------- /curve-fit
class CurveFitRequest(BaseModel):
    xs: list[float]
    ys: list[float]
    family: str = "best fit"
    degree: int = 2
    expr_str: str | None = None
    param_names: list[str] | None = None


class CurveFitResponse(BaseModel):
    family: str
    expression: str
    parameters: dict[str, float]
    r_squared: float | None
    rmse: float | None
    error: str | None = None


@app.post("/curve-fit", response_model=CurveFitResponse)
def curve_fit(request: CurveFitRequest) -> CurveFitResponse:
    if request.family == "best fit":
        results = best_fit(request.xs, request.ys)
        if not results:
            raise HTTPException(status_code=422, detail="No built-in family could fit this data.")
        family, result = max(results.items(), key=lambda kv: kv[1].r_squared if kv[1].r_squared is not None else -1.0)
    else:
        if request.family not in BUILTIN_FAMILIES and request.family != "custom":
            raise HTTPException(status_code=422,
                                  detail=f"Unknown family {request.family!r}. Valid: "
                                          f"{list(BUILTIN_FAMILIES) + ['custom', 'best fit']}")
        family = request.family
        result = fit_curve(request.xs, request.ys, family, degree=request.degree,
                             expr_str=request.expr_str, param_names=request.param_names)
    if result.error:
        raise HTTPException(status_code=422, detail=result.error)
    return CurveFitResponse(family=family, expression=str(result.expr), parameters=result.param_values,
                              r_squared=result.r_squared, rmse=result.rmse)


# --------------------------------------------------------------------- /equivalence
class EquivalenceRequest(BaseModel):
    expr1: str
    expr2: str


class EquivalenceResponse(BaseModel):
    equivalent: bool | None
    method: str
    detail: str


@app.post("/equivalence", response_model=EquivalenceResponse)
def equivalence(request: EquivalenceRequest) -> EquivalenceResponse:
    result = check_equivalence(request.expr1, request.expr2)
    return EquivalenceResponse(equivalent=result.equivalent, method=result.method, detail=result.detail)


# --------------------------------------------------------------------- /dimensional-analysis
class DimensionalAnalysisRequest(BaseModel):
    inputs: dict[str, str]  # {symbol: unit}
    target_unit: str


class DimensionalAnalysisResponse(BaseModel):
    feasible: bool
    degrees_of_freedom: int
    particular_solution: dict[str, Any] | None
    message: str


@app.post("/dimensional-analysis", response_model=DimensionalAnalysisResponse)
def dimensional_analysis(request: DimensionalAnalysisRequest) -> DimensionalAnalysisResponse:
    result = analyze_dimensions(request.inputs, request.target_unit)
    # particular_solution's values are fractions.Fraction (analyze_dimensions
    # searches exponents as Fractions internally, e.g. 1/2 for a square
    # root relationship) -- not JSON-serializable as-is, so convert to
    # float for the API response; the UI shows them as Fractions directly
    # since Streamlit has no such restriction, but a REST response does
    particular_solution = ({k: float(v) for k, v in result.particular_solution.items()}
                            if result.particular_solution else None)
    return DimensionalAnalysisResponse(feasible=result.feasible, degrees_of_freedom=result.degrees_of_freedom,
                                          particular_solution=particular_solution,
                                          message=result.message)
