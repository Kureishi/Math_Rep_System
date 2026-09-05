"""
Adversarial edge-case generator: a QA tool for the SYSTEM ITSELF, not
for a student. Takes an already-extracted ProblemModel and generates
deliberately nasty variants of its own known inputs -- a zero (division
by zero), a flipped sign (a negative time, a negative mass), an
extremely large or extremely small magnitude -- then runs each variant
through this app's own solving + plausibility layers and reports EXACTLY
what happened: solved cleanly, correctly recognized as unsolvable,
timed out, or raised an exception. A human reviewing the report decides
whether a given outcome is acceptable (a ZeroDivisionError-flavored
"unsolvable" for a zero denominator is expected and fine) or reveals an
actual bug (an AttributeError from code that assumed a value could
never be None) -- this module deliberately doesn't pre-judge that,
it just makes every outcome visible instead of the pipeline silently
swallowing it or, worse, crashing the whole app around a Streamlit
callback.

Distinct from the STUDENT-facing plausibility.py (which flags an
ALREADY-SOLVED answer that looks physically off) and domain_utils.py
(which reports where a formula is mathematically undefined FOR THE
ORIGINAL inputs): this module doesn't care whether a result looks
sensible -- it cares whether the pipeline SURVIVES being handed
adversarial input without an unhandled exception reaching the user.

Runs each variant's solve through timeout_utils.run_with_timeout with
its own short timeout, deliberately separate from
settings.computation_timeout_seconds -- an extreme magnitude (a "huge"
or "tiny" variant) can, through _known_substitutions()'s own
sp.nsimplify() call, trigger sympy's occasionally very slow algebraic-
number-reconstruction path (the exact performance cliff
monte_carlo.py's own docstring describes hitting and working around);
this tool intentionally does NOT route around that the way
monte_carlo.py does, since exercising the REAL pipeline path -- warts
included -- is the whole point, but it needs its own timeout so one
slow variant can't hang the entire adversarial suite.
"""
from dataclasses import dataclass

from modules.equation_engine import ProblemModel, build_model
from modules.verifier import _solve_sympy
from modules.plausibility import check_plausibility
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

EDGE_CASE_TIMEOUT_SECONDS = 5  # generous for a single symbolic solve -- if a variant genuinely
                                 # needs longer than this, that slowness IS the finding worth reporting


@dataclass
class EdgeCaseVariant:
    label: str      # human-readable, e.g. "t = 0 (zero)"
    symbol: str
    value: float
    category: str    # "zero" | "negative" | "tiny" | "huge"


@dataclass
class EdgeCaseOutcome:
    variant: EdgeCaseVariant
    target: str
    status: str      # "solved" | "unsolvable" | "timeout" | "exception"
    detail: str
    value: float | None = None


def generate_edge_cases(model: ProblemModel) -> list[EdgeCaseVariant]:
    """One variant per category, per currently-known variable --
    variables that are themselves unknown/solve_for targets aren't
    inputs to perturb, so they're skipped."""
    variants: list[EdgeCaseVariant] = []
    for v in model.variables:
        if v.known_value is None:
            continue
        base = v.known_value
        variants.append(EdgeCaseVariant(f"{v.symbol} = 0 (zero)", v.symbol, 0.0, "zero"))
        neg = -abs(base) if base != 0 else -1.0
        variants.append(EdgeCaseVariant(f"{v.symbol} = {neg:g} (negative)", v.symbol, neg, "negative"))
        tiny = (abs(base) or 1.0) * 1e-9
        variants.append(EdgeCaseVariant(f"{v.symbol} = {tiny:g} (tiny)", v.symbol, tiny, "tiny"))
        huge = (abs(base) or 1.0) * 1e9
        variants.append(EdgeCaseVariant(f"{v.symbol} = {huge:g} (huge)", v.symbol, huge, "huge"))
    return variants


def _solve_and_check_plausibility(sample_model: ProblemModel, target: str) -> dict:
    """The actual pipeline surface under test: solve, then run the
    solved/known values through the plausibility layer too, since that
    layer should never itself raise regardless of how extreme its
    inputs are."""
    answers = _solve_sympy(sample_model)
    known_values = {v.symbol: v.known_value for v in sample_model.variables if v.known_value is not None}
    check_plausibility(sample_model, {**known_values, **answers})
    return answers


def run_edge_case(model: ProblemModel, variant: EdgeCaseVariant, target: str) -> EdgeCaseOutcome:
    """Runs ONE variant through the pipeline and reports what happened.
    This function itself never raises -- an exception from the layers
    under test is caught and reported AS the outcome, since crashing
    the reporting function would defeat the point of a tool meant to
    survive exactly that."""
    try:
        sample_model = build_model(model.raw_json)
        var = next((v for v in sample_model.variables if v.symbol == variant.symbol), None)
        if var is None:
            return EdgeCaseOutcome(variant, target, "exception",
                                     f"'{variant.symbol}' was missing after rebuilding the model.")
        var.known_value = variant.value

        answers = run_with_timeout(_solve_and_check_plausibility, sample_model, target,
                                     timeout=EDGE_CASE_TIMEOUT_SECONDS,
                                     label=f"edge case: {variant.label}")
    except ComputationTimeoutError:
        return EdgeCaseOutcome(variant, target, "timeout",
                                 f"Didn't finish within {EDGE_CASE_TIMEOUT_SECONDS}s -- a possible "
                                 "performance issue with this kind of input, not necessarily a crash.")
    except Exception as e:  # noqa: BLE001 -- deliberately broad: this IS the crash detector
        return EdgeCaseOutcome(variant, target, "exception", f"{type(e).__name__}: {e}")

    value = answers.get(target)
    if value is None:
        return EdgeCaseOutcome(variant, target, "unsolvable",
                                 "No real solution for this target with this input -- handled cleanly.")
    return EdgeCaseOutcome(variant, target, "solved", f"{target} = {value:.6g}", value=value)


def run_adversarial_suite(model: ProblemModel, target: str) -> list[EdgeCaseOutcome]:
    """Generates and runs every edge case for `model`/`target`, in
    generation order."""
    return [run_edge_case(model, variant, target) for variant in generate_edge_cases(model)]
