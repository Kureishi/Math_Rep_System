"""
Integral transforms (Laplace and Fourier, and their inverses) as a
standalone symbolic-math tool -- distinct from the LLM-extraction
pipeline in equation_engine.py the way dimensional_analysis.py and
equivalence.py are: the person types a plain expression directly (no
"problem" to extract, no known/unknown values), so there's nothing for
an LLM to extract here in the first place.

Verification-first, consistent with the rest of the app: this module
never just hands back whatever SymPy's laplace_transform/fourier_transform
computes and calls it done. Every forward transform is round-tripped
through the matching inverse and compared back to the original
expression -- symbolically first (Expr.equals()), falling back to
numeric sampling at a handful of points if that's inconclusive, the
same two-tier pattern equivalence.py and recurrence_utils.py already
use. A transform that doesn't round-trip cleanly is reported as
UNVERIFIED, not silently presented as trustworthy.

SymPy frequently cannot evaluate a transform in closed form (most
functions have no elementary Laplace/Fourier transform) and instead
hands back an inert, unevaluated LaplaceTransform/FourierTransform
object dressed up as an ordinary expression. That's detected explicitly
here (via .has() against the relevant unevaluated-transform class) and
reported honestly as "could not evaluate in closed form" -- not passed
through as if it were a real answer, which is what would happen if the
result were just displayed or substituted into further without this
check.
"""
from dataclasses import dataclass

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# numeric points used for round-trip verification when a symbolic match
# doesn't fall out directly -- kept positive and away from 0 since most
# transform pairs (Laplace especially) are only valid/defined for t > 0
_SAMPLE_POINTS = (0.5, 1.0, 1.7, 2.5, 4.0)

_UNEVALUATED_CLASSES = {
    "laplace": sp.LaplaceTransform,
    "inverse_laplace": sp.InverseLaplaceTransform,
    "fourier": sp.FourierTransform,
    "inverse_fourier": sp.InverseFourierTransform,
}

# what the *other* direction's helper is, keyed by kind -- used to
# round-trip verify a computed transform back to the original expression
_INVERSE_OF = {
    "laplace": "inverse_laplace",
    "inverse_laplace": "laplace",
    "fourier": "inverse_fourier",
    "inverse_fourier": "fourier",
}


@dataclass
class TransformResult:
    kind: str                          # "laplace" | "inverse_laplace" | "fourier" | "inverse_fourier"
    input_expr: sp.Expr | None
    output_expr: sp.Expr | None        # None if evaluation failed or timed out
    convergence_condition: sp.Basic | None = None  # Laplace only: the Re(s) > a condition, if returned
    evaluated: bool = False            # False if SymPy left this as an inert unevaluated transform
    verified: bool = False
    verification_method: str = ""      # "symbolic" | "numeric sampling" | "not attempted"
    verification_detail: str = ""
    error: str | None = None


def _parse(expr_str: str, symbols: dict[str, sp.Symbol]) -> sp.Expr:
    """Parses expr_str using the EXACT symbol objects given in `symbols`
    (e.g. the assumptions-bearing t = Symbol('t', positive=True) a caller
    already built), rather than minting fresh plain symbols of the same
    name -- critical here, since sp.laplace_transform(expr, t, s) silently
    treats expr's "t" as unrelated to the t it was called with if they're
    two different Symbol objects that merely print the same, producing a
    wrong (but not-erroring) transform instead of a clean failure."""
    return parse_expr(expr_str, local_dict=dict(symbols), transformations=_TRANSFORMS)


def _is_evaluated(result: sp.Basic, kind: str) -> bool:
    return not result.has(_UNEVALUATED_CLASSES[kind])


def _round_trip_verify(kind: str, original: sp.Expr, transformed: sp.Expr,
                        orig_var: sp.Symbol, new_var: sp.Symbol) -> tuple[bool, str, str]:
    """Applies the opposite-direction transform to `transformed` and checks
    it reproduces `original`. Returns (verified, method, detail)."""
    inverse_kind = _INVERSE_OF[kind]
    try:
        if inverse_kind in ("inverse_laplace", "laplace"):
            back = sp.laplace_transform(transformed, new_var, orig_var, noconds=True) \
                if inverse_kind == "laplace" else \
                sp.inverse_laplace_transform(transformed, new_var, orig_var)
        else:
            back = sp.fourier_transform(transformed, new_var, orig_var) \
                if inverse_kind == "fourier" else \
                sp.inverse_fourier_transform(transformed, new_var, orig_var)
    except Exception as exc:  # noqa: BLE001
        return False, "not attempted", f"round-trip could not be computed ({exc})"

    if back.has(_UNEVALUATED_CLASSES[inverse_kind]):
        return False, "not attempted", "round-trip transform did not evaluate in closed form"

    try:
        diff = sp.simplify(back - original)
        if diff == 0:
            return True, "symbolic", "round-trip matched the original expression exactly"
    except Exception:  # noqa: BLE001
        pass

    # numeric fallback: sample both sides at a few positive points
    try:
        max_abs = 0.0
        for pt in _SAMPLE_POINTS:
            lhs = complex(back.subs(orig_var, pt))
            rhs = complex(original.subs(orig_var, pt))
            max_abs = max(max_abs, abs(lhs - rhs))
        if max_abs < 1e-6:
            return True, "numeric sampling", f"round-trip matched within {max_abs:.2e} at sample points"
        return False, "numeric sampling", f"round-trip disagreed by up to {max_abs:.3g} at sample points"
    except (TypeError, ValueError) as exc:
        return False, "not attempted", f"round-trip comparison failed ({exc})"


def laplace_transform_expr(expr_str: str, indep_var: str = "t", transform_var: str = "s") -> TransformResult:
    """Computes the Laplace transform of expr_str (a function of indep_var,
    t >= 0 assumed) with respect to transform_var, then round-trip
    verifies it via the inverse transform."""
    t = sp.Symbol(indep_var, positive=True)
    s = sp.Symbol(transform_var)
    try:
        expr = _parse(expr_str, {indep_var: t})
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="laplace", input_expr=None, output_expr=None,
                                error=f"Could not parse expression: {exc}")

    try:
        result, conv, _ = run_with_timeout(
            sp.laplace_transform, expr, t, s, noconds=False, label="laplace_transform")
    except ComputationTimeoutError as exc:
        return TransformResult(kind="laplace", input_expr=expr, output_expr=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="laplace", input_expr=expr, output_expr=None,
                                error=f"SymPy could not compute this transform: {exc}")

    evaluated = _is_evaluated(result, "laplace")
    if not evaluated:
        return TransformResult(kind="laplace", input_expr=expr, output_expr=result,
                                convergence_condition=conv, evaluated=False,
                                verification_method="not attempted",
                                verification_detail="SymPy left this transform unevaluated -- "
                                                     "no closed form was found.")

    verified, method, detail = _round_trip_verify("laplace", expr, result, t, s)
    return TransformResult(kind="laplace", input_expr=expr, output_expr=result,
                            convergence_condition=conv, evaluated=True,
                            verified=verified, verification_method=method, verification_detail=detail)


def inverse_laplace_transform_expr(expr_str: str, transform_var: str = "s",
                                    indep_var: str = "t") -> TransformResult:
    """Computes the inverse Laplace transform of expr_str (a function of
    transform_var) back into indep_var, round-trip verified via the
    forward transform."""
    s = sp.Symbol(transform_var)
    t = sp.Symbol(indep_var, positive=True)
    try:
        expr = _parse(expr_str, {transform_var: s})
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="inverse_laplace", input_expr=None, output_expr=None,
                                error=f"Could not parse expression: {exc}")

    try:
        result = run_with_timeout(sp.inverse_laplace_transform, expr, s, t,
                                   label="inverse_laplace_transform")
    except ComputationTimeoutError as exc:
        return TransformResult(kind="inverse_laplace", input_expr=expr, output_expr=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="inverse_laplace", input_expr=expr, output_expr=None,
                                error=f"SymPy could not compute this inverse transform: {exc}")

    evaluated = _is_evaluated(result, "inverse_laplace")
    if not evaluated:
        return TransformResult(kind="inverse_laplace", input_expr=expr, output_expr=result, evaluated=False,
                                verification_method="not attempted",
                                verification_detail="SymPy left this inverse transform unevaluated -- "
                                                     "no closed form was found.")

    verified, method, detail = _round_trip_verify("inverse_laplace", expr, result, s, t)
    return TransformResult(kind="inverse_laplace", input_expr=expr, output_expr=result, evaluated=True,
                            verified=verified, verification_method=method, verification_detail=detail)


def fourier_transform_expr(expr_str: str, indep_var: str = "x", transform_var: str = "k") -> TransformResult:
    """Computes the Fourier transform of expr_str with respect to
    transform_var, round-trip verified via the inverse transform."""
    x = sp.Symbol(indep_var, real=True)
    k = sp.Symbol(transform_var, real=True)
    try:
        expr = _parse(expr_str, {indep_var: x})
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="fourier", input_expr=None, output_expr=None,
                                error=f"Could not parse expression: {exc}")

    try:
        result = run_with_timeout(sp.fourier_transform, expr, x, k, label="fourier_transform")
    except ComputationTimeoutError as exc:
        return TransformResult(kind="fourier", input_expr=expr, output_expr=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="fourier", input_expr=expr, output_expr=None,
                                error=f"SymPy could not compute this transform: {exc}")

    evaluated = _is_evaluated(result, "fourier")
    if not evaluated:
        return TransformResult(kind="fourier", input_expr=expr, output_expr=result, evaluated=False,
                                verification_method="not attempted",
                                verification_detail="SymPy left this transform unevaluated -- "
                                                     "no closed form was found.")

    verified, method, detail = _round_trip_verify("fourier", expr, result, x, k)
    return TransformResult(kind="fourier", input_expr=expr, output_expr=result, evaluated=True,
                            verified=verified, verification_method=method, verification_detail=detail)


def inverse_fourier_transform_expr(expr_str: str, transform_var: str = "k",
                                    indep_var: str = "x") -> TransformResult:
    """Computes the inverse Fourier transform of expr_str back into
    indep_var, round-trip verified via the forward transform."""
    k = sp.Symbol(transform_var, real=True)
    x = sp.Symbol(indep_var, real=True)
    try:
        expr = _parse(expr_str, {transform_var: k})
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="inverse_fourier", input_expr=None, output_expr=None,
                                error=f"Could not parse expression: {exc}")

    try:
        result = run_with_timeout(sp.inverse_fourier_transform, expr, k, x,
                                   label="inverse_fourier_transform")
    except ComputationTimeoutError as exc:
        return TransformResult(kind="inverse_fourier", input_expr=expr, output_expr=None, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return TransformResult(kind="inverse_fourier", input_expr=expr, output_expr=None,
                                error=f"SymPy could not compute this inverse transform: {exc}")

    evaluated = _is_evaluated(result, "inverse_fourier")
    if not evaluated:
        return TransformResult(kind="inverse_fourier", input_expr=expr, output_expr=result, evaluated=False,
                                verification_method="not attempted",
                                verification_detail="SymPy left this inverse transform unevaluated -- "
                                                     "no closed form was found.")

    verified, method, detail = _round_trip_verify("inverse_fourier", expr, result, k, x)
    return TransformResult(kind="inverse_fourier", input_expr=expr, output_expr=result, evaluated=True,
                            verified=verified, verification_method=method, verification_detail=detail)
