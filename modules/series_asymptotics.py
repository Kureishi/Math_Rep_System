"""
Series expansions (Taylor/Maclaurin, Laurent) and asymptotic expansions
(behavior as a variable -> 0 or -> oo), as a standalone symbolic tool --
same category as transforms.py and dimensional_analysis.py: the person
types a plain expression directly, there's no "problem" for an LLM to
extract.

Verification-first, same ethos as the rest of the app: the truncated
series isn't just handed back on faith that sp.series() did the right
thing. It's numerically compared against the ORIGINAL expression at a
few sample points near the expansion point (or, for an asymptotic
series at infinity, at a few large sample points) -- the error should
shrink as the sample point approaches the expansion point (Taylor/
Laurent case) or grows (asymptotic-at-infinity case), and that trend is
checked explicitly, not just the raw error size at one arbitrary point.

sp.series() has a real failure mode worth naming rather than papering
over: at an essential singularity (e.g. exp(1/x) at x=0) or across a
branch cut it can silently return the input expression completely
UNCHANGED, with no Order term and no exception raised -- i.e. "no
expansion is available" disguised as a normal-looking result. That's
detected explicitly here (via .getO() being absent AND the result
matching the input) and reported honestly, not displayed as if it were
a valid (trivial) expansion.
"""
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)


@dataclass
class SeriesResult:
    kind: str                      # "taylor" | "asymptotic"
    input_expr: sp.Expr | None
    point: sp.Expr | None          # expansion point (0, a finite value, or oo)
    order: int = 0
    truncated: sp.Expr | None = None    # the polynomial/Laurent approximation, O() removed
    order_term: sp.Expr | None = None   # the O(...) term itself, if one was produced
    terms: list[str] = field(default_factory=list)  # human-readable individual terms, for display
    expansion_available: bool = False   # False if SymPy returned the input unchanged (no info added)
    verified: bool = False
    verification_detail: str = ""
    error: str | None = None


def _parse(expr_str: str, var_name: str) -> tuple[sp.Expr, sp.Symbol]:
    var = sp.Symbol(var_name)
    expr = parse_expr(expr_str, local_dict={var_name: var}, transformations=_TRANSFORMS)
    return expr, var


def _split_terms(poly_expr: sp.Expr) -> list[str]:
    """Splits a truncated series into its individual additive terms for
    a readable term-by-term display, in the order SymPy already produced
    (lowest power first for a Taylor/Laurent series around a finite
    point; highest inverse power first for an asymptotic series -- both
    left as-is rather than re-sorted, since that ordering itself
    communicates which end of the expansion is the leading behavior)."""
    if poly_expr.is_Add:
        return [str(t) for t in poly_expr.args]
    return [str(poly_expr)]


def _verify_expansion(original: sp.Expr, truncated: sp.Expr, var: sp.Symbol,
                       point: sp.Expr, asymptotic: bool) -> tuple[bool, str]:
    """Numerically checks that the truncated expansion tracks the
    original function increasingly well as the sample point approaches
    the expansion point (finite-point case) or grows large (asymptotic
    case) -- a real convergence check, not just a single-point error."""
    if asymptotic:
        sample_points = [10.0, 100.0, 1000.0]
    elif point == 0:
        sample_points = [0.3, 0.1, 0.03]
    else:
        try:
            p = float(point)
        except (TypeError, ValueError):
            return False, "expansion point is not numeric; skipped numeric verification"
        sample_points = [p + 0.3, p + 0.1, p + 0.03] if not asymptotic else [p]

    errors = []
    try:
        for pt in sample_points:
            orig_val = complex(original.subs(var, pt))
            trunc_val = complex(truncated.subs(var, pt))
            errors.append(abs(orig_val - trunc_val))
    except (TypeError, ValueError, ZeroDivisionError) as exc:
        return False, f"could not numerically evaluate for verification ({exc})"

    if any(e != e for e in errors):  # NaN check
        return False, "numeric evaluation produced NaN at sample points; could not verify"

    is_shrinking = all(errors[i] >= errors[i + 1] - 1e-9 for i in range(len(errors) - 1))
    small_enough = errors[-1] < 1.0  # loose absolute bound; the trend matters more than this
    if is_shrinking and small_enough:
        pts_desc = ", ".join(f"{e:.2e}" for e in errors)
        direction = "growing" if asymptotic else "approaching the expansion point"
        return True, f"error shrank ({pts_desc}) as the sample point moved toward {direction}"
    return False, (f"error did not consistently shrink at sample points "
                    f"({', '.join(f'{e:.2e}' for e in errors)}) -- expansion may not be reliable here")


def taylor_series(expr_str: str, var_name: str = "x", point: float = 0, order: int = 6) -> SeriesResult:
    """Taylor (or Laurent, if `point` is a pole) series of expr_str in
    var_name around `point`, truncated at `order` terms."""
    try:
        expr, var = _parse(expr_str, var_name)
    except Exception as exc:  # noqa: BLE001
        return SeriesResult(kind="taylor", input_expr=None, point=sp.Integer(point),
                             error=f"Could not parse expression: {exc}")

    pt = sp.nsimplify(point) if point != 0 else sp.Integer(0)
    try:
        series_expr = run_with_timeout(sp.series, expr, var, pt, order + 1, label="series")
    except ComputationTimeoutError as exc:
        return SeriesResult(kind="taylor", input_expr=expr, point=pt, order=order, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return SeriesResult(kind="taylor", input_expr=expr, point=pt, order=order,
                             error=f"SymPy could not expand this series: {exc}")

    order_term = series_expr.getO()
    truncated = series_expr.removeO()

    # detect the "silently returned unchanged" failure mode described above
    # -- but distinguish it from the legitimate case where the function IS
    # already its own exact, zero-remainder Laurent series (e.g. 1/(x-1) at
    # x=1: a single term, nothing more to expand, not a failure at all).
    # is_rational_function(var) is the discriminator: an exact finite-order
    # pole reduces to a rational function of var; a genuine essential
    # singularity (exp(1/x), sin(1/x), ...) does not.
    if order_term is None:
        try:
            unchanged = sp.simplify(truncated - expr) == 0
        except Exception:  # noqa: BLE001
            unchanged = truncated == expr
        if unchanged:
            if truncated.is_rational_function(var):
                return SeriesResult(kind="taylor", input_expr=expr, point=pt, order=order,
                                     truncated=truncated, terms=_split_terms(truncated),
                                     expansion_available=True, verified=True,
                                     verification_detail="This is already the exact, complete Laurent "
                                                           "series at this point (a finite-order pole) -- "
                                                           "there is no remainder to add more terms to.")
            return SeriesResult(kind="taylor", input_expr=expr, point=pt, order=order,
                                 truncated=truncated, expansion_available=False,
                                 verification_detail="SymPy returned the expression unchanged -- "
                                                       "no series expansion is available here "
                                                       "(likely an essential singularity).")

    verified, detail = _verify_expansion(expr, truncated, var, pt, asymptotic=False)
    return SeriesResult(kind="taylor", input_expr=expr, point=pt, order=order,
                         truncated=truncated, order_term=order_term,
                         terms=_split_terms(truncated), expansion_available=True,
                         verified=verified, verification_detail=detail)


def asymptotic_expansion(expr_str: str, var_name: str = "x", order: int = 4) -> SeriesResult:
    """Asymptotic expansion of expr_str as var_name -> +infinity, in
    descending order of significance, truncated to `order` terms."""
    try:
        expr, var = _parse(expr_str, var_name)
    except Exception as exc:  # noqa: BLE001
        return SeriesResult(kind="asymptotic", input_expr=None, point=sp.oo,
                             error=f"Could not parse expression: {exc}")

    try:
        series_expr = run_with_timeout(sp.series, expr, var, sp.oo, order, label="series_asymptotic")
    except ComputationTimeoutError as exc:
        return SeriesResult(kind="asymptotic", input_expr=expr, point=sp.oo, order=order, error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return SeriesResult(kind="asymptotic", input_expr=expr, point=sp.oo, order=order,
                             error=f"SymPy could not expand this asymptotic series: {exc}")

    order_term = series_expr.getO()
    truncated = series_expr.removeO()

    if order_term is None:
        try:
            unchanged = sp.simplify(truncated - expr) == 0
        except Exception:  # noqa: BLE001
            unchanged = truncated == expr
        if unchanged:
            if truncated.is_rational_function(var):
                return SeriesResult(kind="asymptotic", input_expr=expr, point=sp.oo, order=order,
                                     truncated=truncated, terms=_split_terms(truncated),
                                     expansion_available=True, verified=True,
                                     verification_detail="This is already the exact asymptotic "
                                                           "behavior -- there is no remainder to add "
                                                           "more terms to.")
            return SeriesResult(kind="asymptotic", input_expr=expr, point=sp.oo, order=order,
                                 truncated=truncated, expansion_available=False,
                                 verification_detail="SymPy returned the expression unchanged -- no "
                                                       "asymptotic expansion is available here "
                                                       "(the function likely doesn't simplify as x -> oo, "
                                                       "e.g. an essential singularity at infinity).")

    verified, detail = _verify_expansion(expr, truncated, var, sp.oo, asymptotic=True)
    return SeriesResult(kind="asymptotic", input_expr=expr, point=sp.oo, order=order,
                         truncated=truncated, order_term=order_term,
                         terms=_split_terms(truncated), expansion_available=True,
                         verified=verified, verification_detail=detail)
