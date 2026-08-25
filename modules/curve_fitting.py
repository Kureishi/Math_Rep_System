"""
Curve/data fitting: a genuinely different input modality from the rest of
the app. Everywhere else, a symbolic model comes from an LLM reading a
word problem; here it comes from fitting a chosen model family to a table
of (x, y) numbers, typically an uploaded CSV. There's no LLM extraction
step and no "verification against an independent derivation" step in the
usual sense -- the equivalent of verification for a fit is its quality
metrics (R-squared, RMSE, residuals), which is what this module reports
instead.

Deliberately built on numpy only (no scipy) for every BUILT-IN family
(linear, polynomial, exponential, power, logarithmic): each of those is
solved by linearizing and calling numpy.polyfit, which is exact and
non-iterative, rather than pulling in scipy.optimize's nonlinear
least-squares machinery for cases that don't need it. This keeps the
project's local-first/no-heavy-dependency stance intact for the common
cases -- see the docstring on fit_custom() for where that tradeoff stops
being free.
"""
from dataclasses import dataclass, field
from typing import Callable
import csv
import io

import numpy as np
import sympy as sp

X = sp.Symbol("x")


@dataclass
class FitResult:
    family: str
    expr: sp.Expr | None                 # fitted model, numeric params substituted in, in terms of X
    param_values: dict[str, float]
    r_squared: float | None
    rmse: float | None
    residuals: list[float]               # y_i - y_hat_i, in data order
    x_label: str = "x"
    y_label: str = "y"
    error: str | None = None             # set instead of the above on failure

    def predict(self, x_val: float) -> float:
        if self.expr is None:
            raise ValueError(f"No fitted model available: {self.error}")
        return float(self.expr.subs(X, x_val))


# --------------------------------------------------------------------------- CSV input


def parse_xy_csv(text: str) -> tuple[list[float], list[float], str, str]:
    """Parses two-column CSV text into (xs, ys, x_label, y_label).
    Autodetects a header row (kept as axis labels if its cells aren't
    numeric); raises ValueError with a specific, actionable message on
    anything else that goes wrong, since a bad-upload error is the very
    first thing a user of this pipeline will hit."""
    reader = csv.reader(io.StringIO(text.strip()))
    rows = [r for r in reader if r and any(cell.strip() for cell in r)]
    if len(rows) < 2:
        raise ValueError("Need at least 2 data rows to fit a curve.")

    def _is_numeric_row(row: list[str]) -> bool:
        try:
            [float(c) for c in row[:2]]
            return True
        except ValueError:
            return False

    x_label, y_label = "x", "y"
    data_rows = rows
    if not _is_numeric_row(rows[0]):
        if len(rows[0]) >= 2:
            x_label, y_label = rows[0][0].strip(), rows[0][1].strip()
        data_rows = rows[1:]
        if len(data_rows) < 2:
            raise ValueError("Need at least 2 data rows (after the header) to fit a curve.")

    xs, ys = [], []
    for i, row in enumerate(data_rows, start=1):
        if len(row) < 2:
            raise ValueError(f"Row {i} has fewer than 2 columns: {row!r}")
        try:
            xs.append(float(row[0]))
            ys.append(float(row[1]))
        except ValueError as e:
            raise ValueError(f"Row {i} isn't numeric ({row!r}): {e}") from e
    return xs, ys, x_label, y_label


# --------------------------------------------------------------------------- quality metrics


def _quality(ys: np.ndarray, yhat: np.ndarray) -> tuple[float, float, list[float]]:
    residuals = ys - yhat
    ss_res = float(np.sum(residuals ** 2))
    ss_tot = float(np.sum((ys - ys.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else (1.0 if ss_res == 0 else float("nan"))
    rmse = float(np.sqrt(ss_res / len(ys)))
    return r2, rmse, residuals.tolist()


def _err(family: str, message: str) -> FitResult:
    return FitResult(family=family, expr=None, param_values={}, r_squared=None,
                       rmse=None, residuals=[], error=message)


# --------------------------------------------------------------------------- built-in families


def fit_linear(xs: list[float], ys: list[float]) -> FitResult:
    return fit_polynomial(xs, ys, degree=1)


def fit_polynomial(xs: list[float], ys: list[float], degree: int) -> FitResult:
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs_a) < degree + 1:
        return _err("polynomial", f"Need at least {degree + 1} points to fit a degree-{degree} polynomial "
                                    f"(got {len(xs_a)}).")
    coeffs = np.polyfit(xs_a, ys_a, degree)
    yhat = np.polyval(coeffs, xs_a)
    r2, rmse, residuals = _quality(ys_a, yhat)
    expr = sum(sp.Float(float(c), 6) * X ** (degree - i) for i, c in enumerate(coeffs))
    params = {f"c{degree - i}": float(c) for i, c in enumerate(coeffs)}
    family = "linear" if degree == 1 else f"polynomial (degree {degree})"
    return FitResult(family=family, expr=sp.expand(expr), param_values=params,
                       r_squared=r2, rmse=rmse, residuals=residuals)


def fit_exponential(xs: list[float], ys: list[float]) -> FitResult:
    """y = a * exp(b*x), fit by linearizing: ln(y) = ln(a) + b*x.
    Requires every y > 0 (the linearization is undefined otherwise)."""
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.any(ys_a <= 0):
        return _err("exponential", "Exponential fit (y = a*exp(b*x)) requires every y-value to be "
                                     "positive -- at least one value is zero or negative.")
    coeffs = np.polyfit(xs_a, np.log(ys_a), 1)
    b, ln_a = coeffs
    a = float(np.exp(ln_a))
    yhat = a * np.exp(b * xs_a)
    r2, rmse, residuals = _quality(ys_a, yhat)
    expr = sp.Float(a, 6) * sp.exp(sp.Float(float(b), 6) * X)
    return FitResult(family="exponential", expr=expr, param_values={"a": a, "b": float(b)},
                       r_squared=r2, rmse=rmse, residuals=residuals)


def fit_power(xs: list[float], ys: list[float]) -> FitResult:
    """y = a * x^b, fit by linearizing: ln(y) = ln(a) + b*ln(x).
    Requires every x > 0 and y > 0."""
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.any(xs_a <= 0) or np.any(ys_a <= 0):
        return _err("power", "Power fit (y = a*x^b) requires every x-value and y-value to be positive.")
    coeffs = np.polyfit(np.log(xs_a), np.log(ys_a), 1)
    b, ln_a = coeffs
    a = float(np.exp(ln_a))
    yhat = a * xs_a ** b
    r2, rmse, residuals = _quality(ys_a, yhat)
    expr = sp.Float(a, 6) * X ** sp.Float(float(b), 6)
    return FitResult(family="power", expr=expr, param_values={"a": a, "b": float(b)},
                       r_squared=r2, rmse=rmse, residuals=residuals)


def fit_logarithmic(xs: list[float], ys: list[float]) -> FitResult:
    """y = a*ln(x) + b, a direct linear fit against ln(x). Requires
    every x > 0."""
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if np.any(xs_a <= 0):
        return _err("logarithmic", "Logarithmic fit (y = a*ln(x) + b) requires every x-value to be positive.")
    coeffs = np.polyfit(np.log(xs_a), ys_a, 1)
    a, b = coeffs
    yhat = a * np.log(xs_a) + b
    r2, rmse, residuals = _quality(ys_a, yhat)
    expr = sp.Float(float(a), 6) * sp.log(X) + sp.Float(float(b), 6)
    return FitResult(family="logarithmic", expr=expr, param_values={"a": float(a), "b": float(b)},
                       r_squared=r2, rmse=rmse, residuals=residuals)


# --------------------------------------------------------------------------- custom (linear-in-parameters) fit


def fit_custom(xs: list[float], ys: list[float], expr_str: str, param_names: list[str]) -> FitResult:
    """Fits a user-supplied model that's an arbitrary function of x but
    must be LINEAR in its named parameters (e.g. "a*sin(x) + b*x + c" or
    "a*x**2 + b*sqrt(x)" -- any combination of fixed basis functions of
    x, each scaled by one parameter, optionally plus a constant term).

    Why this restriction rather than general nonlinear least squares
    (e.g. "a*sin(b*x)", where b sits inside a function of x): a
    linear-in-parameters model can be solved EXACTLY and non-iteratively
    with plain linear algebra (numpy.linalg.lstsq) -- no scipy, no
    initial-guess sensitivity, no convergence failures. A genuinely
    nonlinear custom fit needs iterative nonlinear optimization (e.g.
    scipy.optimize.curve_fit), which is a real additional dependency;
    rather than pull that in for a capability most curve-fitting
    requests don't need, this fits what it can fit for free and reports
    a specific, honest error explaining why for the rest -- an explicit
    portability tradeoff, not a silently narrower feature.
    """
    try:
        params = [sp.Symbol(p) for p in param_names]
        local_dict = {p.name: p for p in params}
        local_dict["x"] = X
        expr = sp.sympify(expr_str, locals=local_dict)
    except (sp.SympifyError, TypeError, SyntaxError) as e:
        return _err("custom", f"Couldn't parse the model expression: {e}")

    unknown_syms = expr.free_symbols - {X} - set(params)
    if unknown_syms:
        return _err("custom", f"Expression uses symbol(s) not listed as parameters: "
                                f"{', '.join(sorted(s.name for s in unknown_syms))}")

    for i, pi in enumerate(params):
        if sp.diff(expr, pi, 2) != 0:
            return _err("custom", f"Not linear in parameter '{pi.name}' (custom fit only supports "
                                    "models that are linear in every parameter -- try a different "
                                    "parameterization, or use a built-in family).")
        for pj in params[i + 1:]:
            if sp.diff(expr, pi, pj) != 0:
                return _err("custom", f"Parameters '{pi.name}' and '{pj.name}' interact "
                                        "(non-additive) -- custom fit requires each parameter to "
                                        "scale an independent term.")

    basis_exprs = [sp.diff(expr, p) for p in params]
    offset_expr = sp.simplify(expr - sum(p * b for p, b in zip(params, basis_exprs)))
    if offset_expr.free_symbols - {X}:
        return _err("custom", "Internal linearity check failed unexpectedly -- please rephrase the model.")

    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    if len(xs_a) < len(params):
        return _err("custom", f"Need at least {len(params)} points to fit {len(params)} parameters "
                                f"(got {len(xs_a)}).")

    basis_funcs = [sp.lambdify(X, b, "numpy") for b in basis_exprs]
    offset_func = sp.lambdify(X, offset_expr, "numpy")
    try:
        design = np.column_stack([np.broadcast_to(np.asarray(f(xs_a), dtype=float), xs_a.shape)
                                    for f in basis_funcs])
        offset_vals = np.broadcast_to(np.asarray(offset_func(xs_a), dtype=float), xs_a.shape)
    except Exception as e:  # noqa: BLE001
        return _err("custom", f"Couldn't evaluate the model over the given x-values: {e}")

    target = ys_a - offset_vals
    coeffs, *_ = np.linalg.lstsq(design, target, rcond=None)

    param_values = {p.name: float(c) for p, c in zip(params, coeffs)}
    fitted_expr = expr.subs({p: sp.Float(v, 6) for p, v in zip(params, coeffs)})
    predict_func = sp.lambdify(X, fitted_expr, "numpy")
    yhat = np.broadcast_to(np.asarray(predict_func(xs_a), dtype=float), xs_a.shape)
    r2, rmse, residuals = _quality(ys_a, yhat)
    return FitResult(family="custom", expr=sp.expand(fitted_expr), param_values=param_values,
                       r_squared=r2, rmse=rmse, residuals=residuals)


# --------------------------------------------------------------------------- dispatch + "best fit"


BUILTIN_FAMILIES = ("linear", "polynomial", "exponential", "power", "logarithmic")


def fit_curve(xs: list[float], ys: list[float], family: str, degree: int = 2,
              expr_str: str | None = None, param_names: list[str] | None = None) -> FitResult:
    """Single dispatch point used by the UI. `degree` only applies to
    "polynomial"; `expr_str`/`param_names` only apply to "custom"."""
    if len(xs) != len(ys):
        return _err(family, f"x and y have different lengths ({len(xs)} vs {len(ys)}).")
    if len(xs) < 2:
        return _err(family, "Need at least 2 data points to fit anything.")

    if family == "linear":
        return fit_linear(xs, ys)
    if family == "polynomial":
        return fit_polynomial(xs, ys, degree)
    if family == "exponential":
        return fit_exponential(xs, ys)
    if family == "power":
        return fit_power(xs, ys)
    if family == "logarithmic":
        return fit_logarithmic(xs, ys)
    if family == "custom":
        if not expr_str or not param_names:
            return _err("custom", "Custom fit needs both a model expression and a list of parameter names.")
        return fit_custom(xs, ys, expr_str, param_names)
    return _err(family, f"Unknown model family '{family}'.")


def best_fit(xs: list[float], ys: list[float], candidates: tuple[str, ...] = BUILTIN_FAMILIES,
             degree: int = 2) -> dict[str, FitResult]:
    """Tries every candidate family (skipping ones that fail, e.g.
    exponential/power/log on data with non-positive values) and returns
    all successful fits, so the caller/UI can rank them by R-squared
    itself rather than this function silently picking a "winner"."""
    results = {}
    for fam in candidates:
        r = fit_curve(xs, ys, fam, degree=degree)
        if r.error is None:
            results[fam] = r
    return results
