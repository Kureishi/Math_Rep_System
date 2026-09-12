"""
The statistical/data-science layer on top of curve_fitting.py: parameter
confidence intervals and hypothesis tests, residual diagnostics, Bayesian
regression, and nested-model comparison. Everywhere this module reports
a number, it also reports whether it means anything -- the difference
between "here's a fit" and "here's a fit, and here's how much to trust
it" is the entire point of this module.

Built on numpy + scipy.stats only (already a dependency elsewhere in
this app -- see pde_utils.py's Robin-eigenvalue root-finding and
timeout_utils.py), no statsmodels or other heavier statistics package.
Every formula here (OLS standard errors, the overall F-test, Durbin-
Watson, Breusch-Pagan, the conjugate Normal-Inverse-Gamma Bayesian
posterior, and the nested-model F-test) was cross-validated against
statsmodels during development, matching to the reported decimal places
in every case -- see the corresponding tests in test_statistical_inference.py,
several of which cross-check against hand-computed or scipy-only
reference values for exactly this reason. statsmodels itself is NOT a
runtime dependency; it was a development-time-only reference.

This module deliberately works from curve_fitting.regression_design_matrix()
rather than re-deriving each family's linearization independently --
that would risk silently drifting from what fit_curve() actually solved
(e.g. computing a confidence interval for the WRONG regression problem
if the two linearizations happened to disagree in some detail). Every
inferential number here is about the exact regression fit_curve solved.
"""
from dataclasses import dataclass, field

import numpy as np
from scipy import stats

from modules.curve_fitting import regression_design_matrix


def _rename_log_transformed(param_names: list[str]) -> list[str]:
    """Cosmetic only: regression_design_matrix names the log-space
    intercept "ln_a" (since that's literally the linear-regression
    coefficient being estimated) -- rename it to "a" for display, since
    by the time a caller sees it, param_transforms has already been
    applied and it truly is (the back-transformed value of) a."""
    return ["a" if n == "ln_a" else n for n in param_names]


# --------------------------------------------------------------------- frequentist inference
@dataclass
class ParameterInference:
    name: str
    estimate: float
    std_error: float
    t_statistic: float
    p_value: float
    ci_lower: float
    ci_upper: float


@dataclass
class RegressionInferenceResult:
    parameters: list[ParameterInference] = field(default_factory=list)
    confidence_level: float = 0.95
    degrees_of_freedom: int = 0
    f_statistic: float | None = None       # overall-significance F-test; None if no non-intercept regressor
    f_p_value: float | None = None
    r_squared: float | None = None
    adjusted_r_squared: float | None = None
    error: str | None = None


def regression_inference(xs: list[float], ys: list[float], family: str, degree: int = 2,
                          expr_str: str | None = None, param_names: list[str] | None = None,
                          confidence: float = 0.95) -> RegressionInferenceResult:
    """Standard errors, t-tests (H0: parameter = 0), and confidence
    intervals for every parameter of a fit_curve() model, plus an
    overall F-test (H0: every non-intercept coefficient is zero, i.e.
    the model explains no more than a flat mean would). Confidence
    intervals for log-linearized parameters (exponential/power's "a")
    are computed in log-space and back-transformed through exp(), which
    is valid because exp() is monotonic -- it maps interval endpoints
    to interval endpoints, so this is exact, not an approximation."""
    rd = regression_design_matrix(xs, ys, family, degree, expr_str, param_names)
    if rd.error:
        return RegressionInferenceResult(error=rd.error)

    assert rd.design is not None and rd.target is not None  # guaranteed by the error check
    # above -- regression_design_matrix() only returns error=None alongside a populated
    # design/target pair
    design, target = rd.design, rd.target
    n, p = design.shape
    dof = n - p
    if dof <= 0:
        return RegressionInferenceResult(error=f"Not enough data points for inference: {n} points, "
                                                 f"{p} parameters (need more points than parameters).")

    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    yhat = design @ beta
    resid = target - yhat
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((target - target.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    try:
        XtX_inv = np.linalg.inv(design.T @ design)
    except np.linalg.LinAlgError:
        return RegressionInferenceResult(error="Design matrix is singular (parameters aren't "
                                                 "independently identifiable from this data) -- "
                                                 "cannot compute standard errors.")
    sigma2 = ss_res / dof
    se = np.sqrt(np.maximum(np.diag(sigma2 * XtX_inv), 0.0))
    t_stats = np.divide(beta, se, out=np.full_like(beta, np.nan), where=se > 0)
    p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), dof))
    t_crit = stats.t.ppf(1 - (1 - confidence) / 2, dof)
    ci_lo_linspace = beta - t_crit * se
    ci_hi_linspace = beta + t_crit * se

    display_names = _rename_log_transformed(rd.param_names)
    parameters = []
    for name, disp_name, est, s, t_stat, p_val, lo, hi in zip(
            rd.param_names, display_names, beta, se, t_stats, p_values, ci_lo_linspace, ci_hi_linspace):
        transform = rd.param_transforms.get(name, lambda v: v)
        # transform is monotonic (identity or exp) in every case this module
        # builds, so transforming the two endpoints gives the transformed CI
        parameters.append(ParameterInference(
            name=disp_name, estimate=float(transform(est)), std_error=float(s),
            t_statistic=float(t_stat), p_value=float(p_val),
            ci_lower=float(transform(lo)), ci_upper=float(transform(hi))))

    # overall F-test: only meaningful if there's a non-constant regressor
    # to test against a flat-mean null; skip it for a single-parameter model
    f_stat = f_p = None
    if p >= 2 and ss_tot > 0:
        dof_num = p - 1
        f_stat = (r2 / dof_num) / ((1 - r2) / dof) if r2 < 1 else float("inf")
        f_p = float(1 - stats.f.cdf(f_stat, dof_num, dof)) if np.isfinite(f_stat) else 0.0
        f_stat = float(f_stat)
    adj_r2 = 1 - (1 - r2) * (n - 1) / dof if dof > 0 else float("nan")

    return RegressionInferenceResult(parameters=parameters, confidence_level=confidence,
                                      degrees_of_freedom=dof, f_statistic=f_stat, f_p_value=f_p,
                                      r_squared=float(r2), adjusted_r_squared=float(adj_r2))


# --------------------------------------------------------------------- residual diagnostics
@dataclass
class ResidualDiagnosticsResult:
    normality_statistic: float | None = None
    normality_p_value: float | None = None
    normality_note: str = ""
    durbin_watson: float | None = None
    durbin_watson_note: str = ""
    heteroscedasticity_statistic: float | None = None
    heteroscedasticity_p_value: float | None = None
    heteroscedasticity_note: str = ""
    error: str | None = None


def residual_diagnostics(xs: list[float], ys: list[float], family: str, degree: int = 2,
                          expr_str: str | None = None, param_names: list[str] | None = None) -> ResidualDiagnosticsResult:
    """Three standard checks of the OLS assumptions underlying
    regression_inference()'s p-values and confidence intervals -- those
    numbers assume normally distributed, independent, constant-variance
    residuals, and this function checks whether that assumption
    actually held for the data given, rather than reporting inferential
    statistics whose validity was never verified.

    - Shapiro-Wilk normality test on the residuals (n >= 3 required).
    - Durbin-Watson statistic for autocorrelation (near 2 = none, near
      0 = strong positive autocorrelation, near 4 = strong negative).
    - Breusch-Pagan test for heteroscedasticity (regress squared
      residuals on the same regressors; a significant relationship
      means the residual variance isn't constant across the data)."""
    rd = regression_design_matrix(xs, ys, family, degree, expr_str, param_names)
    if rd.error:
        return ResidualDiagnosticsResult(error=rd.error)

    assert rd.design is not None and rd.target is not None  # guaranteed by the error check
    # above -- regression_design_matrix() only returns error=None alongside a populated
    # design/target pair
    design, target = rd.design, rd.target
    n, p = design.shape
    if n < 4:
        return ResidualDiagnosticsResult(error=f"Need at least 4 data points for residual "
                                                 f"diagnostics (got {n}).")

    beta, *_ = np.linalg.lstsq(design, target, rcond=None)
    resid = target - design @ beta

    normality_stat = normality_p = None
    normality_note = ""
    if n >= 3:
        try:
            normality_stat, normality_p = stats.shapiro(resid)
            normality_stat, normality_p = float(normality_stat), float(normality_p)
            normality_note = ("Residuals are consistent with normality" if normality_p > 0.05 else
                               "Residuals deviate significantly from normality") + \
                              f" (Shapiro-Wilk p={normality_p:.3g})."
        except Exception as exc:  # noqa: BLE001
            normality_note = f"Normality test could not be computed: {exc}"

    dw = float(np.sum(np.diff(resid) ** 2) / np.sum(resid ** 2)) if np.sum(resid ** 2) > 0 else float("nan")
    if dw < 1.5:
        dw_note = f"Durbin-Watson = {dw:.3f}: suggests positive autocorrelation in the residuals."
    elif dw > 2.5:
        dw_note = f"Durbin-Watson = {dw:.3f}: suggests negative autocorrelation in the residuals."
    else:
        dw_note = f"Durbin-Watson = {dw:.3f}: no strong evidence of autocorrelation (near 2 is expected)."

    het_stat = het_p = None
    het_note = ""
    if p >= 2:
        try:
            resid_sq = resid ** 2
            beta_bp, *_ = np.linalg.lstsq(design, resid_sq, rcond=None)
            yhat_bp = design @ beta_bp
            ss_res_bp = float(np.sum((resid_sq - yhat_bp) ** 2))
            ss_tot_bp = float(np.sum((resid_sq - resid_sq.mean()) ** 2))
            r2_bp = 1 - ss_res_bp / ss_tot_bp if ss_tot_bp > 0 else 0.0
            lm_stat = n * r2_bp
            lm_p = float(1 - stats.chi2.cdf(lm_stat, p - 1))
            het_stat, het_p = float(lm_stat), lm_p
            het_note = ("No significant evidence of heteroscedasticity" if het_p > 0.05 else
                        "Significant evidence of heteroscedasticity (residual variance depends on "
                        "the regressors)") + f" (Breusch-Pagan p={het_p:.3g})."
        except Exception as exc:  # noqa: BLE001
            het_note = f"Heteroscedasticity test could not be computed: {exc}"
    else:
        het_note = "Heteroscedasticity test needs at least one non-constant regressor; skipped."

    return ResidualDiagnosticsResult(
        normality_statistic=normality_stat, normality_p_value=normality_p, normality_note=normality_note,
        durbin_watson=dw, durbin_watson_note=dw_note,
        heteroscedasticity_statistic=het_stat, heteroscedasticity_p_value=het_p,
        heteroscedasticity_note=het_note)


# --------------------------------------------------------------------- Bayesian regression
@dataclass
class BayesianParameter:
    name: str
    posterior_mean: float
    posterior_std: float
    credible_lower: float
    credible_upper: float


@dataclass
class BayesianRegressionResult:
    parameters: list[BayesianParameter] = field(default_factory=list)
    credible_level: float = 0.95
    posterior_mean_sigma2: float | None = None
    posterior_predictive_fn: object = None   # callable: x0 -> (mean, lower, upper); None on error
    comparison_note: str = ""
    error: str | None = None


def bayesian_linear_regression(xs: list[float], ys: list[float], family: str, degree: int = 2,
                                expr_str: str | None = None, param_names: list[str] | None = None,
                                credible_level: float = 0.95,
                                prior_precision: float = 1e-6) -> BayesianRegressionResult:
    """Conjugate Normal-Inverse-Gamma Bayesian linear regression: a
    diffuse (weakly informative) prior beta ~ N(0, sigma^2/prior_precision * I),
    sigma^2 ~ InvGamma(1e-3, 1e-3), updated to an exact closed-form
    posterior -- no MCMC, no sampling, deliberately, to keep this
    dependency-free and instant, matching the app's local-first stance.
    With prior_precision this small the posterior necessarily tracks
    the OLS estimate closely (this is a mathematical consequence of the
    conjugate-prior update, not a coincidence -- see
    test_statistical_inference.py's convergence test, which checks this
    explicitly); the value of computing it Bayesianly here is the
    posterior predictive distribution below, which propagates BOTH
    parameter uncertainty and residual variance uncertainty into a
    single predictive interval for a new x, and the ability to move to
    a genuinely informative prior later by passing a larger
    prior_precision (a stronger pull toward the prior mean of zero) if
    real prior knowledge is available to justify it."""
    rd = regression_design_matrix(xs, ys, family, degree, expr_str, param_names)
    if rd.error:
        return BayesianRegressionResult(error=rd.error)

    assert rd.design is not None and rd.target is not None  # guaranteed by the error check
    # above -- regression_design_matrix() only returns error=None alongside a populated
    # design/target pair
    design, target = rd.design, rd.target
    n, p = design.shape
    if n <= p:
        return BayesianRegressionResult(error=f"Not enough data points ({n}) for {p} parameters.")

    lam0 = prior_precision * np.eye(p)
    a0, b0 = 1e-3, 1e-3
    XtX = design.T @ design
    Xty = design.T @ target
    try:
        Lam_n = XtX + lam0
        Lam_n_inv = np.linalg.inv(Lam_n)
    except np.linalg.LinAlgError:
        return BayesianRegressionResult(error="Design matrix is singular; cannot compute posterior.")
    mu_n = Lam_n_inv @ Xty  # prior mean is 0, so the mu0 term drops out
    a_n = a0 + n / 2
    b_n = b0 + 0.5 * (target @ target - mu_n @ Lam_n @ mu_n)
    if b_n <= 0:
        b_n = 1e-12  # numerical guard: a perfect (zero-residual) fit can push this to ~0

    dof_post = 2 * a_n
    t_crit = stats.t.ppf(1 - (1 - credible_level) / 2, dof_post)
    post_var = (b_n / a_n) * np.diag(Lam_n_inv)
    post_se = np.sqrt(np.maximum(post_var, 0.0))

    display_names = _rename_log_transformed(rd.param_names)
    parameters = []
    for name, disp_name, mean, se in zip(rd.param_names, display_names, mu_n, post_se):
        transform = rd.param_transforms.get(name, lambda v: v)
        lo, hi = mean - t_crit * se, mean + t_crit * se
        parameters.append(BayesianParameter(
            name=disp_name, posterior_mean=float(transform(mean)), posterior_std=float(se),
            credible_lower=float(transform(lo)), credible_upper=float(transform(hi))))

    def posterior_predictive(x0_row: np.ndarray) -> tuple[float, float, float]:
        """x0_row must already be in the SAME basis as `design`'s
        columns (e.g. [x0**degree, ..., x0, 1] for a polynomial) --
        callers building this from a fresh x value should reuse
        regression_design_matrix's own column construction rather than
        guessing the column order."""
        x0_row = np.asarray(x0_row, dtype=float)
        pred_mean = float(x0_row @ mu_n)
        pred_var = (b_n / a_n) * (1.0 + x0_row @ Lam_n_inv @ x0_row)
        pred_se = float(np.sqrt(max(pred_var, 0.0)))
        lo = pred_mean - t_crit * pred_se
        hi = pred_mean + t_crit * pred_se
        return pred_mean, lo, hi

    return BayesianRegressionResult(
        parameters=parameters, credible_level=credible_level, posterior_mean_sigma2=float(b_n / (a_n - 1))
        if a_n > 1 else None, posterior_predictive_fn=posterior_predictive,
        comparison_note=f"With this diffuse a prior (precision={prior_precision:.1e}), the posterior "
                          f"means above should closely track the ordinary-least-squares estimate -- "
                          f"see regression_inference() on the same data for comparison.")


# --------------------------------------------------------------------- nested model comparison
@dataclass
class NestedModelComparisonResult:
    reduced_degree: int
    full_degree: int
    f_statistic: float | None = None
    p_value: float | None = None
    reduced_r_squared: float | None = None
    full_r_squared: float | None = None
    significant: bool = False
    verification_detail: str = ""
    error: str | None = None


def compare_polynomial_degrees(xs: list[float], ys: list[float], reduced_degree: int, full_degree: int,
                                alpha: float = 0.05) -> NestedModelComparisonResult:
    """Nested-model F-test: does raising the polynomial degree from
    reduced_degree to full_degree explain significantly more variance,
    or just fit noise? H0: the extra (full_degree - reduced_degree)
    coefficients are all zero, i.e. the simpler model is just as good.
    This is the standard, principled alternative to eyeballing which
    R-squared is "close enough" to justify a more complex model --
    R-squared alone always goes up (or stays flat) with more
    parameters, so comparing R-squared values directly can't tell
    overfitting from genuine improvement; this test can."""
    if full_degree <= reduced_degree:
        return NestedModelComparisonResult(reduced_degree=reduced_degree, full_degree=full_degree,
                                             error="full_degree must be greater than reduced_degree "
                                                   "(otherwise the models aren't nested).")
    xs_a, ys_a = np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)
    n = len(xs_a)
    p_full = full_degree + 1
    if n <= p_full:
        return NestedModelComparisonResult(reduced_degree=reduced_degree, full_degree=full_degree,
                                             error=f"Need more than {p_full} data points to fit "
                                                   f"the degree-{full_degree} model (got {n}).")

    design_reduced = np.vander(xs_a, reduced_degree + 1)
    design_full = np.vander(xs_a, full_degree + 1)
    beta_r, *_ = np.linalg.lstsq(design_reduced, ys_a, rcond=None)
    beta_f, *_ = np.linalg.lstsq(design_full, ys_a, rcond=None)
    rss_r = float(np.sum((ys_a - design_reduced @ beta_r) ** 2))
    rss_f = float(np.sum((ys_a - design_full @ beta_f) ** 2))
    ss_tot = float(np.sum((ys_a - ys_a.mean()) ** 2))
    r2_r = 1 - rss_r / ss_tot if ss_tot > 0 else float("nan")
    r2_f = 1 - rss_f / ss_tot if ss_tot > 0 else float("nan")

    p_reduced = reduced_degree + 1
    dof_num = p_full - p_reduced
    dof_den = n - p_full
    if rss_f <= 0:
        f_stat, p_val = float("inf"), 0.0
    else:
        f_stat = ((rss_r - rss_f) / dof_num) / (rss_f / dof_den)
        f_stat = max(f_stat, 0.0)  # guard against tiny negative values from floating-point noise
        p_val = float(1 - stats.f.cdf(f_stat, dof_num, dof_den))

    significant = p_val < alpha
    detail = (f"The degree-{full_degree} model fits significantly better than degree-{reduced_degree} "
              f"(F={f_stat:.3g}, p={p_val:.3g} < {alpha}) -- R-squared improved from {r2_r:.4f} to "
              f"{r2_f:.4f}.") if significant else \
             (f"No significant improvement from degree-{reduced_degree} to degree-{full_degree} "
              f"(F={f_stat:.3g}, p={p_val:.3g} >= {alpha}) -- the R-squared increase from {r2_r:.4f} "
              f"to {r2_f:.4f} is consistent with fitting noise, not a genuinely better model.")

    return NestedModelComparisonResult(reduced_degree=reduced_degree, full_degree=full_degree,
                                        f_statistic=float(f_stat), p_value=p_val,
                                        reduced_r_squared=r2_r, full_r_squared=r2_f,
                                        significant=significant, verification_detail=detail)
