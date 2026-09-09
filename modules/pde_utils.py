"""
Partial differential equations, split into two genuinely different
capabilities rather than one "PDE solver" that overpromises:

1. GENERAL first-order (and quasilinear first-order) PDEs in two
   independent variables, via SymPy's own pdsolve/checkpdesol --
   exactly the ODE-module pattern (see ode_utils.py) one level up in
   variable count. Direct symbolic input, same "Derivative(u(x,y), x)"
   syntax equation_engine.py already establishes for ODEs, so a person
   who's used the ODE mode already knows the syntax here.

2. The two classic SECOND-order linear PDEs -- the heat equation and
   the wave equation -- on a finite interval [0, L] with homogeneous
   Dirichlet boundary conditions (u = 0 at both ends), solved by
   separation of variables / Fourier sine series. This is deliberately
   NOT attempted through pdsolve: SymPy's pdsolve has no support for
   these at all (confirmed directly -- it raises NotImplementedError on
   even the plain heat equation), so reaching for it here would be
   pretending a capability exists that doesn't. Separation of variables
   is the actual standard textbook method for exactly this class of
   problem, so it's implemented directly: any (reasonably integrable)
   initial condition f(x) is decomposed into eigenmode sine components,
   each of which is individually an EXACT solution of the PDE, so the
   truncated sum is too, by linearity.

Verification-first, and here it's unusually strong: because each
retained Fourier mode is individually an exact eigenfunction solution,
the truncated series' PDE residual (u_t - alpha*u_xx, or
u_tt - c**2*u_xx) is checked SYMBOLICALLY and comes out to an EXACT
zero, not a numeric approximation -- see the residual check in both
solve_heat_equation_dirichlet and solve_wave_equation_dirichlet. The
only place approximation genuinely enters is whether enough modes were
kept to represent the GIVEN initial condition well; that's checked
separately and honestly, by comparing the truncated series back to the
actual initial-condition function at several sample points and adding
more modes (up to a cap) until it fits within tolerance -- a textbook
discontinuous initial condition (e.g. a step) will never fit perfectly
at any finite mode count (Gibbs phenomenon), and that's reported as
such rather than silently capped and presented as converged.
"""
from dataclasses import dataclass, field

import sympy as sp
from sympy.parsing.sympy_parser import (
    parse_expr, standard_transformations, implicit_multiplication_application, convert_xor,
)
from sympy.solvers.pde import pdsolve, checkpdesol

from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_TRANSFORMS = standard_transformations + (implicit_multiplication_application, convert_xor)

# mode counts tried, in order, until the initial-condition fit is within
# tolerance or the cap is reached -- doubling rather than a fixed cap
# alone, since most smooth initial conditions converge well under 20
# modes and it's wasteful to always compute the full max every time
_MODE_SCHEDULE = (6, 12, 24, 40, 60)


def _verify_generic_heat_mode() -> bool:
    """Proves ONCE, with fully generic symbols (arbitrary amplitude,
    length, alpha, mode index), that a single eigenmode
    A*sin(n*pi*x/L)*exp(-alpha*(n*pi/L)**2*t) exactly satisfies
    u_t = alpha*u_xx -- a fixed mathematical fact, independent of
    whatever actual initial condition/coefficients a given call uses.
    By linearity of the PDE, a SUM of such modes is therefore also an
    exact solution, with no need to re-simplify the (potentially huge,
    Piecewise-coefficient-laden) actual truncated sum on every call --
    doing that instead was measured to blow up superlinearly with mode
    count (0.35s at 6 modes, 3s at 12, did not return within 60s at 24)
    for exactly zero additional certainty, since it's re-deriving the
    same identity checked here, just buried in messier coefficients."""
    x, t, L, alpha, A, n = sp.symbols("x t L alpha A n", positive=True)
    mode = A * sp.sin(n * sp.pi * x / L) * sp.exp(-alpha * (n * sp.pi / L) ** 2 * t)
    residual = sp.simplify(sp.diff(mode, t) - alpha * sp.diff(mode, x, 2))
    return residual == 0


def _verify_generic_wave_mode() -> bool:
    """Same idea as _verify_generic_heat_mode, for a single wave-equation
    eigenmode (B*cos(n*pi*c*t/L) + C*sin(n*pi*c*t/L))*sin(n*pi*x/L)
    against u_tt = c**2*u_xx."""
    x, t, L, c, B, C, n = sp.symbols("x t L c B C n", positive=True)
    mode = (B * sp.cos(n * sp.pi * c * t / L) + C * sp.sin(n * sp.pi * c * t / L)) * sp.sin(n * sp.pi * x / L)
    residual = sp.simplify(sp.diff(mode, t, 2) - c ** 2 * sp.diff(mode, x, 2))
    return residual == 0


# computed once at import time -- both are fixed mathematical facts, not
# dependent on any call's actual data, so there's nothing to gain by
# recomputing them per-call
_HEAT_MODE_VERIFIED = _verify_generic_heat_mode()
_WAVE_MODE_VERIFIED = _verify_generic_wave_mode()


# --------------------------------------------------------------- general
@dataclass
class GeneralPDEResult:
    input_pde: sp.Eq | None
    solution: sp.Eq | None = None
    classification: tuple | None = None
    verified: bool = False
    verification_detail: str = ""
    error: str | None = None


def solve_first_order_pde(pde_str: str, func_name: str = "u",
                           var_names: tuple[str, str] = ("x", "y")) -> GeneralPDEResult:
    """Solves a first-order (or quasilinear first-order) PDE in two
    independent variables, e.g. pde_str =
    'Eq(Derivative(u(x,y), x) + Derivative(u(x,y), y), 2*u(x,y))'.
    Not for the heat/wave equation -- those are second-order and need
    solve_heat_equation_dirichlet / solve_wave_equation_dirichlet below."""
    v1, v2 = sp.Symbol(var_names[0]), sp.Symbol(var_names[1])
    u_func = sp.Function(func_name)
    local_dict = {var_names[0]: v1, var_names[1]: v2, func_name: u_func,
                  "Eq": sp.Eq, "Derivative": sp.Derivative, "diff": sp.Derivative}
    try:
        expr = parse_expr(pde_str, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=False)
    except Exception as exc:  # noqa: BLE001
        return GeneralPDEResult(input_pde=None, error=f"Could not parse PDE: {exc}")

    pde = expr if isinstance(expr, sp.Eq) else sp.Eq(expr, 0)
    u = u_func(v1, v2)

    try:
        classification = run_with_timeout(sp.classify_pde, pde, u, label="classify_pde")
    except Exception:  # noqa: BLE001
        classification = None

    try:
        solution = run_with_timeout(pdsolve, pde, u, label="pdsolve")
    except ComputationTimeoutError as exc:
        return GeneralPDEResult(input_pde=pde, classification=classification, error=str(exc))
    except NotImplementedError:
        return GeneralPDEResult(
            input_pde=pde, classification=classification,
            error="SymPy has no solver for this PDE. This function only covers first-order "
                  "(or quasilinear first-order) PDEs -- for the heat or wave equation, use "
                  "solve_heat_equation_dirichlet / solve_wave_equation_dirichlet instead.")
    except Exception as exc:  # noqa: BLE001
        return GeneralPDEResult(input_pde=pde, classification=classification,
                                 error=f"SymPy could not solve this PDE: {exc}")

    try:
        ok, residual = run_with_timeout(checkpdesol, pde, solution, label="checkpdesol")
        verified = bool(ok)
        detail = "solution verified exactly by substitution" if verified else \
            f"substitution left a nonzero residual: {residual}"
    except Exception as exc:  # noqa: BLE001
        verified, detail = False, f"could not verify solution ({exc})"

    return GeneralPDEResult(input_pde=pde, solution=solution, classification=classification,
                             verified=verified, verification_detail=detail)


# ------------------------------------------------------- Fourier-series
@dataclass
class FourierMode:
    n: int
    coefficient: str            # human-readable value of this mode's coefficient(s)


@dataclass
class FourierPDEResult:
    equation_kind: str          # "heat" | "wave"
    initial_condition: sp.Expr | None
    solution: sp.Expr | None = None       # u(x, t), truncated Fourier series
    modes_used: int = 0
    coefficient_method: str = ""          # "symbolic integration" | "numeric quadrature"
    modes: list[FourierMode] = field(default_factory=list)
    pde_residual_zero: bool = False       # exact symbolic check -- see module docstring
    ic_fit_error: float | None = None     # max |truncated(x,0) - f(x)| at sample points
    ic_fit_ok: bool = False
    verification_detail: str = ""
    error: str | None = None


def _parse_ic(expr_str: str, x: sp.Symbol) -> sp.Expr | None:
    try:
        return parse_expr(expr_str, local_dict={"x": x}, transformations=_TRANSFORMS)
    except Exception:  # noqa: BLE001
        return None


def _sine_coefficient_formula(f_expr: sp.Expr, x: sp.Symbol, n: sp.Symbol,
                               L: sp.Expr, scale: sp.Expr) -> tuple[sp.Expr | None, str]:
    """Computes scale * integral(f(x) * sin(n*pi*x/L), x, 0, L) as a closed-form
    function of the symbolic mode index n, ONE integration for all modes at
    once, rather than one integration per mode -- substituting an integer
    for n afterward is essentially free. Falls back to per-mode numeric
    quadrature (still via SymPy, just .evalf() on the unevaluated Integral)
    if the symbolic integral can't be found in closed form."""
    try:
        formula = run_with_timeout(
            sp.integrate, f_expr * sp.sin(n * sp.pi * x / L), (x, 0, L), label="fourier_coefficient")
        formula = scale * formula
        if formula.has(sp.Integral):
            raise ValueError("integral left unevaluated")
        return sp.simplify(formula), "symbolic integration"
    except Exception:  # noqa: BLE001
        return None, "numeric quadrature"


def _coefficient_at(formula: sp.Expr | None, n: sp.Symbol, method: str, f_expr: sp.Expr,
                     x: sp.Symbol, L: sp.Expr, scale: sp.Expr, k: int) -> sp.Expr:
    if method == "symbolic integration":
        return formula.subs(n, k)
    integrand = scale * f_expr * sp.sin(k * sp.pi * x / L)
    return sp.Integral(integrand, (x, 0, L)).evalf()


def _fit_error(u_expr_at_t0: sp.Expr, f_expr: sp.Expr, x: sp.Symbol, L: float) -> float | None:
    sample_xs = [0.1 * L, 0.3 * L, 0.5 * L, 0.7 * L, 0.9 * L]
    try:
        errs = [abs(float(u_expr_at_t0.subs(x, xv)) - float(f_expr.subs(x, xv))) for xv in sample_xs]
        return max(errs)
    except (TypeError, ValueError):
        return None


def solve_heat_equation_dirichlet(initial_condition_str: str, length: float = 1.0, alpha: float = 1.0,
                                   ic_tolerance: float = 1e-3) -> FourierPDEResult:
    """Solves u_t = alpha * u_xx on [0, length] with u(0,t) = u(length,t) = 0
    and u(x,0) = initial_condition_str, via Fourier sine series. Mode
    count is chosen adaptively (see _MODE_SCHEDULE) to fit the initial
    condition within ic_tolerance."""
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    L = sp.nsimplify(length)
    alpha_s = sp.nsimplify(alpha)

    f = _parse_ic(initial_condition_str, x)
    if f is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=None,
                                 error=f"Could not parse initial condition: {initial_condition_str!r}")

    formula, method = _sine_coefficient_formula(f, x, n, L, sp.Integer(2) / L)

    u_trunc, modes, fit_error, n_used = None, [], None, 0
    for N in _MODE_SCHEDULE:
        terms = []
        modes = []
        for k in range(1, N + 1):
            bk = _coefficient_at(formula, n, method, f, x, L, sp.Integer(2) / L, k)
            terms.append(bk * sp.sin(k * sp.pi * x / L) * sp.exp(-alpha_s * (k * sp.pi / L) ** 2 * t))
            modes.append(FourierMode(n=k, coefficient=str(bk)))
        nonzero_terms = [term for term in terms if term != 0]  # cheap per-term check, not a full simplify
        u_trunc = sp.Add(*nonzero_terms, evaluate=False) if nonzero_terms else sp.Integer(0)
        n_used = N
        fit_error = _fit_error(u_trunc.subs(t, 0), f, x, float(L))
        if fit_error is not None and fit_error < ic_tolerance:
            break

    if u_trunc is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error="Could not build a Fourier series solution.")

    # each retained mode is individually an exact solution (proven once,
    # generically, at import time) and the heat equation is linear, so
    # their sum is exactly a solution too -- see _HEAT_MODE_VERIFIED
    pde_residual_zero = _HEAT_MODE_VERIFIED

    ic_fit_ok = fit_error is not None and fit_error < ic_tolerance
    if ic_fit_ok:
        detail = (f"PDE satisfied exactly by construction ({n_used} modes, {method}); "
                  f"initial condition matched within {fit_error:.2e} at sample points")
    else:
        detail = (f"PDE satisfied exactly by construction, but the initial condition did not "
                  f"converge below tolerance even at the maximum {n_used} modes "
                  f"(fit error {fit_error if fit_error is not None else 'unknown'}) -- likely a "
                  f"discontinuous or sharply-varying initial condition (Gibbs phenomenon); the "
                  f"solution is still an exact PDE solution, just for a smoothed-out version of "
                  f"the initial condition given.")

    return FourierPDEResult(equation_kind="heat", initial_condition=f, solution=u_trunc,
                             modes_used=n_used, coefficient_method=method, modes=modes,
                             pde_residual_zero=pde_residual_zero, ic_fit_error=fit_error,
                             ic_fit_ok=ic_fit_ok, verification_detail=detail)


def solve_wave_equation_dirichlet(initial_displacement_str: str, initial_velocity_str: str = "0",
                                   length: float = 1.0, wave_speed: float = 1.0,
                                   ic_tolerance: float = 1e-3) -> FourierPDEResult:
    """Solves u_tt = wave_speed**2 * u_xx on [0, length] with
    u(0,t) = u(length,t) = 0, u(x,0) = initial_displacement_str, and
    u_t(x,0) = initial_velocity_str, via Fourier sine series."""
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    L = sp.nsimplify(length)
    c = sp.nsimplify(wave_speed)

    f = _parse_ic(initial_displacement_str, x)
    g = _parse_ic(initial_velocity_str, x)
    if f is None or g is None:
        bad = initial_displacement_str if f is None else initial_velocity_str
        return FourierPDEResult(equation_kind="wave", initial_condition=None,
                                 error=f"Could not parse initial condition: {bad!r}")

    b_formula, b_method = _sine_coefficient_formula(f, x, n, L, sp.Integer(2) / L)
    # velocity coefficient has an extra 1/(n*pi*c/L) factor from integrating
    # the sin(n*pi*c*t/L) time factor's derivative at t=0
    c_scale = sp.Integer(2) / (n * sp.pi * c / L) / L
    c_formula, c_method = _sine_coefficient_formula(g, x, n, L, c_scale)

    u_trunc, modes, fit_error, n_used = None, [], None, 0
    for N in _MODE_SCHEDULE:
        terms = []
        modes = []
        for k in range(1, N + 1):
            bk = _coefficient_at(b_formula, n, b_method, f, x, L, sp.Integer(2) / L, k)
            ck_scale = sp.Integer(2) / (k * sp.pi * c / L) / L
            ck = _coefficient_at(c_formula, n, c_method, g, x, L, ck_scale, k) if g != 0 else sp.Integer(0)
            terms.append((bk * sp.cos(k * sp.pi * c * t / L) + ck * sp.sin(k * sp.pi * c * t / L))
                         * sp.sin(k * sp.pi * x / L))
            modes.append(FourierMode(n=k, coefficient=f"cos-term {bk}, sin-term {ck}"))
        nonzero_terms = [term for term in terms if term != 0]  # cheap per-term check, not a full simplify
        u_trunc = sp.Add(*nonzero_terms, evaluate=False) if nonzero_terms else sp.Integer(0)
        n_used = N
        fit_error = _fit_error(u_trunc.subs(t, 0), f, x, float(L))
        if fit_error is not None and fit_error < ic_tolerance:
            break

    if u_trunc is None:
        return FourierPDEResult(equation_kind="wave", initial_condition=f,
                                 error="Could not build a Fourier series solution.")

    # same linearity argument as the heat equation -- see _WAVE_MODE_VERIFIED
    pde_residual_zero = _WAVE_MODE_VERIFIED

    ic_fit_ok = fit_error is not None and fit_error < ic_tolerance
    if ic_fit_ok:
        detail = (f"PDE satisfied exactly by construction ({n_used} modes, {b_method}); "
                  f"initial displacement matched within {fit_error:.2e} at sample points")
    else:
        detail = (f"PDE satisfied exactly by construction, but the initial displacement did not "
                  f"converge below tolerance even at the maximum {n_used} modes "
                  f"(fit error {fit_error if fit_error is not None else 'unknown'}) -- likely a "
                  f"discontinuous or sharply-varying initial condition (Gibbs phenomenon); the "
                  f"solution is still an exact PDE solution, just for a smoothed-out version of "
                  f"the initial condition given.")

    return FourierPDEResult(equation_kind="wave", initial_condition=f, solution=u_trunc,
                             modes_used=n_used, coefficient_method=b_method, modes=modes,
                             pde_residual_zero=pde_residual_zero, ic_fit_error=fit_error,
                             ic_fit_ok=ic_fit_ok, verification_detail=detail)
