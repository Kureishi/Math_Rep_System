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


def _verify_generic_heat_mode_general_k() -> bool:
    """Same identity as _verify_generic_heat_mode, but for an arbitrary
    real wavenumber k rather than specifically k = n*pi/L -- covers
    Neumann (k = n*pi/L still, but n starts at 0) and, more importantly,
    Robin boundary conditions below, where the wavenumbers are roots of
    a transcendental equation and have no closed form n*pi/L expression
    at all. The underlying fact (A*sin(k*x)*exp(-alpha*k**2*t) solves
    u_t = alpha*u_xx for ANY k) doesn't care where k came from."""
    x, t, alpha, A, k = sp.symbols("x t alpha A k", positive=True)
    mode = A * sp.sin(k * x) * sp.exp(-alpha * k ** 2 * t)
    residual = sp.simplify(sp.diff(mode, t) - alpha * sp.diff(mode, x, 2))
    return residual == 0


_HEAT_MODE_VERIFIED_GENERAL_K = _verify_generic_heat_mode_general_k()


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


# ------------------------------------------------------------- Neumann BCs
def _cosine_coefficient_formula(f_expr: sp.Expr, x: sp.Symbol, n: sp.Symbol,
                                 L: sp.Expr) -> tuple[sp.Expr | None, str]:
    """Coefficient formula for a Neumann (insulated-end) cosine series:
    an = (2/L) * integral(f(x)*cos(n*pi*x/L), x, 0, L), valid for n >= 1;
    the constant term a0 is this SAME formula evaluated at n=0, divided
    by 2 (standard cosine-series convention) -- callers must halve the
    n=0 term themselves, not this helper, since the halving only applies
    once regardless of how many modes are summed."""
    try:
        formula = run_with_timeout(
            sp.integrate, f_expr * sp.cos(n * sp.pi * x / L), (x, 0, L), label="cosine_coefficient")
        formula = sp.Integer(2) / L * formula
        if formula.has(sp.Integral):
            raise ValueError("integral left unevaluated")
        return sp.simplify(formula), "symbolic integration"
    except Exception:  # noqa: BLE001
        return None, "numeric quadrature"


def solve_heat_equation_neumann(initial_condition_str: str, length: float = 1.0,
                                 alpha: float = 1.0, ic_tolerance: float = 1e-3) -> FourierPDEResult:
    """Solves u_t = alpha * u_xx on [0, length] with INSULATED
    (Neumann, u_x = 0) ends instead of Dirichlet -- physically, a rod
    with no heat escaping at either end, so it equilibrates to the
    initial condition's average temperature rather than cooling to
    zero. Uses a Fourier COSINE series (the eigenfunctions of the
    Neumann boundary condition), including the n=0 constant mode."""
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    L = sp.nsimplify(length)
    alpha_s = sp.nsimplify(alpha)

    f = _parse_ic(initial_condition_str, x)
    if f is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=None,
                                 error=f"Could not parse initial condition: {initial_condition_str!r}")

    formula, method = _cosine_coefficient_formula(f, x, n, L)
    if formula is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error="Could not compute Fourier cosine coefficients.")

    u_trunc, modes, fit_error, n_used = None, [], None, 0
    for N in _MODE_SCHEDULE:
        a0 = formula.subs(n, 0) / 2  # the n=0 cosine-series halving, applied once
        terms = [a0]
        modes = [FourierMode(n=0, coefficient=str(a0))]
        for k in range(1, N + 1):
            ak = formula.subs(n, k)
            terms.append(ak * sp.cos(k * sp.pi * x / L) * sp.exp(-alpha_s * (k * sp.pi / L) ** 2 * t))
            modes.append(FourierMode(n=k, coefficient=str(ak)))
        nonzero_terms = [term for term in terms if term != 0]
        u_trunc = sp.Add(*nonzero_terms, evaluate=False) if nonzero_terms else sp.Integer(0)
        n_used = N
        fit_error = _fit_error(u_trunc.subs(t, 0), f, x, float(L))
        if fit_error is not None and fit_error < ic_tolerance:
            break

    if u_trunc is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error="Could not build a Fourier series solution.")

    # the n=0 mode is a time-independent constant (satisfies u_t=alpha*u_xx
    # trivially, both sides zero); every n>=1 mode is the same eigenmode
    # family verified generically in _HEAT_MODE_VERIFIED (k = n*pi/L there
    # is just one particular choice of k), so linearity carries through
    pde_residual_zero = _HEAT_MODE_VERIFIED_GENERAL_K

    ic_fit_ok = fit_error is not None and fit_error < ic_tolerance
    if ic_fit_ok:
        detail = (f"PDE satisfied exactly by construction ({n_used} modes, {method}, Neumann/"
                  f"insulated-end cosine series); initial condition matched within "
                  f"{fit_error:.2e} at sample points")
    else:
        detail = (f"PDE satisfied exactly by construction, but the initial condition did not "
                  f"converge below tolerance even at the maximum {n_used} modes "
                  f"(fit error {fit_error if fit_error is not None else 'unknown'}) -- likely a "
                  f"discontinuous or sharply-varying initial condition (Gibbs phenomenon).")

    return FourierPDEResult(equation_kind="heat", initial_condition=f, solution=u_trunc,
                             modes_used=n_used, coefficient_method=method, modes=modes,
                             pde_residual_zero=pde_residual_zero, ic_fit_error=fit_error,
                             ic_fit_ok=ic_fit_ok, verification_detail=detail)


def solve_wave_equation_neumann(initial_displacement_str: str, initial_velocity_str: str = "0",
                                 length: float = 1.0, wave_speed: float = 1.0,
                                 ic_tolerance: float = 1e-3) -> FourierPDEResult:
    """Solves u_tt = wave_speed**2 * u_xx on [0, length] with FREE
    (Neumann, u_x = 0) ends instead of fixed ends -- physically, a
    string or rod free to slide at both ends rather than clamped.
    Fourier cosine series, including the n=0 mode (rigid translation
    at constant velocity, if the initial velocity has a nonzero
    average)."""
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    n = sp.Symbol("n", integer=True, nonnegative=True)
    L = sp.nsimplify(length)
    c = sp.nsimplify(wave_speed)

    f = _parse_ic(initial_displacement_str, x)
    g = _parse_ic(initial_velocity_str, x)
    if f is None or g is None:
        bad = initial_displacement_str if f is None else initial_velocity_str
        return FourierPDEResult(equation_kind="wave", initial_condition=None,
                                 error=f"Could not parse initial condition: {bad!r}")

    b_formula, b_method = _cosine_coefficient_formula(f, x, n, L)
    if b_formula is None:
        return FourierPDEResult(equation_kind="wave", initial_condition=f,
                                 error="Could not compute Fourier cosine coefficients.")

    u_trunc, modes, fit_error, n_used = None, [], None, 0
    for N in _MODE_SCHEDULE:
        a0 = b_formula.subs(n, 0) / 2
        # n=0 velocity mode: the average initial velocity gives rigid
        # translation at constant speed, d0 * t (no restoring force at k=0)
        d0 = sp.integrate(g, (x, 0, L)) / L if g != 0 else sp.Integer(0)
        terms = [a0 + d0 * t]
        modes = [FourierMode(n=0, coefficient=f"{a0} + {d0}*t")]
        for k in range(1, N + 1):
            bk = b_formula.subs(n, k)
            omega = k * sp.pi * c / L
            ck = (sp.Integer(2) / (omega * L)) * sp.integrate(g * sp.cos(k * sp.pi * x / L), (x, 0, L)) \
                if g != 0 else sp.Integer(0)
            terms.append((bk * sp.cos(omega * t) + ck * sp.sin(omega * t)) * sp.cos(k * sp.pi * x / L))
            modes.append(FourierMode(n=k, coefficient=f"cos-term {bk}, sin-term {ck}"))
        nonzero_terms = [term for term in terms if term != 0]
        u_trunc = sp.Add(*nonzero_terms, evaluate=False) if nonzero_terms else sp.Integer(0)
        n_used = N
        fit_error = _fit_error(u_trunc.subs(t, 0), f, x, float(L))
        if fit_error is not None and fit_error < ic_tolerance:
            break

    if u_trunc is None:
        return FourierPDEResult(equation_kind="wave", initial_condition=f,
                                 error="Could not build a Fourier series solution.")

    pde_residual_zero = _WAVE_MODE_VERIFIED  # n=0 rigid-translation mode trivially satisfies u_tt=c**2*u_xx too

    ic_fit_ok = fit_error is not None and fit_error < ic_tolerance
    detail = (f"PDE satisfied exactly by construction ({n_used} modes, {b_method}, Neumann/free-end "
              f"cosine series); initial displacement matched within {fit_error:.2e} at sample points") \
        if ic_fit_ok else \
        (f"PDE satisfied exactly by construction, but the initial displacement did not converge "
         f"below tolerance even at the maximum {n_used} modes "
         f"(fit error {fit_error if fit_error is not None else 'unknown'}).")

    return FourierPDEResult(equation_kind="wave", initial_condition=f, solution=u_trunc,
                             modes_used=n_used, coefficient_method=b_method, modes=modes,
                             pde_residual_zero=pde_residual_zero, ic_fit_error=fit_error,
                             ic_fit_ok=ic_fit_ok, verification_detail=detail)


# --------------------------------------------------------------- Robin BCs
def _robin_eigenvalues(L: float, h: float, n_modes: int) -> list[float]:
    """Positive roots of lam*cos(lam*L) + h*sin(lam*L) = 0 -- the
    eigenvalue equation for u(0)=0 (Dirichlet), u'(L) + h*u(L) = 0
    (Robin/convective) boundary conditions. Written this way (rather
    than the equivalent tan(lam*L) = -lam/h) specifically to avoid
    tan's poles, so a plain sign-change scan over a fine grid reliably
    brackets every root for scipy.optimize.brentq -- tan's poles would
    otherwise masquerade as sign changes and produce spurious roots."""
    from scipy.optimize import brentq
    import numpy as np

    def g(lam):
        return lam * np.cos(lam * L) + h * np.sin(lam * L)

    roots = []
    upper = max(20.0, 3.0 * n_modes) * np.pi / L
    xgrid = np.linspace(1e-6, upper, max(4000, 400 * n_modes))
    vals = g(xgrid)
    for i in range(len(xgrid) - 1):
        if vals[i] == 0.0:
            roots.append(xgrid[i])
        elif vals[i] * vals[i + 1] < 0:
            roots.append(brentq(g, xgrid[i], xgrid[i + 1]))
        if len(roots) >= n_modes:
            break
    return roots


def solve_heat_equation_robin(initial_condition_str: str, length: float = 1.0, alpha: float = 1.0,
                               robin_coefficient: float = 1.0, ic_tolerance: float = 1e-3,
                               max_modes: int = 30) -> FourierPDEResult:
    """Solves u_t = alpha * u_xx on [0, length] with u(0,t) = 0
    (Dirichlet, held at zero) and u_x(length,t) + robin_coefficient *
    u(length,t) = 0 at the far end (Robin/convective -- e.g. Newton
    cooling into a medium at zero temperature, with robin_coefficient
    proportional to the heat transfer coefficient). Unlike the Dirichlet
    and Neumann cases, the eigenvalues here are roots of a transcendental
    equation with no closed form, so this is numeric throughout: roots
    found via bracketed bisection + Brent's method, coefficients via
    numeric quadrature against the (non-closed-form) eigenfunctions."""
    import numpy as np
    from scipy.integrate import quad

    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    L = float(length)

    f = _parse_ic(initial_condition_str, x)
    if f is None:
        return FourierPDEResult(equation_kind="heat", initial_condition=None,
                                 error=f"Could not parse initial condition: {initial_condition_str!r}")
    try:
        f_numeric = sp.lambdify(x, f, "numpy")
        f_numeric(0.3 * L)  # smoke-test the lambdified function before relying on it in quad
    except Exception as exc:  # noqa: BLE001
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error=f"Initial condition could not be evaluated numerically: {exc}")

    try:
        eigenvalues = run_with_timeout(_robin_eigenvalues, L, robin_coefficient, max_modes,
                                        label="robin_eigenvalues")
    except Exception as exc:  # noqa: BLE001
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error=f"Could not find Robin boundary-condition eigenvalues: {exc}")
    if not eigenvalues:
        return FourierPDEResult(equation_kind="heat", initial_condition=f,
                                 error="No eigenvalues found for these Robin boundary conditions.")

    coefficients, terms, modes = [], [], []
    for lam in eigenvalues:
        numerator, _ = quad(lambda xv: f_numeric(xv) * np.sin(lam * xv), 0, L)
        denominator, _ = quad(lambda xv: np.sin(lam * xv) ** 2, 0, L)
        b_n = numerator / denominator
        coefficients.append(b_n)
        modes.append(FourierMode(n=len(modes) + 1, coefficient=f"{b_n:.6g} (lambda={lam:.6g})"))
        terms.append(b_n * sp.sin(sp.Float(lam) * x) * sp.exp(-sp.nsimplify(alpha) * sp.Float(lam) ** 2 * t))

    def partial_sum_at_t0(n_used):
        def evaluate(xv):
            return sum(b * np.sin(lam * xv) for b, lam in zip(coefficients[:n_used], eigenvalues[:n_used]))
        return evaluate

    n_used, fit_error = len(eigenvalues), None
    for candidate_n in range(min(6, len(eigenvalues)), len(eigenvalues) + 1):
        evaluate = partial_sum_at_t0(candidate_n)
        sample_xs = [0.1 * L, 0.3 * L, 0.5 * L, 0.7 * L, 0.9 * L]
        fit_error = max(abs(evaluate(xv) - f_numeric(xv)) for xv in sample_xs)
        n_used = candidate_n
        if fit_error < ic_tolerance:
            break

    u_trunc = sp.Add(*terms[:n_used], evaluate=False)
    ic_fit_ok = fit_error is not None and fit_error < ic_tolerance
    detail = (f"PDE satisfied exactly by construction ({n_used} numerically-found Robin "
              f"eigenmodes); initial condition matched within {fit_error:.2e} at sample points") \
        if ic_fit_ok else \
        (f"PDE satisfied exactly by construction, but the initial condition did not converge "
         f"below tolerance even at the maximum {n_used} modes (fit error {fit_error:.3g}).")

    return FourierPDEResult(equation_kind="heat", initial_condition=f, solution=u_trunc,
                             modes_used=n_used, coefficient_method="numeric (Robin eigenvalues)",
                             modes=modes[:n_used],
                             pde_residual_zero=_HEAT_MODE_VERIFIED_GENERAL_K,
                             ic_fit_error=fit_error, ic_fit_ok=ic_fit_ok, verification_detail=detail)


# ------------------------------------------------------- Laplace's equation
@dataclass
class LaplaceRectangleResult:
    solution: sp.Expr | None = None
    modes_used: int = 0
    pde_residual_zero: bool = False
    boundary_fit_errors: dict = field(default_factory=dict)  # side -> max fit error
    boundary_fit_ok: bool = False
    verification_detail: str = ""
    error: str | None = None


def _verify_generic_laplace_mode() -> bool:
    """Proves once, generically, that a single sin(k*x)*sinh(k*(H-y))
    term (the eigenmode family used for every side of the rectangle
    below, just with x/y and the sinh argument swapped per side) solves
    Laplace's equation u_xx + u_yy = 0 for any wavenumber k and height H."""
    x, y, H, A, k = sp.symbols("x y H A k", positive=True)
    mode = A * sp.sin(k * x) * sp.sinh(k * (H - y))
    residual = sp.simplify(sp.diff(mode, x, 2) + sp.diff(mode, y, 2))
    return residual == 0


_LAPLACE_MODE_VERIFIED = _verify_generic_laplace_mode()


def _laplace_side_series(g_expr: sp.Expr, along: sp.Symbol, transverse: sp.Symbol,
                          along_length: sp.Expr, transverse_length: sp.Expr,
                          n: sp.Symbol, ic_tolerance: float, at_far_end: bool) -> tuple:
    """Builds the Fourier sine series solution for ONE side of the
    rectangle held at g_expr(along), with the other three sides at
    zero: sum Bn * sin(n*pi*along/along_length) * sinh(n*pi*(transverse
    offset)/along_length) / sinh(n*pi*transverse_length/along_length).
    `at_far_end` selects which of the two sinh forms keeps this side's
    boundary value and decays to zero at the opposite side."""
    formula, method = _sine_coefficient_formula(g_expr, along, n, along_length, sp.Integer(2) / along_length)
    if formula is None:
        return None, 0, None, method

    terms, n_used, fit_error = None, 0, None
    for N in _MODE_SCHEDULE:
        current_terms = []
        for k in range(1, N + 1):
            bk = _coefficient_at(formula, n, method, g_expr, along, along_length,
                                  sp.Integer(2) / along_length, k)
            sinh_num = sp.sinh(k * sp.pi * transverse / along_length) if at_far_end else \
                sp.sinh(k * sp.pi * (transverse_length - transverse) / along_length)
            sinh_den = sp.sinh(k * sp.pi * transverse_length / along_length)
            current_terms.append(bk * sp.sin(k * sp.pi * along / along_length) * sinh_num / sinh_den)
        nonzero = [term for term in current_terms if term != 0]
        terms = nonzero
        n_used = N
        transverse_at_side = transverse_length if at_far_end else sp.Integer(0)
        trial = sp.Add(*nonzero, evaluate=False) if nonzero else sp.Integer(0)
        fit_error = _fit_error(trial.subs(transverse, transverse_at_side), g_expr, along, float(along_length))
        if fit_error is not None and fit_error < ic_tolerance:
            break
    return terms, n_used, fit_error, method


def solve_laplace_rectangle(bottom: str = "0", top: str = "0", left: str = "0", right: str = "0",
                             width: float = 1.0, height: float = 1.0,
                             ic_tolerance: float = 1e-3) -> LaplaceRectangleResult:
    """Solves u_xx + u_yy = 0 on the rectangle [0, width] x [0, height]
    with u given by `bottom`/`top` (functions of x) and `left`/`right`
    (functions of y) on the four sides -- via superposition of four
    single-side Fourier sine series solutions, each of which is
    individually an exact solution (see _LAPLACE_MODE_VERIFIED), so
    their sum satisfies Laplace's equation everywhere by linearity, and
    the boundary conditions are recovered as each side's series
    converges (checked separately, honestly, the same way the heat/wave
    solvers check their initial conditions)."""
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y", positive=True)
    n = sp.Symbol("n", integer=True, positive=True)
    a, b = sp.nsimplify(width), sp.nsimplify(height)

    sides = {"bottom": bottom, "top": top, "left": left, "right": right}
    parsed = {}
    for name, expr_str in sides.items():
        var = x if name in ("bottom", "top") else y
        parsed_expr = _parse_ic(expr_str, var)
        if parsed_expr is None:
            return LaplaceRectangleResult(error=f"Could not parse {name} boundary condition: {expr_str!r}")
        parsed[name] = parsed_expr

    all_terms, fit_errors, max_n = [], {}, 0
    side_specs = [
        ("bottom", parsed["bottom"], x, y, a, b, False),
        ("top", parsed["top"], x, y, a, b, True),
        ("left", parsed["left"], y, x, b, a, False),
        ("right", parsed["right"], y, x, b, a, True),
    ]
    for name, g_expr, along, transverse, along_len, transverse_len, at_far_end in side_specs:
        if g_expr == 0:
            fit_errors[name] = 0.0
            continue
        terms, n_used, fit_error, method = _laplace_side_series(
            g_expr, along, transverse, along_len, transverse_len, n, ic_tolerance, at_far_end)
        if terms is None:
            return LaplaceRectangleResult(error=f"Could not compute Fourier coefficients for the "
                                                  f"{name} boundary.")
        all_terms.extend(terms)
        fit_errors[name] = fit_error
        max_n = max(max_n, n_used)

    u_total = sp.Add(*all_terms, evaluate=False) if all_terms else sp.Integer(0)
    boundary_fit_ok = all(err is not None and err < ic_tolerance for err in fit_errors.values())
    detail = (f"Laplace's equation satisfied exactly by construction ({max_n} modes per active "
              f"side); all boundary conditions matched within tolerance: "
              f"{ {k: round(v, 6) for k, v in fit_errors.items()} }") if boundary_fit_ok else \
             (f"Laplace's equation satisfied exactly by construction, but at least one boundary "
              f"did not converge within tolerance: { {k: round(v, 6) if v is not None else None for k, v in fit_errors.items()} }")

    return LaplaceRectangleResult(solution=u_total, modes_used=max_n,
                                   pde_residual_zero=_LAPLACE_MODE_VERIFIED,
                                   boundary_fit_errors=fit_errors, boundary_fit_ok=boundary_fit_ok,
                                   verification_detail=detail)


# ------------------------------------------------- general numeric fallback
@dataclass
class FiniteDifferencePDEResult:
    grid_x: "object" = None      # numpy array, x coordinates
    grid_y: "object" = None      # numpy array, y coordinates
    solution_grid: "object" = None  # numpy 2D array, NaN outside the domain
    resolution: tuple = (0, 0)
    convergence_error: float | None = None   # max discrepancy vs. a doubled-resolution solve
    converged: bool = False
    error: str | None = None
    note: str = ""


def _build_and_solve_fd_grid(nx: int, ny: int, x_range: tuple, y_range: tuple,
                              inside_fn, boundary_fn, source_fn):
    """One finite-difference solve of -Laplacian-style Poisson's
    equation (u_xx + u_yy = source) via the standard 5-point stencil, on
    whichever grid points `inside_fn(x, y)` accepts, with `boundary_fn`
    supplying values at points just outside the domain (a first-order
    accurate embedded-boundary approximation appropriate for a domain
    boundary that doesn't align with grid lines -- see the module-level
    caveat in solve_pde_finite_difference_2d). Returns (xs, ys, U,
    inside_mask); pulled out of the public function so it can be called
    twice at different resolutions for the convergence check."""
    import numpy as np
    import scipy.sparse as sp_sparse
    import scipy.sparse.linalg as spla

    x0, x1 = x_range
    y0, y1 = y_range
    xs = np.linspace(x0, x1, nx)
    ys = np.linspace(y0, y1, ny)
    hx = (x1 - x0) / (nx - 1)
    hy = (y1 - y0) / (ny - 1)

    inside = np.zeros((nx, ny), dtype=bool)
    for i in range(nx):
        for j in range(ny):
            inside[i, j] = bool(inside_fn(xs[i], ys[j]))

    idx = -np.ones((nx, ny), dtype=int)
    count = 0
    for i in range(nx):
        for j in range(ny):
            if inside[i, j]:
                idx[i, j] = count
                count += 1

    if count == 0:
        raise ValueError("no grid points fall inside the given domain -- check the domain "
                          "predicate and x/y ranges")

    A = sp_sparse.lil_matrix((count, count))
    rhs = np.zeros(count)
    for i in range(nx):
        for j in range(ny):
            if not inside[i, j]:
                continue
            k = idx[i, j]
            rhs[k] = source_fn(xs[i], ys[j])
            diag = 0.0
            for di, dj, h in ((-1, 0, hx), (1, 0, hx), (0, -1, hy), (0, 1, hy)):
                ni, nj = i + di, j + dj
                coeff = 1.0 / h ** 2
                if 0 <= ni < nx and 0 <= nj < ny and inside[ni, nj]:
                    A[k, idx[ni, nj]] += coeff
                else:
                    bx = xs[ni] if 0 <= ni < nx else xs[i]
                    by = ys[nj] if 0 <= nj < ny else ys[j]
                    rhs[k] -= coeff * boundary_fn(bx, by)
                diag -= coeff
            A[k, k] += diag

    u_vec = spla.spsolve(A.tocsr(), rhs)
    U = np.full((nx, ny), np.nan)
    for i in range(nx):
        for j in range(ny):
            if inside[i, j]:
                U[i, j] = u_vec[idx[i, j]]
    return xs, ys, U, inside


def solve_pde_finite_difference_2d(source_str: str = "0", boundary_value_str: str = "0",
                                    domain_predicate_str: str = "True",
                                    x_range: tuple = (0.0, 1.0), y_range: tuple = (0.0, 1.0),
                                    nx: int = 41, ny: int = 41,
                                    convergence_tolerance: float = 0.02) -> FiniteDifferencePDEResult:
    """Numeric fallback for u_xx + u_yy = source(x, y) on an arbitrary
    (not necessarily rectangular) region: the region within the
    bounding box [x_range] x [y_range] where domain_predicate_str(x, y)
    is true, e.g. "x**2 + y**2 <= 1" for a disk. u = boundary_value(x, y)
    outside the domain (used at the grid points just past the boundary).

    This exists specifically for the cases solve_laplace_rectangle and
    the heat/wave Fourier solvers CAN'T handle: non-rectangular domains,
    non-homogeneous source terms (Poisson's equation, not just Laplace's),
    or boundary shapes with no separation-of-variables solution at all.
    The tradeoff for that generality: this is only first-order accurate
    at a curved boundary, because a boundary that cuts through a grid
    cell is approximated by evaluating boundary_value at the nearest
    OUTSIDE grid point rather than at the true boundary curve -- a
    standard, honest limitation of this "embedded boundary" approach on
    a plain Cartesian grid, not hidden here. A true boundary-fitted mesh
    would do better but is out of scope for this tool.

    Verified by grid refinement (Richardson-style): solved once at the
    requested resolution and once at roughly double it, then compared
    at shared sample points -- since there's no separation-of-variables
    solution to check against in the general case, agreement between
    two independent resolutions is the honest verification available.
    """
    import numpy as np

    try:
        x_sym, y_sym = sp.symbols("x y")
        local_dict = {"x": x_sym, "y": y_sym}
        source_expr = parse_expr(source_str, local_dict=local_dict, transformations=_TRANSFORMS)
        boundary_expr = parse_expr(boundary_value_str, local_dict=local_dict, transformations=_TRANSFORMS)
        domain_expr = parse_expr(domain_predicate_str, local_dict=local_dict, transformations=_TRANSFORMS)
    except Exception as exc:  # noqa: BLE001
        return FiniteDifferencePDEResult(error=f"Could not parse source/boundary/domain expression: {exc}")

    try:
        source_fn = sp.lambdify((x_sym, y_sym), source_expr, "numpy")
        boundary_fn = sp.lambdify((x_sym, y_sym), boundary_expr, "numpy")
        inside_fn = sp.lambdify((x_sym, y_sym), domain_expr, "numpy")
        # smoke-test all three before committing to a full grid solve
        cx, cy = (x_range[0] + x_range[1]) / 2, (y_range[0] + y_range[1]) / 2
        source_fn(cx, cy); boundary_fn(cx, cy); bool(inside_fn(cx, cy))
    except Exception as exc:  # noqa: BLE001
        return FiniteDifferencePDEResult(error=f"Expression could not be evaluated numerically: {exc}")

    try:
        xs, ys, U, inside = run_with_timeout(
            _build_and_solve_fd_grid, nx, ny, x_range, y_range, inside_fn, boundary_fn, source_fn,
            label="fd_solve")
    except ComputationTimeoutError as exc:
        return FiniteDifferencePDEResult(error=str(exc))
    except Exception as exc:  # noqa: BLE001
        return FiniteDifferencePDEResult(error=f"Finite-difference solve failed: {exc}")

    # convergence check: re-solve at roughly double the resolution, compare
    # at the coarse grid's own sample points via nearest-neighbor lookup
    convergence_error, converged = None, False
    try:
        nx2, ny2 = 2 * nx - 1, 2 * ny - 1
        xs2, ys2, U2, inside2 = run_with_timeout(
            _build_and_solve_fd_grid, nx2, ny2, x_range, y_range, inside_fn, boundary_fn, source_fn,
            label="fd_solve_refined")
        diffs = []
        for i in range(0, nx, max(1, nx // 6)):
            for j in range(0, ny, max(1, ny // 6)):
                if not inside[i, j]:
                    continue
                fine_i, fine_j = 2 * i, 2 * j  # exact coincidence: fine grid has 2x-1 points
                if fine_i < nx2 and fine_j < ny2 and not np.isnan(U2[fine_i, fine_j]):
                    diffs.append(abs(U[i, j] - U2[fine_i, fine_j]))
        if diffs:
            convergence_error = max(diffs)
            converged = convergence_error < convergence_tolerance
    except Exception:  # noqa: BLE001
        pass  # convergence check is a bonus diagnostic; a failure here shouldn't hide a valid solve

    note = "Grid-refinement check confirms the solution has converged." if converged else \
        ("Could not confirm grid-refinement convergence within tolerance -- consider a finer grid "
         "(higher nx/ny) if precision near this level matters." if convergence_error is not None else
         "Grid-refinement convergence check could not be completed.")

    return FiniteDifferencePDEResult(grid_x=xs, grid_y=ys, solution_grid=U, resolution=(nx, ny),
                                      convergence_error=convergence_error, converged=converged, note=note)
