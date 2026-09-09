"""Tests for modules/pde_utils.py -- first-order PDEs via pdsolve, and
the heat/wave equations on a finite interval via Fourier sine series."""
import sympy as sp

from modules.pde_utils import solve_first_order_pde, solve_heat_equation_dirichlet, solve_wave_equation_dirichlet


# ---------------------------------------------------------------- first-order (general)
def test_first_order_linear_pde_solved_and_verified():
    result = solve_first_order_pde("Eq(Derivative(u(x,y), x) + Derivative(u(x,y), y), 0)")
    assert result.error is None
    assert result.solution is not None
    assert result.verified


def test_quasilinear_first_order_pde_solved_and_verified():
    result = solve_first_order_pde("Eq(Derivative(u(x,y), x) + Derivative(u(x,y), y), 2*u(x,y))")
    assert result.error is None
    assert result.verified


def test_second_order_pde_rejected_with_helpful_message():
    """pdsolve genuinely has no support for the heat equation -- this
    must fail cleanly with a message pointing at the right function,
    not a raw traceback or a wrong silent answer."""
    result = solve_first_order_pde("Eq(Derivative(u(x,y), y), Derivative(u(x,y), x, 2))")
    assert result.error is not None
    assert "solve_heat_equation_dirichlet" in result.error


def test_first_order_pde_parse_error():
    result = solve_first_order_pde("not @@ valid")
    assert result.error is not None
    assert result.input_pde is None


# ---------------------------------------------------------------- heat equation
def test_heat_equation_pure_mode_initial_condition_exact():
    """A pure sin(pi*x/L) initial condition is a single eigenmode --
    the solution should be exactly that mode decaying in time, with a
    near-zero initial-condition fit error even at the smallest mode
    count tried."""
    result = solve_heat_equation_dirichlet("sin(pi*x)", length=1.0, alpha=1.0)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.ic_fit_ok
    assert result.ic_fit_error < 1e-9
    assert result.modes_used == 6  # smallest schedule entry, since it already fits


def test_heat_equation_smooth_initial_condition_converges():
    result = solve_heat_equation_dirichlet("x*(1-x)", length=1.0, alpha=0.5)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.ic_fit_ok
    assert result.ic_fit_error < 1e-3


def test_heat_equation_boundary_conditions_satisfied():
    """u(0,t) = u(L,t) = 0 should hold for every mode by construction
    (sin(n*pi*0/L) = sin(n*pi) = 0), regardless of the initial condition."""
    result = solve_heat_equation_dirichlet("x*(1-x)", length=1.0, alpha=1.0)
    x = sp.Symbol("x", positive=True)
    assert result.solution.subs(x, 0) == 0
    assert sp.simplify(result.solution.subs(x, 1)) == 0


def test_heat_equation_discontinuous_initial_condition_does_not_converge():
    """A step-function initial condition can never be fit to within a
    tight tolerance at any finite mode count (Gibbs phenomenon) -- this
    must be reported honestly as not-fully-converged, not silently
    capped and presented as a good fit."""
    result = solve_heat_equation_dirichlet(
        "Piecewise((1, x < 0.5), (0, True))", length=1.0, alpha=1.0, ic_tolerance=1e-3)
    assert result.error is None
    assert result.pde_residual_zero  # still an exact PDE solution
    assert not result.ic_fit_ok       # but the IC itself never converges
    assert result.modes_used == 60    # ran through the full schedule


def test_heat_equation_parse_error():
    result = solve_heat_equation_dirichlet("not @@ valid")
    assert result.error is not None


# ---------------------------------------------------------------- wave equation
def test_wave_equation_pure_mode_zero_velocity():
    result = solve_wave_equation_dirichlet("sin(pi*x)", "0", length=1.0, wave_speed=1.0)
    t = sp.Symbol("t", positive=True)
    x = sp.Symbol("x", positive=True)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.ic_fit_ok
    assert result.solution.equals(sp.sin(sp.pi * x) * sp.cos(sp.pi * t))


def test_wave_equation_zero_displacement_nonzero_velocity():
    result = solve_wave_equation_dirichlet("0", "sin(pi*x)", length=1.0, wave_speed=2.0)
    assert result.error is None
    assert result.pde_residual_zero
    # initial displacement should be exactly zero everywhere
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    assert result.solution.subs(t, 0) == 0


def test_wave_equation_boundary_conditions_satisfied():
    result = solve_wave_equation_dirichlet("x*(1-x)", "0", length=1.0, wave_speed=1.0)
    x = sp.Symbol("x", positive=True)
    assert result.solution.subs(x, 0) == 0
    assert sp.simplify(result.solution.subs(x, 1)) == 0


def test_wave_equation_parse_error():
    result = solve_wave_equation_dirichlet("not @@ valid", "0")
    assert result.error is not None
