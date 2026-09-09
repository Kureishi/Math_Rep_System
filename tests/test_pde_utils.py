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


# ---------------------------------------------------------------- Neumann BCs
def test_heat_equation_neumann_long_time_limit_is_average_temperature():
    """An insulated rod (no heat escapes at either end) must equilibrate
    to the average of its initial temperature distribution, not decay
    to zero -- this is the key physical difference from the Dirichlet
    case and a genuine correctness check, not just a residual check."""
    from modules.pde_utils import solve_heat_equation_neumann
    result = solve_heat_equation_neumann("x*(1-x)", length=1.0, alpha=0.5)
    x = sp.Symbol("x", positive=True)
    t = sp.Symbol("t", positive=True)
    expected_average = sp.integrate(x * (1 - x), (x, 0, 1))
    assert result.error is None
    assert result.pde_residual_zero
    late_time_value = float(result.solution.subs({t: 1000, x: 0.3}))
    assert abs(late_time_value - float(expected_average)) < 1e-6


def test_wave_equation_neumann_rigid_translation_from_uniform_velocity():
    """A free string given a uniform initial velocity with no restoring
    force at the k=0 mode must translate rigidly at that velocity (u =
    v0 * t), not oscillate -- a real physical prediction, not just an
    identity check."""
    from modules.pde_utils import solve_wave_equation_neumann
    result = solve_wave_equation_neumann("0", "2", length=1.0, wave_speed=1.0)
    t = sp.Symbol("t", positive=True)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.solution.subs(t, 3) == 6


def test_heat_equation_neumann_parse_error():
    from modules.pde_utils import solve_heat_equation_neumann
    result = solve_heat_equation_neumann("not @@ valid")
    assert result.error is not None


# ---------------------------------------------------------------- Robin BCs
def test_heat_equation_robin_converges_and_satisfies_pde():
    from modules.pde_utils import solve_heat_equation_robin
    result = solve_heat_equation_robin("x*(1-x)", length=1.0, alpha=1.0, robin_coefficient=2.0)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.ic_fit_ok
    assert result.ic_fit_error < 1e-3


def test_heat_equation_robin_eigenvalues_are_not_multiples_of_pi():
    """Distinguishing feature of Robin BCs versus Dirichlet/Neumann:
    the eigenvalues are roots of a transcendental equation, NOT evenly
    spaced multiples of pi/L -- this pins down that the numeric
    eigenvalue solver is actually being used, not accidentally falling
    back to a Dirichlet-style closed form."""
    from modules.pde_utils import _robin_eigenvalues
    eigenvalues = _robin_eigenvalues(L=1.0, h=2.0, n_modes=3)
    import math
    for lam in eigenvalues:
        assert abs(lam % math.pi) > 1e-3  # not (approximately) a multiple of pi


def test_heat_equation_robin_parse_error():
    from modules.pde_utils import solve_heat_equation_robin
    result = solve_heat_equation_robin("not @@ valid")
    assert result.error is not None


# ---------------------------------------------------------------- Laplace's equation
def test_laplace_rectangle_single_side_boundary_recovered():
    from modules.pde_utils import solve_laplace_rectangle
    result = solve_laplace_rectangle(bottom="x*(1-x)", width=1.0, height=1.0)
    x = sp.Symbol("x", positive=True)
    y = sp.Symbol("y", positive=True)
    assert result.error is None
    assert result.pde_residual_zero
    assert result.boundary_fit_ok
    # the three zero sides must be (numerically) exactly zero everywhere on them
    assert result.solution.subs(y, 1) == 0
    assert result.solution.subs(x, 0) == 0
    assert result.solution.subs(x, 1) == 0


def test_laplace_rectangle_all_zero_boundaries_gives_zero_solution():
    from modules.pde_utils import solve_laplace_rectangle
    result = solve_laplace_rectangle()
    assert result.error is None
    assert result.solution == 0
    assert result.boundary_fit_ok


def test_laplace_rectangle_parse_error():
    from modules.pde_utils import solve_laplace_rectangle
    result = solve_laplace_rectangle(bottom="not @@ valid")
    assert result.error is not None


# ---------------------------------------------------------------- finite-difference fallback
def test_finite_difference_matches_known_exact_poisson_solution_on_disk():
    """u_xx + u_yy = -4 on the unit disk with u = 0 on the boundary has
    the known exact solution u = 1 - x**2 - y**2 -- a real independent
    check the finite-difference solver's answer can be measured against,
    not just an internal residual check."""
    import numpy as np
    from modules.pde_utils import solve_pde_finite_difference_2d
    result = solve_pde_finite_difference_2d(
        source_str="-4", boundary_value_str="0", domain_predicate_str="x**2+y**2<=1",
        x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), nx=41, ny=41)
    assert result.error is None
    i = np.argmin(np.abs(result.grid_x - 0.0))
    j = np.argmin(np.abs(result.grid_y - 0.0))
    center_value = result.solution_grid[i, j]
    assert abs(center_value - 1.0) < 0.1  # first-order boundary accuracy at this resolution


def test_finite_difference_accuracy_improves_with_resolution():
    """A real convergence check: the error against the known exact
    solution above should shrink as the grid is refined, confirming
    this isn't just coincidentally close at one resolution."""
    import numpy as np
    from modules.pde_utils import solve_pde_finite_difference_2d
    errors = []
    for n in (21, 81):
        result = solve_pde_finite_difference_2d(
            source_str="-4", boundary_value_str="0", domain_predicate_str="x**2+y**2<=1",
            x_range=(-1.2, 1.2), y_range=(-1.2, 1.2), nx=n, ny=n)
        i = np.argmin(np.abs(result.grid_x - 0.0))
        j = np.argmin(np.abs(result.grid_y - 0.0))
        errors.append(abs(result.solution_grid[i, j] - 1.0))
    assert errors[1] < errors[0]


def test_finite_difference_empty_domain_reports_error_not_crash():
    from modules.pde_utils import solve_pde_finite_difference_2d
    result = solve_pde_finite_difference_2d(domain_predicate_str="x>100")
    assert result.error is not None


def test_finite_difference_parse_error():
    from modules.pde_utils import solve_pde_finite_difference_2d
    result = solve_pde_finite_difference_2d(source_str="not @@ valid")
    assert result.error is not None
