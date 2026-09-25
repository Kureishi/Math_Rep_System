"""
Shared recurrence (difference-equation) solving helper, used by both
solver.py and verifier.py. Structured identically to ode_utils.py -- the
discrete-time sibling of that module -- and lives in its own file for the
same reason: solver.py already imports from verifier.py, so a module both
of them need has to sit outside either one to avoid a circular import.
"""
import sympy as sp
from sympy.core.function import AppliedUndef

from modules.equation_engine import ProblemModel
from modules.timeout_utils import run_with_timeout


def solve_recurrence(model: ProblemModel) -> dict[str, sp.Expr]:
    """rsolve()s every recurrence-kind equation, applying any initial
    conditions that match its function. Returns {function_name: closed_form_expr}
    (NOT wrapped in Eq -- rsolve returns the closed-form expression directly).
    Only standalone (uncoupled) recurrences are supported; a system of
    mutually-referencing recurrences (e.g. two interacting accounts) isn't
    solved here -- sympy has no rsolve_system equivalent to dsolve_system."""
    result: dict[str, sp.Expr] = {}
    for e in model.equations:
        if e.kind != "recurrence" or e.sympy_eq is None:
            continue
        funcs = e.sympy_eq.atoms(AppliedUndef)
        if not funcs:
            continue
        # the function applied at its "base" argument (e.g. a(n), not a(n+1))
        # -- rsolve wants the function itself, not a shifted application
        func_name = str(next(iter(funcs)).func)
        indep_var = _independent_variable(funcs)
        if indep_var is None:
            continue
        func_base = sp.Function(func_name)(indep_var)

        ics = {}
        for ic in model.initial_conditions:
            if ic.sympy_eq is None:
                continue
            lhs_funcs = ic.sympy_eq.lhs.atoms(AppliedUndef)
            if lhs_funcs and str(next(iter(lhs_funcs)).func) == func_name:
                ics[ic.sympy_eq.lhs] = ic.sympy_eq.rhs

        try:
            sol = run_with_timeout(sp.rsolve, e.sympy_eq, func_base, ics or None, label="rsolve")
            if sol is not None:
                result[func_name] = sol
        except Exception:  # noqa: BLE001
            continue
    return result


def _independent_variable(applied_funcs) -> sp.Symbol | None:
    """Recovers the bare independent variable (e.g. 'n') from a set of
    applied functions like {a(n+1), a(n)} -- their arguments are
    expressions in that variable (n+1, n), so we take the free symbols of
    whichever argument is simplest (a bare symbol) if one exists, else the
    free symbols of any argument."""
    for f in applied_funcs:
        arg = f.args[0]
        if arg.is_Symbol:
            return arg
    for f in applied_funcs:
        free = f.args[0].free_symbols
        if free:
            return next(iter(free))
    return None


def extract_step_map(model: ProblemModel, func_name: str) -> tuple[sp.Expr, sp.Symbol] | None:
    """For a FIRST-ORDER recurrence a(n+1) = g(a(n)), returns (g(x), x) --
    the one-step map a cobweb diagram (build_cobweb_plot in plotter.py)
    needs, with x a fresh generic symbol standing in for "the current
    value" so g can be lambdified and evaluated at any point, not just
    along the actual solved sequence. Returns None when there's no
    recurrence-kind equation defining func_name, when the equation isn't
    genuinely first-order (anything beyond exactly {a(n), a(n+1)} -- e.g.
    a(n+2) = a(n+1) + a(n) has no single-variable map to draw), when a
    shift isn't a plain integer offset from n, or when solving for a(n+1)
    fails or leaves the map depending on n itself (a(n+1) = g(a(n), n) has
    no fixed curve to plot either -- a cobweb diagram needs g to be a
    function of the PREVIOUS VALUE alone, not of the step index too)."""
    for e in model.equations:
        if e.kind != "recurrence" or e.sympy_eq is None:
            continue
        matching = [f for f in e.sympy_eq.atoms(AppliedUndef) if str(f.func) == func_name]
        if not matching:
            continue
        indep_var = _independent_variable(matching)
        if indep_var is None:
            return None

        shift_of = {}
        for f in matching:
            offset = sp.simplify(f.args[0] - indep_var)
            if not offset.is_number:
                return None
            shift_of[offset] = f  # keep the ACTUAL atom from the equation for each shift --
            # equation_engine.py parses everything with evaluate=False, so a freshly
            # reconstructed sp.Function(func_name)(indep_var + 1) is a structurally
            # DIFFERENT (if mathematically equal) Add node than the one actually
            # embedded in e.sympy_eq (different internal .args order, different
            # hash) -- sp.solve()/subs() would silently fail to match it. Using the
            # real atom sidesteps that entirely, the same way verify_recurrence_solution
            # above does (it substitutes via f.args[0] straight from eq_sympy.atoms(),
            # never reconstructing a shift argument by hand).
        if set(shift_of) != {sp.Integer(0), sp.Integer(1)}:
            return None  # not first-order in exactly {a(n), a(n+1)}

        next_app, cur_app = shift_of[sp.Integer(1)], shift_of[sp.Integer(0)]
        try:
            solutions = sp.solve(e.sympy_eq, next_app)
        except Exception:  # noqa: BLE001
            return None
        if not solutions:
            return None

        placeholder = sp.Symbol(f"{func_name}_x")
        g = solutions[0].subs(cur_app, placeholder)
        if g.has(next_app) or g.has(cur_app) or indep_var in g.free_symbols:
            return None
        return g, placeholder
    return None


def verify_recurrence_solution(eq_sympy: sp.Eq, func_name: str, closed_form: sp.Expr,
                                 indep_var: sp.Symbol, tolerance: float = 1e-6,
                                 sample_points: tuple = (0, 1, 2, 3, 5, 8)) -> tuple[bool, sp.Expr]:
    """Verifies a recurrence solution by substituting the closed-form
    expression for every shifted application of the function (a(n),
    a(n+1), a(n+2), ...) found in the original equation, then checking the
    resulting identity holds -- analogous to checkodesol, but there's no
    direct sympy equivalent for recurrences, so this substitute-and-check
    is done by hand.

    Tries an exact symbolic check first; falls back to numeric sampling at
    a handful of integer points if that doesn't land exactly on zero. The
    fallback matters in practice: a closed form that mixes irrational
    numbers (e.g. sqrt(5) in a Fibonacci-style solution) with float-valued
    initial conditions accumulates floating-point noise on the order of
    1e-17, which is a rounding artifact, not a real inconsistency -- exact
    symbolic zero is simply not achievable in that case even though the
    solution is correct."""
    applied = eq_sympy.atoms(AppliedUndef)
    subs_map = {}
    for f in applied:
        if str(f.func) != func_name:
            continue
        shift_arg = f.args[0]  # e.g. n+1, n+2, or just n
        subs_map[f] = closed_form.subs(indep_var, shift_arg)

    lhs = eq_sympy.lhs.subs(subs_map)
    rhs = eq_sympy.rhs.subs(subs_map)
    try:
        residual = sp.simplify(lhs - rhs)
    except Exception:  # noqa: BLE001
        residual = lhs - rhs

    if residual == 0:
        return True, residual

    # numeric fallback: sample the (symbolic-in-n) residual at several
    # integer points and check it's negligibly small at all of them
    try:
        max_abs = 0.0
        for pt in sample_points:
            val = complex(residual.subs(indep_var, pt))
            max_abs = max(max_abs, abs(val))
        return (max_abs < tolerance), residual
    except (TypeError, ValueError):
        return False, residual
