"""
Represents a linear system of "equation"-kind relations explicitly as
A x = b, instead of only ever handing a scalar list to sp.solve() (which
is what solver.py's algebraic path still does for the actual answer --
this module is an additional, richer VIEW onto the same equations, not a
replacement solve path).

Why this earns its own module rather than living inline in solver.py:
  - the coefficient matrix itself is worth showing (structural
    representation, e.g. a Markov transition matrix or a stiffness
    matrix for coupled springs)
  - eigenvalues of a square coefficient matrix are meaningful on their
    own (stability of a linear system, natural frequencies of a coupled
    oscillator) independent of "solve for x"
  - sp.solve() on an inconsistent or underdetermined system just
    returns an empty list/dict -- it doesn't say *why*. Rank comparison
    (rank(A) vs rank([A|b])) distinguishes "no solution" (inconsistent)
    from "infinitely many solutions" (underdetermined) from "unique
    solution" (full rank), the same classification a linear-algebra
    course teaches, rather than an opaque "SymPy found nothing."

Only triggers for genuine systems: at least 2 linear equations over at
least 2 shared unknown symbols. A single equation in one unknown stays on
the existing scalar path -- representing that as a 1x1 matrix would add
ceremony without adding information.
"""
from dataclasses import dataclass
import sympy as sp

from modules.equation_engine import ProblemModel, Equation, target_kind


@dataclass
class MatrixSystemResult:
    symbols: list[str]
    A: sp.Matrix
    b: sp.Matrix
    is_square: bool
    determinant: sp.Expr | None          # only set when is_square
    eigenvalues: dict[sp.Expr, int] | None  # value -> algebraic multiplicity; only set when is_square
    rank_A: int
    rank_augmented: int
    consistent: bool
    unique: bool
    solution: dict[str, sp.Expr] | None
    classification: str                   # human-readable summary


def _is_linear(exprs: list[sp.Expr], symbols: list[sp.Symbol]) -> bool:
    """sp.linear_eq_to_matrix raises for anything genuinely nonlinear in
    the given symbols -- used as the linearity test itself rather than
    hand-rolling a degree check, since it's the same code path that then
    actually builds the matrix."""
    try:
        sp.linear_eq_to_matrix(exprs, symbols)
        return True
    except Exception:  # noqa: BLE001
        return False


def _is_sequentially_solvable(exprs: list[sp.Expr], symbols: list[str]) -> bool:
    """True if every unknown can be pinned down one at a time -- at each
    step, some remaining equation has exactly one still-unresolved symbol
    among `symbols`, so plain substitution/elimination (solve that one,
    plug it into the rest, repeat) fully resolves the system without
    ever needing two unknowns solved simultaneously. This is the actual
    test for whether the matrix/eigenvalue machinery earns its keep --
    NOT just "are there >=2 equations and >=2 unknowns somewhere in the
    model", since e.g. "a = (v_f-v_i)/t" and "d = 0.5*a*t^2 + t*v_i" both
    being present doesn't make solving for `a` a coupled problem: the
    first equation determines `a` completely on its own, and `d` never
    needed to enter the picture.

    False means genuinely coupled -- e.g. 2x+3y=8, x-y=1, where neither
    equation alone isolates x or y -- which is exactly when showing A x=b
    (and eigenvalues, rank-based classification, etc.) adds real
    information over "just substitute". Also correctly returns False
    (i.e. "show the matrix view") for underdetermined/inconsistent
    systems, since nothing there gets fully pinned down by substitution
    either, and the rank-based classification is precisely what explains why."""
    remaining_eqs = list(exprs)
    remaining_syms = set(symbols)
    progressed = True
    while remaining_eqs and remaining_syms and progressed:
        progressed = False
        for i, ex in enumerate(remaining_eqs):
            present = {s.name for s in ex.free_symbols} & remaining_syms
            if len(present) == 1:
                remaining_syms -= present
                remaining_eqs.pop(i)
                progressed = True
                break
    return not remaining_syms


def build_linear_system(equations: list[Equation], unknown_symbols: list[str],
                         knowns: dict | None = None, force: bool = False) -> tuple[sp.Matrix, sp.Matrix, list[str]] | None:
    """Builds A, b for A x = b from the given "equation"-kind relations,
    restricted to whichever of unknown_symbols actually appear in them.
    `knowns` (a {Symbol: value} dict, e.g. from verifier._known_substitutions)
    is substituted first so already-known variables don't show up as
    spurious matrix columns. Returns None if the result wouldn't be a
    genuine system (fewer than 2 equations, fewer than 2 shared unknowns,
    or the relations aren't linear in those unknowns).

    By default ALSO returns None when the system is sequentially
    solvable by plain substitution (see _is_sequentially_solvable) --
    the matrix view doesn't earn its keep there. Pass force=True to skip
    that check and get the matrix representation anyway, which is what
    solver.py's "alternate method" (Cramer's rule) uses: showing a
    second way to solve something, on request, is a different situation
    than deciding what the DEFAULT view should be."""
    knowns = knowns or {}
    eqs = [e for e in equations if e.kind == "equation" and e.sympy_eq is not None]
    if len(eqs) < 2:
        return None

    exprs = [sp.expand((e.sympy_eq.lhs - e.sympy_eq.rhs).subs(knowns)) for e in eqs]

    present = set()
    for ex in exprs:
        present |= {s.name for s in ex.free_symbols}
    syms = [s for s in unknown_symbols if s in present]
    if len(syms) < 2:
        return None

    sym_objs = [sp.Symbol(s) for s in syms]
    if not _is_linear(exprs, sym_objs):
        return None

    if not force and _is_sequentially_solvable(exprs, syms):
        return None

    try:
        A, b = sp.linear_eq_to_matrix(exprs, sym_objs)
    except Exception:  # noqa: BLE001
        return None
    return A, b, syms


def analyze_linear_system(A: sp.Matrix, b: sp.Matrix, symbols: list[str]) -> MatrixSystemResult:
    """Rank-based classification (unique / infinite / inconsistent),
    determinant + eigenvalues when square, and the actual solution (a
    parametric family's free variables are simply omitted from the
    returned dict, since they have no single value)."""
    augmented = A.row_join(b)
    rank_A = A.rank()
    rank_aug = augmented.rank()
    n = len(symbols)
    is_square = A.rows == A.cols

    consistent = rank_A == rank_aug
    unique = consistent and rank_A == n

    determinant = A.det() if is_square else None
    eigenvalues = A.eigenvals() if is_square else None

    solution: dict[str, sp.Expr] | None = None
    if consistent:
        try:
            sol_set = sp.linsolve((A, b), sp.symbols(symbols))
            if sol_set:
                sol_tuple = next(iter(sol_set))
                if unique:
                    solution = {name: sp.simplify(val) for name, val in zip(symbols, sol_tuple)}
                else:
                    # keep only the components that came out as pure numbers/
                    # constants (no leftover free symbol) -- the rest are the
                    # free parameters of the infinite family and have no
                    # single value to report
                    solution = {name: sp.simplify(val) for name, val in zip(symbols, sol_tuple)
                                if not val.free_symbols}
        except Exception:  # noqa: BLE001
            solution = None

    if not consistent:
        classification = (f"No solution -- the system is inconsistent (rank(A)={rank_A} < "
                           f"rank([A|b])={rank_aug}).")
    elif unique:
        classification = f"Unique solution (full rank, {rank_A}={n})."
    else:
        classification = (f"Infinitely many solutions -- underdetermined (rank(A)={rank_A} < "
                           f"{n} unknowns).")
    if is_square and determinant == 0 and consistent and not unique:
        classification += " The coefficient matrix is singular (det(A) = 0)."

    return MatrixSystemResult(
        symbols=symbols, A=A, b=b, is_square=is_square,
        determinant=determinant, eigenvalues=eigenvalues,
        rank_A=rank_A, rank_augmented=rank_aug,
        consistent=consistent, unique=unique,
        solution=solution, classification=classification,
    )


def linear_system_view(model: ProblemModel, knowns: dict | None = None, force: bool = False) -> MatrixSystemResult | None:
    """Top-level entry point: given a whole ProblemModel, find the
    equation-kind unknowns actually being solved for (or, absent an
    explicit solve_for restricted to equation-kind targets, every unknown
    equation-kind symbol) and build+analyze the matrix system. Returns
    None if the model doesn't contain a genuine (>=2 eq, >=2 unknown)
    linear system. `force` passes through to build_linear_system -- see
    its docstring."""
    unknown_symbols = [v.symbol for v in model.variables
                        if v.known_value is None and not v.is_function and not v.is_vector]
    # also fold in solve_for targets that are algebraic but weren't
    # declared as a Variable (defensive -- normally they are)
    for t in model.solve_for:
        if target_kind(model, t) == "equation" and t not in unknown_symbols:
            unknown_symbols.append(t)
    if len(unknown_symbols) < 2:
        return None

    built = build_linear_system(model.equations, unknown_symbols, knowns, force=force)
    if built is None:
        return None
    A, b, syms = built
    return analyze_linear_system(A, b, syms)
