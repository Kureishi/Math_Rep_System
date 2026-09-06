"""
Dimensional-analysis-only mode: given just the DIMENSIONS of a set of
input quantities and a desired output dimension -- no numbers, no
explicit formula -- proposes candidate exponent combinations
(y = C * a^p1 * b^p2 * ...) whose dimensions actually work out. The
Buckingham-Pi-style "what combination of these could this possibly be"
exploration physics students are sometimes asked to do directly, before
any equation is even proposed. A genuinely different kind of problem
from anything else in this app: nothing here is about solving a stated
equation, it's about DISCOVERING which combinations of units could
plausibly combine into a desired unit at all.

Reuses units_checker.py's existing SI unit-parsing (the same machinery
verifier.py's own dimensional-consistency pass depends on) rather than
reimplementing it -- this module only adds the EXPONENT-SOLVING layer
on top, via plain linear algebra: each unit maps to a vector of
exponents over the 7 SI base dimensions (length, mass, time, ...), and
finding a dimensionally-valid combination of inputs for a target
reduces to solving a linear system for the input exponents.

When more inputs are given than there are independent dimensions
involved, the system is UNDERDETERMINED -- there's a whole FAMILY of
dimensionally-valid exponent combinations, not just one, differing by
any dimensionless group built from the inputs (the literal definition
of a Buckingham-Pi group). Rather than only reporting one particular
solution, a bounded search over small rational exponents surfaces
several concrete, readable candidates from that family.
"""
import itertools
from dataclasses import dataclass, field
from fractions import Fraction

import sympy as sp
from sympy.physics.units.systems.si import SI

from modules.units_checker import parse_unit, UnitParseError

_BASE_DIMENSIONS = ["length", "mass", "time", "current", "temperature",
                     "amount_of_substance", "luminous_intensity"]

MAX_SEARCH_VARIABLES = 6   # a bounded grid search is only run up to this many input
                             # quantities -- beyond that the grid itself (search points
                             # per variable, raised to this many variables) gets too large
MAX_CANDIDATES = 20         # stop collecting concrete candidates once this many are found


def _dimension_vector(unit_str: str) -> tuple:
    """Maps a unit string to a dense tuple of exponents over the 7 SI
    base dimensions, in the fixed order _BASE_DIMENSIONS -- e.g. 'm/s'
    -> (1, 0, -1, 0, 0, 0, 0). Dense/fixed-order (rather than a dict
    that omits zero entries) so two vectors can be compared and
    combined with plain tuple/arithmetic operations, no dict-merging
    edge cases. Raises UnitParseError (from units_checker.parse_unit)
    for a token this app doesn't recognize as a unit at all."""
    expr = parse_unit(unit_str)
    _, dim = SI._collect_factor_and_dimension(expr)
    deps = SI.get_dimension_system().get_dimensional_dependencies(dim)
    by_name = {str(k.name): sp.Rational(v) for k, v in deps.items()}
    return tuple(by_name.get(name, sp.Rational(0)) for name in _BASE_DIMENSIONS)


@dataclass
class CandidateFormula:
    exponents: dict
    description: str


@dataclass
class DimensionalAnalysisResult:
    feasible: bool
    degrees_of_freedom: int = 0
    particular_solution: dict = None
    candidates: list = field(default_factory=list)
    message: str = ""


def _format_candidate(exponents: dict) -> str:
    parts = []
    for name, exp in exponents.items():
        if exp == 1:
            parts.append(name)
        elif exp == -1:
            parts.append(f"{name}^-1")
        else:
            parts.append(f"{name}^{exp}")
    return "C * " + " * ".join(parts) if parts else "C (dimensionless)"


def analyze_dimensions(input_units: dict, target_unit: str,
                         search_min=Fraction(-3), search_max=Fraction(3),
                         search_step=Fraction(1, 2)) -> DimensionalAnalysisResult:
    """`input_units` maps a candidate input variable's name -> its unit
    string (e.g. {"m": "kg", "v": "m/s"}); `target_unit` is the desired
    output's unit string. Returns whether ANY combination of the given
    inputs' dimensions can produce the target dimension at all, and if
    so, a particular solution plus a handful of concrete small-exponent
    candidates found by bounded search."""
    if not input_units:
        return DimensionalAnalysisResult(False, message="Need at least one input quantity.")

    try:
        input_vectors = {name: _dimension_vector(u) for name, u in input_units.items()}
        target_vector = _dimension_vector(target_unit)
    except UnitParseError as e:
        return DimensionalAnalysisResult(False, message=str(e))

    names = list(input_vectors.keys())
    D = sp.Matrix([[input_vectors[name][i] for name in names] for i in range(len(_BASE_DIMENSIONS))])
    t = sp.Matrix([target_vector[i] for i in range(len(_BASE_DIMENSIONS))])

    exponent_syms = sp.symbols(f"e0:{len(names)}")
    system_rows = list(D * sp.Matrix(exponent_syms) - t)
    solutions = sp.linsolve(system_rows, exponent_syms)

    if not solutions:
        return DimensionalAnalysisResult(
            False, message="No combination of these input dimensions can produce the target "
                            "dimension -- the units are fundamentally incompatible (e.g. no "
                            "power of a length-only input can ever produce a time).",
        )

    solution_tuple = next(iter(solutions))
    free_symbols = set()
    for expr in solution_tuple:
        free_symbols |= expr.free_symbols
    dof = len(free_symbols)

    zero_subs = {s: sp.Integer(0) for s in free_symbols}
    particular = {
        name: Fraction(str(sp.nsimplify(expr.subs(zero_subs))))
        for name, expr in zip(names, solution_tuple)
    }

    candidates = []
    if len(names) <= MAX_SEARCH_VARIABLES:
        n_steps = int((search_max - search_min) / search_step) + 1
        grid = [search_min + i * search_step for i in range(n_steps)]
        seen = set()
        for combo in itertools.product(grid, repeat=len(names)):
            total = [sp.Rational(0)] * len(_BASE_DIMENSIONS)
            for name, exp in zip(names, combo):
                if exp == 0:
                    continue
                vec = input_vectors[name]
                for i in range(len(_BASE_DIMENSIONS)):
                    total[i] += vec[i] * sp.Rational(exp.numerator, exp.denominator)
            if tuple(total) == target_vector:
                nonzero = {name: exp for name, exp in zip(names, combo) if exp != 0}
                key = tuple(sorted(nonzero.items()))
                if key not in seen and (nonzero or target_vector == tuple([sp.Rational(0)] * 7)):
                    seen.add(key)
                    candidates.append(CandidateFormula(nonzero, _format_candidate(nonzero)))
            if len(candidates) >= MAX_CANDIDATES:
                break

    message = (
        f"Feasible -- {dof} degree(s) of freedom beyond the exponents shown, meaning a whole "
        f"family of dimensionally-valid combinations exists (any dimensionless group built from "
        f"these inputs can be freely mixed in)."
        if dof else
        "Feasible -- these dimensions uniquely determine the exponents, up to an overall "
        "dimensionless constant C."
    )
    return DimensionalAnalysisResult(True, degrees_of_freedom=dof, particular_solution=particular,
                                       candidates=candidates, message=message)
