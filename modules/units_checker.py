"""
Dimensional-consistency checking via sympy.physics.units.

This catches a class of error the numeric-balance check in verifier.py
cannot: numeric substitution only ever sees plain numbers, so a derivation
that (say) equates a distance to a velocity would sail through as long as
the numbers happened to match. This module substitutes each symbol's *unit*
(not its value) and checks that both sides of every equation -- and every
additive term within a side -- share the same physical dimension.
"""
import itertools

import sympy as sp
from sympy.physics import units as u
from sympy.physics.units import Quantity
from sympy.physics.units.systems.si import SI
from sympy.parsing.sympy_parser import parse_expr, standard_transformations, convert_xor

# Deliberately NOT using implicit_multiplication_application here: that
# transform silently decomposes an unrecognized multi-character token (e.g.
# a typo'd or made-up unit like "sprockets") into a product of any
# single-letter symbols it happens to contain that ARE in local_dict (e.g.
# the 's' in "sprockets" resolving to "second"), producing a bogus but
# valid-looking dimension instead of failing loudly. Whitespace between
# unit tokens (e.g. "kg m/s^2") is normalized to explicit '*' before
# parsing instead, so multi-token unit strings still work correctly.
TRANSFORMS = standard_transformations + (convert_xor,)

# Common unit tokens a model is likely to produce, mapped to sympy.physics.units
# quantities. Currency/percent/angle are treated as dimensionless ("count") on
# purpose -- they're not SI physical dimensions, but shouldn't block checking
# of an equation that mixes them with genuinely dimensional quantities.
UNIT_ALIASES: dict[str, sp.Expr] = {
    "m": u.meter, "meter": u.meter, "meters": u.meter, "metre": u.meter, "metres": u.meter,
    "km": u.kilometer, "kilometer": u.kilometer, "kilometers": u.kilometer,
    "cm": u.centimeter, "mm": u.millimeter,
    "ft": u.feet, "feet": u.feet, "foot": u.feet, "in": u.inch, "inch": u.inch,
    "mi": u.mile, "mile": u.mile, "miles": u.mile, "mph": u.mile/u.hour,
    "s": u.second, "sec": u.second, "second": u.second, "seconds": u.second,
    "min": u.minute, "minute": u.minute, "minutes": u.minute,
    "hr": u.hour, "h": u.hour, "hour": u.hour, "hours": u.hour,
    "yr": u.year, "year": u.year, "years": u.year, "day": u.day, "days": u.day,
    "kg": u.kilogram, "kilogram": u.kilogram, "g": u.gram, "gram": u.gram,
    "lb": u.pound, "lbs": u.pound,
    "N": u.newton, "newton": u.newton, "newtons": u.newton,
    "J": u.joule, "joule": u.joule, "joules": u.joule,
    "W": u.watt, "watt": u.watt, "watts": u.watt,
    "Pa": u.pascal, "pascal": u.pascal,
    "Hz": u.hertz, "hertz": u.hertz,
    "A": u.ampere, "V": u.volt, "volt": u.volt, "volts": u.volt,
    "ohm": u.ohm, "C": u.coulomb, "coulomb": u.coulomb,
    "K": u.kelvin, "degK": u.kelvin,
    "mol": u.mole, "mole": u.mole,
    "rad": u.radian, "radian": u.radian, "deg": u.degree, "degree": u.degree, "degrees": u.degree,
    "L": u.liter, "liter": u.liter, "liters": u.liter, "litre": u.liter,
    # not true SI dimensions, but common in word problems -- treated as
    # dimensionless "count" so they don't block checking a mixed equation
    "$": sp.Integer(1), "USD": sp.Integer(1), "dollars": sp.Integer(1), "dollar": sp.Integer(1),
    "%": sp.Integer(1), "percent": sp.Integer(1),
    "unitless": sp.Integer(1), "dimensionless": sp.Integer(1), "count": sp.Integer(1),
    "people": sp.Integer(1), "items": sp.Integer(1), "units": sp.Integer(1),
}


class UnitParseError(Exception):
    pass


def parse_unit(unit_str: str | None) -> sp.Expr:
    """Parses a unit string like 'm/s^2' or 'kg m/s^2' into a sympy units
    expression. Raises UnitParseError if any token can't be resolved to a
    known unit -- silently guessing would be worse than not checking."""
    if unit_str is None or not unit_str.strip():
        return sp.Integer(1)
    normalized = " ".join(unit_str.split())  # collapse whitespace
    normalized = normalized.replace(" ", "*").replace("·", "*").replace("per", "/")
    try:
        expr = parse_expr(normalized, local_dict=UNIT_ALIASES, transformations=TRANSFORMS)
    except Exception as e:  # noqa: BLE001
        raise UnitParseError(f"couldn't parse unit '{unit_str}': {e}") from e
    if expr.free_symbols:
        # Quantity objects (u.meter etc.) are atomic and don't appear here --
        # only genuinely unresolved bare identifiers do.
        unresolved = ", ".join(sorted(s.name for s in expr.free_symbols))
        raise UnitParseError(f"unrecognized unit token(s) in '{unit_str}': {unresolved}")
    return expr


def dimension_of(expr: sp.Expr):
    """SI dimension of a units-expression. Raises ValueError (from SymPy)
    if the expression itself is dimensionally invalid, e.g. adding a length
    to a time inside one side of an equation."""
    _, dim = SI._collect_factor_and_dimension(expr)
    return dim


def dims_equivalent(dim_a, dim_b) -> bool:
    return SI.get_dimension_system().equivalent_dims(dim_a, dim_b)


_placeholder_counter = itertools.count()


def make_dimension_placeholder(dim) -> sp.Expr:
    """A fresh, uniquely-named Quantity carrying exactly the given
    dimension (scale factor 1). Used instead of substituting the same
    canonical unit object (e.g. the module-level u.meter) for two
    DIFFERENT symbols that happen to share a unit string.

    Why this matters: if two distinct symbols 'a' and 'b' both have unit
    "m" and both get substituted with the literal same u.meter object,
    SymPy sees them as interchangeable -- so checking the dimension of
    "a - b" silently collapses to "meter - meter = 0" (a bare, dimension-
    less zero) before the dimension is ever computed, hiding a check that
    should have correctly reported "length" for that difference. Giving
    each distinct symbol its own placeholder object (same dimension, but
    not the same object) prevents that false cancellation while still
    correctly validating that same-dimension terms combine and
    different-dimension terms don't."""
    q = Quantity(f"_dim_ph_{next(_placeholder_counter)}")
    SI.set_quantity_dimension(q, dim)
    SI.set_quantity_scale_factor(q, sp.Integer(1))
    return q
