"""
Unit conversion sweep: once an answer's unit is known, offer it in a
handful of common alternate units (SI <-> imperial, and same-dimension
compound units like m/s <-> km/h <-> mph) instead of leaving the person
to do that conversion by hand.

Built entirely on units_checker.py's existing unit-parsing/dimension
infrastructure (parse_unit, dimension_of, dims_equivalent) rather than
introducing a second unit-string format -- if a unit string is valid
for the dimensional-consistency checker, it's valid here too, and vice
versa.

Deliberately excludes temperature scale conversions (Celsius <-> 
Fahrenheit <-> Kelvin): those are AFFINE (value * scale + offset), not
pure multiplicative scale factors, and sympy.physics.units.convert_to()
only handles the multiplicative case -- silently applying it to
Celsius/Fahrenheit would produce a wrong number that looks plausible.
Kelvin-only display is left as-is rather than faked with a wrong
formula; a correct Celsius/Fahrenheit conversion would need its own
explicit (non-sympy) affine-conversion path, which is a reasonable
future addition but isn't safe to bolt on as "just another unit string."
"""
import sympy as sp
from sympy.physics.units import convert_to

from modules.units_checker import parse_unit, dimension_of, dims_equivalent, UnitParseError

# Each profile is (a representative unit string for the dimension, the
# alternate unit strings worth offering for that dimension). Matched at
# runtime by dimension (dims_equivalent), not by string, so e.g. "meter"
# and "m" both hit the length profile.
_ALTERNATE_PROFILES: list[tuple[str, list[str]]] = [
    ("m", ["m", "km", "cm", "mm", "ft", "in", "mi"]),
    ("s", ["s", "min", "hr", "day"]),
    ("kg", ["kg", "g", "lb"]),
    ("m/s", ["m/s", "km/hr", "mph", "ft/s"]),
    ("N", ["N", "kg*m/s^2"]),
    ("J", ["J", "N*m", "W*s"]),
    ("W", ["W", "J/s"]),
    ("Pa", ["Pa", "N/m^2"]),
    ("L", ["L", "mL"]),
]


def convert_value(value: float, from_unit: str, to_unit: str) -> float | None:
    """Converts `value` from `from_unit` to `to_unit`. Returns None if
    either unit string can't be parsed, or the two units don't share a
    dimension (converting meters to kilograms, say) -- never raises, so
    callers sweeping several candidate units don't need a try/except
    around every single one."""
    try:
        from_expr = parse_unit(from_unit)
        to_expr = parse_unit(to_unit)
    except UnitParseError:
        return None
    if from_expr == sp.Integer(1) or to_expr == sp.Integer(1):
        return None  # dimensionless (or unparsed-as-dimensionless) -- nothing to convert
    if not dims_equivalent(dimension_of(from_expr), dimension_of(to_expr)):
        return None
    try:
        factor = sp.simplify(convert_to(value * from_expr, to_expr) / to_expr)
        return float(factor)
    except Exception:  # noqa: BLE001
        return None


def sweep_conversions(value: float, unit: str | None) -> list[tuple[str, float]]:
    """Returns [(alternate_unit, converted_value), ...] for whichever
    _ALTERNATE_PROFILES entry matches `unit`'s dimension, excluding the
    unit the value is already expressed in. Returns [] if `unit` is
    None/unparseable/dimensionless, or doesn't match any known profile
    (e.g. an exotic or made-up unit) -- there's simply nothing to offer
    in that case, not an error."""
    if not unit:
        return []
    try:
        unit_expr = parse_unit(unit)
    except UnitParseError:
        return []
    if unit_expr == sp.Integer(1):
        return []

    unit_dim = dimension_of(unit_expr)
    for representative, alternates in _ALTERNATE_PROFILES:
        try:
            rep_dim = dimension_of(parse_unit(representative))
        except UnitParseError:
            continue
        if dims_equivalent(unit_dim, rep_dim):
            results = []
            for alt in alternates:
                if alt.strip() == unit.strip():
                    continue
                converted = convert_value(value, unit, alt)
                if converted is not None:
                    results.append((alt, converted))
            return results
    return []
