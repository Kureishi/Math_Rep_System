"""
Physical plausibility sanity check: a soft advisory layer sitting above
domain_utils.py. domain_utils finds where a formula is mathematically
UNDEFINED (division by zero, a negative even-root argument, a log of a
non-positive number) -- those are genuine correctness failures. This
module instead flags values that are mathematically well-defined but
land far outside what's normal for the kind of quantity involved: a
car's acceleration coming out to 500 m/s^2, a computed mass that's
negative even though nothing declared a sign restriction on it, a
Kelvin temperature below absolute zero.

Deliberately NOT wired into VerificationReport.passed / report.add() the
way domain_utils is: an out-of-range magnitude doesn't mean the math is
wrong (a problem CAN legitimately be about a rocket sled or a
hypothetical extreme scenario), only that it's worth a second look.
Every note this module produces is advisory ("check this against
real-world expectations"), never a pass/fail verdict -- unlike
domain_utils, which genuinely can prove a formula is undefined.

Two independent layers:
  1. Category + unit magnitude ranges -- a small curated table, keyed by
     a domain category inferred from model.problem_domain (kinematics,
     mechanics, finance, thermodynamics, electricity) and then by the
     variable's declared unit. Deliberately coarse: "typical" ranges for
     ordinary textbook-style problems, not universal physical limits --
     a value outside the range isn't IMPOSSIBLE, just unusual enough to
     flag.
  2. A meaning-based positivity heuristic, independent of domain/unit:
     quantities whose MEANING implies they can't be negative (mass,
     time elapsed, distance, age, price, count, ...) get flagged if they
     come out negative and NO domain restriction was declared for them.
     If a domain WAS declared, physical_validity.py already owns that
     variable and this module stays out of the way -- this heuristic
     exists specifically for the case nothing was declared to catch it.
"""
from dataclasses import dataclass

from modules.equation_engine import ProblemModel


@dataclass
class PlausibilityNote:
    symbol: str
    meaning: str
    value: float
    unit: str | None
    category: str      # matched domain category ("kinematics", "finance", ...) or "general"
    message: str


@dataclass
class MagnitudeRange:
    unit_patterns: tuple[str, ...]   # normalized unit strings this range applies to
    typical_min: float
    typical_max: float
    label: str                        # human name for the quantity, e.g. "acceleration"


def _norm_unit(u: str | None) -> str:
    if not u:
        return ""
    u = u.strip().lower().replace(" ", "")
    u = u.replace("meters", "m").replace("metre", "m").replace("metres", "m").replace("meter", "m")
    u = u.replace("seconds", "s").replace("second", "s")
    u = u.replace("²", "^2").replace("**2", "^2")
    return u


# domain-category keyword table -- matched against model.problem_domain
# (a short free-text label the extraction LLM writes, e.g. "kinematics",
# "compound interest", "population dynamics"), not a fixed enum, so this
# is necessarily a keyword match rather than an exact lookup.
_DOMAIN_KEYWORDS: list[tuple[str, str]] = [
    ("kinemat", "kinematics"),
    ("projectile", "kinematics"),
    ("free fall", "kinematics"),
    ("finance", "finance"),
    ("interest", "finance"),
    ("invest", "finance"),
    ("loan", "finance"),
    ("compound", "finance"),
    ("annuity", "finance"),
    ("mortgage", "finance"),
    ("thermo", "thermodynamics"),
    ("heat", "thermodynamics"),
    ("temperature", "thermodynamics"),
    ("circuit", "electricity"),
    ("electric", "electricity"),
    ("current", "electricity"),
    ("voltage", "electricity"),
    ("resistance", "electricity"),
    ("dynamics", "mechanics"),
    ("mechanic", "mechanics"),
    ("force", "mechanics"),
    ("energy", "mechanics"),
    ("momentum", "mechanics"),
]


def _infer_category(problem_domain: str) -> str:
    lower = (problem_domain or "").lower()
    for keyword, category in _DOMAIN_KEYWORDS:
        if keyword in lower:
            return category
    return "general"


# Deliberately small and coarse -- textbook-word-problem "typical", not
# a physics reference table. New categories/units are cheap to add here
# as gaps are noticed; absence of an entry means this module simply has
# no opinion about that quantity, not that it's been checked and found fine.
_MAGNITUDE_TABLE: dict[str, list[MagnitudeRange]] = {
    "kinematics": [
        MagnitudeRange(("m/s",), 0, 343, "speed"),                 # up to the speed of sound in air
        MagnitudeRange(("m/s^2", "m/s2"), -100, 100, "acceleration"),  # a fighter jet pulls ~90 m/s^2
        MagnitudeRange(("km/h", "kph"), 0, 1200, "speed"),
        MagnitudeRange(("mph",), 0, 760, "speed"),
    ],
    "mechanics": [
        MagnitudeRange(("n", "newton", "newtons"), -1e6, 1e6, "force"),
        MagnitudeRange(("kg",), 0, 1e5, "mass"),
        MagnitudeRange(("j", "joule", "joules"), -1e9, 1e9, "energy"),
        MagnitudeRange(("w", "watt", "watts"), -1e7, 1e7, "power"),
    ],
    "finance": [
        MagnitudeRange(("%", "percent"), -50, 100, "rate"),   # ordinary interest/growth rates
    ],
    "thermodynamics": [
        MagnitudeRange(("k", "kelvin"), 0, 6000, "temperature"),          # 0K floor, ~sun-surface ceiling
        MagnitudeRange(("c", "celsius", "°c"), -273.15, 6000, "temperature"),
    ],
    "electricity": [
        MagnitudeRange(("a", "amp", "amps", "ampere", "amperes"), -1000, 1000, "current"),
        MagnitudeRange(("v", "volt", "volts"), -100000, 100000, "voltage"),
        MagnitudeRange(("ohm", "ohms", "Ω"), 0, 1e7, "resistance"),
    ],
}

# meaning keywords implying a physically non-negative quantity, checked
# regardless of category/unit -- catches the "negative computed mass"
# case even when problem_domain doesn't match any category above.
_NONNEGATIVE_MEANING_KEYWORDS = (
    "mass", "time", "duration", "age", "distance", "length", "height",
    "width", "radius", "diameter", "area", "volume", "price", "cost",
    "count", "number of", "period", "frequency", "population",
)


def _matches_nonnegative_meaning(meaning: str) -> str | None:
    lower = (meaning or "").lower()
    for kw in _NONNEGATIVE_MEANING_KEYWORDS:
        if kw in lower:
            return kw
    return None


def check_plausibility(model: ProblemModel, values: dict[str, float]) -> list[PlausibilityNote]:
    """`values` maps variable-symbol-name -> a numeric value to sanity-
    check -- typically the problem's own known inputs plus whatever this
    model's algebraic targets solved to. Returns [] for a symbol this
    module has no opinion about (no matching category/unit range AND no
    recognizable non-negativity-implying meaning) -- absence of a flag
    is NOT a claim that a value is fine, just that this small curated
    table doesn't cover it."""
    category = _infer_category(model.problem_domain)
    ranges = _MAGNITUDE_TABLE.get(category, [])
    var_by_symbol = {v.symbol: v for v in model.variables}

    notes: list[PlausibilityNote] = []
    for symbol, value in values.items():
        var = var_by_symbol.get(symbol)
        meaning = var.meaning if var else symbol
        unit = _norm_unit(var.unit if var else None)

        # 1. magnitude-range check, by category + unit
        for mr in ranges:
            if unit in mr.unit_patterns:
                if value < mr.typical_min or value > mr.typical_max:
                    notes.append(PlausibilityNote(
                        symbol=symbol, meaning=meaning, value=value, unit=var.unit if var else None,
                        category=category,
                        message=(f"{meaning} ({symbol}) = {value:.6g} {var.unit or ''} is outside the "
                                  f"typical range for {mr.label} in a {category} problem "
                                  f"({mr.typical_min:g} to {mr.typical_max:g}) -- worth checking against "
                                  "real-world expectations."),
                    ))
                break  # only the first unit-matching range in this category applies

        # 2. meaning-based positivity heuristic -- skipped entirely when
        # a domain restriction was already declared, since
        # physical_validity.py owns that variable in that case.
        if var is not None and var.domain is None and value < 0:
            kw = _matches_nonnegative_meaning(meaning)
            if kw:
                notes.append(PlausibilityNote(
                    symbol=symbol, meaning=meaning, value=value, unit=var.unit if var else None,
                    category=category,
                    message=(f"{meaning} ({symbol}) = {value:.6g} {var.unit or ''} came out negative, "
                              f"but nothing marked it as sign-restricted and \"{kw}\" is usually a "
                              "non-negative quantity -- worth checking against real-world expectations."),
                ))
    return notes
