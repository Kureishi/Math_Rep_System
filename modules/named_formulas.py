"""
Named-formula recognizer: a small curated pattern library (analogous to
plausibility.py's magnitude-range table) that recognizes when a derived
equation matches a well-known named result -- "this is Newton's second
law," "this is compound interest," "this is the Pythagorean theorem" --
and labels it. Purely a pedagogical/provenance touch: naming a formula a
student may already recognize from a textbook adds credibility and a
learning hook that a bare derived equation doesn't.

Matching is STRUCTURAL -- the same variable-name-independent fingerprint
similarity.py's canonicalize_equation() already computes for "find
similar past problems" -- not full algebraic equivalence. To cover the
common case of a formula being solved for a different one of its own
variables (F=m*a vs a=F/m vs m=F/a), each named formula is stored as ONE
base equation and every algebraic rearrangement (solving for each of its
own symbols in turn) is precomputed once at import time and matched
against too. This still won't recognize an equation that's been
rearranged some OTHER algebraically-equivalent way (e.g. F - m*a = 0
written as a difference rather than an assignment) -- a documented scope
limit, the same one self_consistency.py's shapes_match already has, not
a silent gap.

Two structurally IDENTICAL formulas are a real possibility with a
purely structural approach -- "y = m*x + b" (slope-intercept) and
"v = v0 + a*t" (kinematics) canonicalize to the exact same shape, since
they're the same algebraic form with different variable meanings. When
a shape matches more than one candidate, the model's own problem_domain
label is used to disambiguate (each formula carries a loose set of
domain keywords, the same lightweight substring-matching approach
plausibility.py uses); when domain doesn't resolve it either, every
tied candidate is returned rather than silently guessing one.
"""
import itertools

import sympy as sp

from modules.equation_engine import Equation
from modules.timeout_utils import run_with_timeout, ComputationTimeoutError

_MAX_PERMUTATION_SYMBOLS = 4  # 4! = 24 permutations, cheap even for a complex expression;
                                # a 5th+ symbol makes the full search too slow for formulas
                                # involving powers/logs (compound interest, a quadratic-derived
                                # sqrt rearrangement), so those fall back to the cheaper,
                                # occasionally-imperfect single-pass ordering below


def _fallback_shape(eq: sp.Eq) -> str:
    """The cheap O(1) fallback for equations with too many free symbols
    to permutation-search -- a single preorder-traversal-based
    placeholder assignment, same technique as
    similarity.canonicalize_equation(). Can occasionally miss a match
    for the same reason documented on _true_canonical_shape (sympy's
    internal Add/Mul argument order depends partly on symbol names), but
    that's a documented, narrow trade for staying fast on the handful of
    5+-symbol formulas in this module's table."""
    seen_order: list[sp.Symbol] = []
    seen_set: set[sp.Symbol] = set()
    for node in sp.preorder_traversal(eq):
        if isinstance(node, sp.Symbol) and node not in seen_set:
            seen_set.add(node)
            seen_order.append(node)
    mapping = {s: sp.Symbol(f"_s{i}") for i, s in enumerate(seen_order)}
    return sp.srepr(eq.xreplace(mapping))


def _true_canonical_shape(eq: sp.Eq) -> str:
    """A stronger, permutation-invariant canonical fingerprint than
    similarity.canonicalize_equation() alone provides. That function's
    placeholder order comes from sp.preorder_traversal -- which,
    surprisingly, isn't purely a function of the equation's STRUCTURE:
    sympy's own internal storage order for commutative Add/Mul
    arguments is partly determined by the actual symbol NAMES involved.
    Two structurally-identical equations with different symbol names can
    therefore get their placeholders assigned in a different order and
    fail to match even though nothing about the underlying shape
    differs. Discovered while building this module -- "a = (v_f - v_i)
    / t" failed to match its own registered kinematics-formula
    rearrangement (the algebraically identical "a = (v - v0) / t")
    despite being the exact same shape, purely because of which
    variable names sympy happened to sort first internally.

    Tries every permutation of a placeholder alphabet against the
    equation's free symbols and keeps the lexicographically smallest
    resulting sp.srepr() -- a TRUE canonical form, independent of both
    symbol names and sympy's internal argument-storage order. Capped at
    _MAX_PERMUTATION_SYMBOLS free symbols (see _fallback_shape for what
    happens above that) -- O(n!) is fine at n<=4, but blows up on a
    complex power/log expression well before n=5-6."""
    symbols = sorted(eq.free_symbols, key=lambda s: s.name)
    if not symbols:
        return sp.srepr(eq)
    if len(symbols) > _MAX_PERMUTATION_SYMBOLS:
        return _fallback_shape(eq)
    best = None
    for perm in itertools.permutations(range(len(symbols))):
        mapping = {symbols[i]: sp.Symbol(f"_s{perm[i]}") for i in range(len(symbols))}
        # xreplace (literal, non-simplifying substitution) rather than
        # subs -- both give the identical resulting structure here since
        # every substitution is symbol-for-symbol, but subs re-triggers
        # sympy's automatic simplification/reflattening on every call,
        # which is the actual cost driver for a formula with a power
        # tower (compound interest's (1+r/n)^(n*t), e.g.) -- xreplace
        # is a straight structural swap and measured ~4-5x faster for
        # exactly this reason
        rep = sp.srepr(eq.xreplace(mapping))
        if best is None or rep < best:
            best = rep
    assert best is not None  # itertools.permutations always yields at least one permutation
    # (even of an empty sequence, it yields exactly one empty tuple), so the loop above always
    # runs at least once
    return best


_BASE_FORMULAS: list[tuple[str, str, sp.Eq, tuple[str, ...]]] = []


def _register(name: str, description: str, eq: sp.Eq, domain_keywords: tuple[str, ...] = ()):
    _BASE_FORMULAS.append((name, description, eq, domain_keywords))


# ---------------------------------------------------------------- mechanics / kinematics

_F, _m, _a = sp.symbols("F m a")
_register("Newton's second law", "force = mass × acceleration", sp.Eq(_F, _m * _a),
           ("force", "newton", "acceleration"))

_v, _v0, _at, _t = sp.symbols("v v0 a t")
_register("Kinematics: velocity-time", "v = v₀ + a·t", sp.Eq(_v, _v0 + _at * _t),
           ("kinemat", "motion", "velocity"))

_x, _x0, _v0b, _t2, _a2 = sp.symbols("x x0 v0 t a")
_register("Kinematics: position-time", "x = x₀ + v₀t + ½at²",
           sp.Eq(_x, _x0 + _v0b * _t2 + sp.Rational(1, 2) * _a2 * _t2**2),
           ("kinemat", "motion", "position", "displacement"))

_KE, _mk, _vk = sp.symbols("KE m v")
_register("Kinetic energy", "KE = ½mv²", sp.Eq(_KE, sp.Rational(1, 2) * _mk * _vk**2),
           ("kinetic", "energy"))

_p, _mp, _vp = sp.symbols("p m v")
_register("Momentum", "p = mv", sp.Eq(_p, _mp * _vp), ("momentum",))

_W, _Fw, _d = sp.symbols("W F d")
_register("Work", "W = F·d", sp.Eq(_W, _Fw * _d), ("work",))

_c, _av, _bv = sp.symbols("c a_ b_")
_register("Pythagorean theorem", "c² = a² + b² for a right triangle",
           sp.Eq(_c**2, _av**2 + _bv**2), ("geometry", "triangle", "pythagor", "hypotenuse"))

_rho, _mr, _V = sp.symbols("rho m V")
_register("Density", "ρ = m / V", sp.Eq(_rho, _mr / _V), ("density",))

_Fg, _G, _m1, _m2, _rg = sp.symbols("F G m1 m2 r")
_register("Newton's law of universal gravitation", "F = Gm₁m₂ / r²",
           sp.Eq(_Fg, _G * _m1 * _m2 / _rg**2), ("gravit", "astro", "orbit"))

# ---------------------------------------------------------------- electricity

_V_, _I_, _R_ = sp.symbols("V I R")
_register("Ohm's law", "V = IR", sp.Eq(_V_, _I_ * _R_), ("voltage", "current", "resistance", "ohm"))

_P_, _I2, _R2 = sp.symbols("P I R")
_register("Electrical power (I²R form)", "P = I²R", sp.Eq(_P_, _I2**2 * _R2),
           ("electric", "circuit", "power", "watt"))

_Fc, _kc, _q1, _q2, _rc = sp.symbols("F k q1 q2 r")
_register("Coulomb's law", "F = kq₁q₂ / r²", sp.Eq(_Fc, _kc * _q1 * _q2 / _rc**2),
           ("charge", "coulomb"))

# ---------------------------------------------------------------- finance

_A, _P, _r, _n, _t3 = sp.symbols("A P r n t")
_register("Compound interest", "A = P(1 + r/n)^(nt)",
           sp.Eq(_A, _P * (1 + _r / _n) ** (_n * _t3)), ("financ", "interest", "compound", "invest"))

_A2, _P2, _r2, _t4 = sp.symbols("A P r t")
_register("Simple interest", "A = P(1 + rt)", sp.Eq(_A2, _P2 * (1 + _r2 * _t4)),
           ("financ", "interest", "simple", "loan"))

# ---------------------------------------------------------------- thermodynamics

_Pg, _Vg, _n2, _Rg, _Tg = sp.symbols("P V n R T")
_register("Ideal gas law", "PV = nRT", sp.Eq(_Pg * _Vg, _n2 * _Rg * _Tg),
           ("thermo", "gas", "ideal"))

_Q, _mq, _cq, _dT = sp.symbols("Q m c dT")
_register("Specific heat", "Q = mcΔT", sp.Eq(_Q, _mq * _cq * _dT),
           ("thermo", "heat", "temperature"))


def _all_rearrangements(eq: sp.Eq) -> list[sp.Eq]:
    """The base equation plus one rearrangement per own free symbol
    (solved for that symbol) -- covers "F=m*a" being recognized whether
    written as F=m*a, a=F/m, or m=F/a. Each solve gets its own short
    timeout: some formulas (compound interest solved for the rate or
    the exponent, e.g.) are genuinely slow for sp.solve to invert
    (transcendental, LambertW-branch algebra) -- a timed-out symbol is
    simply skipped rather than letting one hard rearrangement stall
    this module's import for every user of the app."""
    variants = [eq]
    for sym in sorted(eq.free_symbols, key=lambda s: s.name):
        try:
            solutions = run_with_timeout(sp.solve, eq, sym, timeout=0.3,
                                           label=f"named-formula rearrangement for {sym}")
        except ComputationTimeoutError:
            continue
        except Exception:  # noqa: BLE001
            continue
        for sol in solutions:
            if sym not in sol.free_symbols:
                variants.append(sp.Eq(sym, sol))
    return variants


def _build_shape_table() -> dict[str, list[tuple[str, str, tuple[str, ...]]]]:
    table: dict[str, list[tuple[str, str, tuple[str, ...]]]] = {}
    for name, description, base_eq, keywords in _BASE_FORMULAS:
        for variant in _all_rearrangements(base_eq):
            try:
                shape = _true_canonical_shape(variant)
            except Exception:  # noqa: BLE001
                continue
            entries = table.setdefault(shape, [])
            if not any(e[0] == name for e in entries):
                entries.append((name, description, keywords))
    return table


_SHAPE_TABLE: dict[str, list[tuple[str, str, tuple[str, ...]]]] | None = None


def _get_shape_table() -> dict[str, list[tuple[str, str, tuple[str, ...]]]]:
    """Built lazily on first use, not at import time -- computing every
    base formula's rearrangements (each its own sp.solve() call) plus
    their permutation-invariant shapes takes a few seconds even with
    the optimizations above, and paying that cost unconditionally on
    every app startup (whether or not this feature is ever used in a
    given session) is worse than paying it once, the first time a
    problem's derived equation is actually displayed."""
    global _SHAPE_TABLE
    if _SHAPE_TABLE is None:
        _SHAPE_TABLE = _build_shape_table()
    return _SHAPE_TABLE


def recognize_formula(eq: Equation, context_text: str = "") -> list[tuple[str, str]]:
    """Returns a list of (name, description) candidates whose shape
    matches `eq` -- empty if none matched, a single entry for the
    unambiguous common case, or more than one when the shape is
    genuinely ambiguous (see module docstring) and `context_text`'s
    keywords didn't resolve it either.

    `context_text` should be more than just the problem's domain label:
    several formulas of the same broad category collide on shape alone
    (F=m*a, p=m*v, W=F*d, and m=ρV, e.g., are ALL "y = a*b" once
    canonicalized) and a domain tag like "mechanics" matches all of
    them equally. Including the equation's OWN variable MEANING strings
    (e.g. "force", "acceleration") in `context_text` is what actually
    resolves that -- a formula whose own keyword ("force", "newton")
    appears in that text is a much stronger signal than a shared
    high-level domain ever is."""
    if eq.kind != "equation" or eq.sympy_eq is None:
        return []
    try:
        shape = _true_canonical_shape(eq.sympy_eq)
    except Exception:  # noqa: BLE001
        return []

    candidates = _get_shape_table().get(shape, [])
    if len(candidates) <= 1:
        return [(name, desc) for name, desc, _ in candidates]

    context_lower = (context_text or "").lower()
    domain_matches = [(name, desc) for name, desc, keywords in candidates
                        if any(kw in context_lower for kw in keywords)]
    if len(domain_matches) == 1:
        return domain_matches
    return [(name, desc) for name, desc, _ in candidates]
