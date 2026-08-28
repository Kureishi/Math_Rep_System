"""
"Find similar past problems": structural similarity based on equation
SHAPE rather than problem-text wording -- two problems that both boil
down to "a = (v_f - v_i) / t" should match even if one is about a car
and the other's about a rocket, and even though their variable names
might differ (a/v_f/v_i/t vs x/y/z/w). A pure text/keyword search over
problem_text wouldn't catch that; a plain equation-string match
wouldn't either (different variable names -> different strings).

canonicalize_equation() strips away what's arbitrary (variable NAMES)
while preserving what's structural (which operations combine which
symbols, and any numeric coefficients -- 0.5*a*t**2 and a*t**2 are
different formulas even with identical symbol names). Two problems are
compared by the Jaccard similarity of their equation-shape SETS (order-
independent, since a problem with the same two equations listed in a
different order should still match perfectly).

Scoped to "equation"/"ode"/"recurrence"-kind relations with plain
Symbol free variables. ODE/recurrence function NAMES (the "T" in T(t))
aren't themselves canonicalized -- only symbols appearing as plain
sp.Symbol nodes are (independent variables like t/n, and any other
symbols used inside the relation) -- so two ODEs with the same structure
but different function names won't match as closely as they might
otherwise. Documented as a scope limitation rather than quietly wrong:
canonicalizing applied-function names too is a reasonable future
extension, left out here to keep the canonicalization pass simple and
because plain-symbol equations (the algebraic case) are the overwhelming
majority of problems this matters for.
"""
import sympy as sp

from modules.equation_engine import ProblemModel


def canonicalize_equation(eq: sp.Eq) -> str:
    """Replaces every plain Symbol in `eq` with an anonymous placeholder
    (_s0, _s1, ... in order of first appearance via preorder traversal),
    leaving numeric coefficients and the overall operator structure
    untouched, then returns sp.srepr() of the result -- a canonical,
    name-independent fingerprint of the equation's shape."""
    seen_order: list[sp.Symbol] = []
    seen_set: set[sp.Symbol] = set()
    for node in sp.preorder_traversal(eq):
        if isinstance(node, sp.Symbol) and node not in seen_set:
            seen_set.add(node)
            seen_order.append(node)
    mapping = {s: sp.Symbol(f"_s{i}") for i, s in enumerate(seen_order)}
    return sp.srepr(eq.subs(mapping))


def problem_shape(model: ProblemModel) -> frozenset[str]:
    """The set of canonical equation-shape fingerprints for a whole
    problem, one per equation/ode/recurrence-kind relation, prefixed
    with its kind so e.g. an "equation" and an "ode" that happen to
    canonicalize to the same underlying shape string don't collide.
    Inequality-kind relations and objectives are left out -- their
    "shape" isn't a plain equation to canonicalize the same way."""
    shapes = set()
    for eq in model.equations:
        if eq.kind not in ("equation", "ode", "recurrence") or eq.sympy_eq is None:
            continue
        try:
            shapes.add(f"{eq.kind}:{canonicalize_equation(eq.sympy_eq)}")
        except Exception:  # noqa: BLE001
            continue
    return frozenset(shapes)


def jaccard_similarity(a: frozenset[str], b: frozenset[str]) -> float:
    """|intersection| / |union|, in [0, 1]. Two empty shape sets (e.g.
    two objective-only optimization problems with no equation-kind
    relations at all) return 0.0 -- there's no structural signal to
    call them "similar" on, even though the sets are technically equal."""
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def find_similar_shapes(target_shape: frozenset[str], candidates: list[tuple[int, frozenset[str]]],
                          limit: int = 5, min_similarity: float = 0.3) -> list[tuple[int, float]]:
    """Ranks `candidates` (a list of (id, shape) pairs, e.g. loaded from
    history) by Jaccard similarity to target_shape, keeping only those
    at or above min_similarity, highest first."""
    scored = [(cid, jaccard_similarity(target_shape, shape)) for cid, shape in candidates]
    scored = [(cid, s) for cid, s in scored if s >= min_similarity]
    scored.sort(key=lambda pair: pair[1], reverse=True)
    return scored[:limit]
