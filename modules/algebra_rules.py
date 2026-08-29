"""
Algebra-rule tagging: names which algebraic TECHNIQUE was needed to
isolate a target in an equation, as a one-line tag attached to a solve
step. sp.solve() doesn't expose the incremental moves it makes
internally (there's no "trace" of "subtract 3 from both sides, then
divide by 2" to read back out) -- rather than fabricate a step sequence
sp.solve() never actually took, this module does a STRUCTURAL
classification of the equation itself: what shape does the target
appear in, and what's the standard named technique for that shape?
That's honest about being a classification, not a derivation, while
still naming the technique (linear isolation, quadratic formula, taking
a root, applying an inverse function, ...) the way a textbook section
heading would.

Classification order (first match wins), from most to least specific:
1. Target appears on both sides -> collect-and-combine.
2. Target-side expression is a genuine polynomial in the target
   (sp.Poly succeeds) -> classified by degree (linear/quadratic/higher).
3. Target appears under a root or in a denominator (a Pow node with a
   fractional or negative exponent) -> root / reciprocal.
4. Target appears inside a named function (log, exp, a trig function) ->
   isolated via that function's inverse.
5. Nothing matched -> a generic fallback tag, not a wrong guess.
"""
import sympy as sp

_INVERSE_FUNC_NAMES = {
    "log": "exponentiating both sides (log's inverse)",
    "exp": "taking the natural log of both sides",
    "sin": "applying arcsin (inverse sine)",
    "cos": "applying arccos (inverse cosine)",
    "tan": "applying arctan (inverse tangent)",
    "asin": "applying sine (inverse of arcsin)",
    "acos": "applying cosine (inverse of arccos)",
    "atan": "applying tangent (inverse of arctan)",
}

_DEGREE_NAMES = {1: "linear", 2: "quadratic"}


def classify_isolation(equation: sp.Eq, target: sp.Symbol) -> str:
    """Returns a short, human-readable tag describing the technique
    used to isolate `target` in `equation`. Never raises -- falls back
    to a generic tag if the equation's shape doesn't match any of the
    named cases below, rather than guessing wrong."""
    if not isinstance(equation, sp.Eq):
        # sp.Eq() of two identical or two purely-numeric sides
        # auto-evaluates straight to a BooleanTrue/BooleanFalse rather
        # than staying an Eq object (e.g. Eq(x, x) -> True) -- nothing
        # to isolate in either case.
        return "this equation is a trivial identity with nothing to isolate"

    lhs_has = target in equation.lhs.free_symbols
    rhs_has = target in equation.rhs.free_symbols
    if not lhs_has and not rhs_has:
        return "target doesn't appear in this equation"
    if lhs_has and rhs_has:
        return ("target appears on both sides: collect every occurrence onto one side, combine "
                "like terms, then isolate (dividing by its combined coefficient, or factoring "
                "if it appears nonlinearly)")

    expr = sp.expand(equation.lhs - equation.rhs)

    try:
        degree = sp.Poly(expr, target).degree()
        name = _DEGREE_NAMES.get(degree, f"degree-{degree} polynomial")
        if degree == 1:
            return "linear in the target: move every other term to the other side, then divide " \
                   "by the target's coefficient"
        if degree == 2:
            return "quadratic in the target: solved via the quadratic formula (or factoring)"
        return f"{name} in the target: solved via SymPy's general polynomial solver"
    except sp.PolynomialError:
        pass

    for node in sp.preorder_traversal(expr):
        if isinstance(node, sp.Pow) and node.exp.is_number and target in node.base.free_symbols:
            if node.exp.is_negative:
                return "target appears in a denominator: isolated by multiplying both sides by " \
                       "that denominator, then dividing"
            if node.exp.is_Rational and node.exp.q != 1:
                return "target appears under a root: isolated by raising both sides to the " \
                       "matching power"

    for func_name, description in _INVERSE_FUNC_NAMES.items():
        func = getattr(sp, func_name)
        atoms = [a for a in expr.atoms(func) if target in a.free_symbols]
        if atoms:
            return f"target appears inside {func_name}(...): isolated by {description}"

    return "isolated via SymPy's general equation solver (no single named technique matched " \
           "this equation's shape)"
