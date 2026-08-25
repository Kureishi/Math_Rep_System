"""
Vector algebra helpers, exposed two ways:

1. As plain functions (`dot`, `cross`, `magnitude`, `unit_vector`,
   `angle_between`, `distance`) other modules can call directly.
2. Bound into equation_engine._local_dict() under the same names, so the
   extraction LLM can write a normal "equation"-kind relation like
   `Eq(W, dot(F, d))` and have it parse straight into a scalar SymPy
   expression -- F and d stay genuine vectors (built from the variable's
   declared components) all the way up to the moment the dot/cross
   product collapses them to a number. This is deliberately NOT a new
   equation "kind": a dot/cross/magnitude expression reduces to an
   ordinary scalar the instant it's evaluated (see the parser check
   below), so it flows through the existing equation/verification/
   solving pipeline unchanged. What changes is that the LLM never has to
   pre-decompose "a force of 10N at 30 degrees" into Fx/Fy itself and
   hope the decomposition was faithful to what "dot product" or "cross
   product" actually mean -- SymPy computes the projection/product from
   genuine vector objects.

Represents a vector as a SymPy column Matrix (2 or 3 rows) of its
component symbols/values -- chosen over sympy.vector's CoordSys3D
because Matrix composes directly with the existing parse_expr/local_dict
machinery (a CoordSys3D basis vector isn't something parse_expr's
function-call syntax can easily construct from a components list).
"""
import sympy as sp

VectorLike = sp.Matrix


def make_vector(components: list) -> sp.Matrix:
    """Builds a column vector from raw components (symbols, numbers, or
    already-parsed SymPy expressions)."""
    return sp.Matrix([sp.sympify(c) for c in components])


def _as_vector3(v: sp.Matrix) -> sp.Matrix:
    """Pads a 2-component vector to 3D (zero z-component) so cross()
    can always use SymPy's built-in 3D cross product under the hood."""
    if v.shape == (3, 1):
        return v
    if v.shape == (2, 1):
        return sp.Matrix([v[0], v[1], sp.Integer(0)])
    raise ValueError(f"Expected a 2- or 3-component vector, got shape {v.shape}")


def dot(u: sp.Matrix, v: sp.Matrix) -> sp.Expr:
    """Dot product -> always a scalar."""
    if u.shape != v.shape:
        raise ValueError(f"dot(): mismatched vector dimensions {u.shape} vs {v.shape}")
    return sp.expand((u.T * v)[0])


def cross(u: sp.Matrix, v: sp.Matrix) -> sp.Matrix | sp.Expr:
    """Cross product. Two 3-vectors give a 3-vector (the usual case,
    e.g. torque r x F in 3D). Two 2-vectors give the SCALAR z-component
    of what would be the 3D cross product -- the standard convention for
    2D torque/angular-momentum problems (r x F reduces to a single
    signed number when everything lies in the xy-plane)."""
    if u.shape == (2, 1) and v.shape == (2, 1):
        return sp.expand(u[0] * v[1] - u[1] * v[0])
    return _as_vector3(u).cross(_as_vector3(v))


def magnitude(v) -> sp.Expr:
    """Euclidean norm. Works on a vector (Matrix) or falls back to
    absolute value for a plain scalar, so it's safe to call even if a
    caller passes an already-scalar expression."""
    if hasattr(v, "shape"):
        return sp.sqrt(sp.expand(dot(v, v)))
    return sp.Abs(v)


def unit_vector(v: sp.Matrix) -> sp.Matrix:
    """Vector divided by its own magnitude. Left symbolic (not
    simplified into a single fraction) so it stays readable in step
    output; callers that need a numeric answer should substitute knowns
    and evaluate afterward."""
    m = magnitude(v)
    return v / m


def angle_between(u: sp.Matrix, v: sp.Matrix) -> sp.Expr:
    """Angle between two vectors, in radians, via acos(u.v / (|u||v|))."""
    return sp.acos(dot(u, v) / (magnitude(u) * magnitude(v)))


def angle_between_deg(u: sp.Matrix, v: sp.Matrix) -> sp.Expr:
    """Same as angle_between but converted to degrees -- offered
    separately (rather than making callers remember `* 180/pi`) since
    almost every physics word problem wants degrees in the final answer."""
    return angle_between(u, v) * 180 / sp.pi


def distance(a, b) -> sp.Expr:
    """Euclidean distance between two points/vectors. Accepts either
    sympy.geometry Point/Point2D/Point3D objects (uses their own
    .distance(), which also handles point-to-line etc. if extended
    later) or plain vectors/Matrices (falls back to magnitude(a - b))."""
    if hasattr(a, "distance"):
        return a.distance(b)
    return magnitude(sp.Matrix(a) - sp.Matrix(b))


def make_point(*coords):
    """sp.Point2D for 2 coordinates, sp.Point3D for 3 -- lets the parser
    call a single `Point(...)` name regardless of dimensionality."""
    if len(coords) == 2:
        return sp.Point2D(*coords)
    if len(coords) == 3:
        return sp.Point3D(*coords)
    raise ValueError("Point() needs 2 or 3 coordinates")


def vector_local_dict() -> dict:
    """The names to fold into equation_engine._local_dict() so vector
    expressions parse directly. Kept as its own function (rather than
    equation_engine importing each name individually) so adding a new
    vector helper here only requires listing it in one place."""
    return {
        "Vector": lambda *args: make_vector(list(args)),
        "dot": dot,
        "cross": cross,
        "magnitude": magnitude,
        "norm": magnitude,          # common alias
        "unit": unit_vector,
        "angle_between": angle_between,
        "angle_between_deg": angle_between_deg,
        "distance": distance,
        "Point": make_point,
    }


def vector_summary(name: str, components: list[str], subs: dict) -> dict | None:
    """Given a declared vector variable's component symbol names and a
    {Symbol: value} substitution dict of known values, returns a small
    display-ready summary (numeric components, magnitude, unit vector)
    if every component is known -- otherwise None (nothing numeric to
    show yet, e.g. before the user has filled in values)."""
    syms = [sp.Symbol(c) for c in components]
    if not all(s in subs for s in syms):
        return None
    v = make_vector([subs[s] for s in syms])
    mag = magnitude(v)
    try:
        mag_val = float(mag)
    except (TypeError, ValueError):
        mag_val = None
    return {
        "name": name,
        "components": {c: float(subs[sp.Symbol(c)]) for c in components},
        "magnitude": mag_val,
        "magnitude_expr": mag,
    }
