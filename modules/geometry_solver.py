"""
Triangle solving (SSS, SAS, ASA, AAS, and the genuinely ambiguous SSA
case) plus a labeled Plotly schematic of the result -- the "given some
sides and angles of a triangle, find the rest, and draw it with the
angles marked" capability a lot of geometry word problems actually
need, which this app never had a way to render before (see
vector_utils.py for the pre-existing, much narrower vector-algebra
geometry support -- dot products and distances, not shapes).

Deliberately built generically enough that BOTH the standalone
"Geometry" mode (direct numeric input) AND the word-problem solver's
geometry recognition (see equation_engine.py's GeometrySpec) can call
the exact same solve_triangle()/render_triangle() functions -- there is
only one triangle-solving implementation in this codebase, not two
that could quietly drift apart.

A real, easy-to-get-wrong pitfall this module deliberately avoids:
once one angle is known via the LAW OF COSINES (acos, range 0-180
degrees, unambiguous), every OTHER angle in the same triangle is also
found via law of cosines, never law of sines' asin() -- asin only
returns values in [-90, 90], so it silently returns the WRONG angle
(a value and its supplement look identical to asin) whenever the true
angle is obtuse. This is not a hypothetical: solving a=10, b=3, C=20
degrees via asin for the next angle gives 28.1 degrees; the correct
value (confirmed by acos, and by the angles summing to exactly 180) is
151.9 degrees. asin is used ONLY where the ambiguity itself is the
subject -- the SSA case below, which can genuinely have two valid
triangles, and both must be found and reported, not silently collapsed
to one.
"""
import math
from dataclasses import dataclass

import plotly.graph_objects as go

# side/angle label convention throughout: side 'a' is opposite vertex/angle
# 'A' (i.e. side a = the segment BC), side 'b' opposite angle B (segment AC),
# side 'c' opposite angle C (segment AB) -- the universal textbook convention
_SIDES = ("a", "b", "c")
_ANGLES = ("A", "B", "C")


@dataclass
class TriangleSolution:
    a: float
    b: float
    c: float
    A: float  # degrees
    B: float
    C: float
    vertices: dict[str, tuple[float, float]]  # {"A": (x,y), "B": (x,y), "C": (x,y)}
    angle_sum_check: float  # should be exactly 180.0 -- see module docstring
    law_of_cosines_residual: float  # max residual across all 3 self-consistency checks
    verified: bool


@dataclass
class TriangleSolveResult:
    solutions: list[TriangleSolution]  # 0, 1, or 2 (the SSA ambiguous case)
    case: str  # "SSS" | "SAS" | "ASA/AAS" | "SSA" | ""
    error: str | None = None


def _place_vertices(a: float, b: float, c: float) -> dict[str, tuple[float, float]]:
    """Places the triangle in the plane for rendering: A at the origin,
    B along the positive x-axis at distance c (so side AB = c lies flat,
    a natural default orientation), C found via the law of cosines angle
    at A and side b's length. Purely a DISPLAY convenience -- the actual
    solved side lengths and angles don't depend on this choice at all,
    any placement would do; this one just reliably avoids a degenerate
    (zero-height) layout for any valid triangle."""
    A_rad = math.acos(max(-1.0, min(1.0, (b ** 2 + c ** 2 - a ** 2) / (2 * b * c))))
    return {
        "A": (0.0, 0.0),
        "B": (c, 0.0),
        "C": (b * math.cos(A_rad), b * math.sin(A_rad)),
    }


def _build_solution(a: float, b: float, c: float, A: float, B: float, C: float) -> TriangleSolution:
    """Common finishing step for every solved case: places vertices for
    rendering and runs the two self-consistency checks (angle sum,
    law-of-cosines residual) that make this "verified", not just
    computed -- the same verification-first standard applied everywhere
    else in this app (see e.g. tensor_calculus.py's metric-compatibility
    check, or ode_utils.py's numerical cross-check)."""
    vertices = _place_vertices(a, b, c)
    angle_sum = A + B + C
    residuals = [
        abs(a ** 2 - (b ** 2 + c ** 2 - 2 * b * c * math.cos(math.radians(A)))),
        abs(b ** 2 - (a ** 2 + c ** 2 - 2 * a * c * math.cos(math.radians(B)))),
        abs(c ** 2 - (a ** 2 + b ** 2 - 2 * a * b * math.cos(math.radians(C)))),
    ]
    max_residual = max(residuals)
    verified = abs(angle_sum - 180.0) < 1e-6 and max_residual < 1e-6
    return TriangleSolution(a=a, b=b, c=c, A=A, B=B, C=C, vertices=vertices,
                              angle_sum_check=angle_sum, law_of_cosines_residual=max_residual,
                              verified=verified)


def solve_triangle(knowns: dict[str, float]) -> TriangleSolveResult:
    """Solves a triangle given exactly 3 of {a, b, c, A, B, C} (sides in
    any consistent length unit, angles in DEGREES). At least one side
    must be given -- three angles alone (AAA) fix the triangle's shape
    but not its size, so there's no unique answer to return."""
    given_sides = [s for s in _SIDES if s in knowns]
    given_angles = [ang for ang in _ANGLES if ang in knowns]
    if len(knowns) != 3:
        return TriangleSolveResult(solutions=[], case="",
                                     error=f"Need exactly 3 known values (got {len(knowns)}).")
    if not given_sides:
        return TriangleSolveResult(solutions=[], case="",
                                     error="At least one side length is needed -- three angles alone "
                                            "(AAA) determine the triangle's shape but not its size.")
    for key, val in knowns.items():
        if key in _SIDES and val <= 0:
            return TriangleSolveResult(solutions=[], case="", error=f"Side {key} must be positive.")
        if key in _ANGLES and not (0 < val < 180):
            return TriangleSolveResult(solutions=[], case="",
                                         error=f"Angle {key} must be strictly between 0 and 180 degrees.")

    try:
        if len(given_sides) == 3:
            return _solve_sss(knowns)
        if len(given_sides) == 2 and len(given_angles) == 1:
            side1, side2 = given_sides
            angle = given_angles[0]
            # SAS (included angle) if the known angle sits BETWEEN the two known
            # sides -- angle X is between sides Y and Z whenever X is not the
            # same letter as either known side (e.g. sides a,b meet at vertex C)
            if angle not in (side1.upper(), side2.upper()):
                return _solve_sas(knowns, side1, side2, angle)
            return _solve_ssa(knowns, side1, side2, angle)
        if len(given_angles) == 2:
            return _solve_asa_aas(knowns)
        return TriangleSolveResult(solutions=[], case="",
                                     error="This combination of knowns doesn't correspond to a "
                                            "standard solvable case (need SSS, SAS, ASA, AAS, or SSA).")
    except (ValueError, ZeroDivisionError) as exc:
        return TriangleSolveResult(solutions=[], case="", error=f"Could not solve: {exc}")


def _solve_sss(knowns: dict[str, float]) -> TriangleSolveResult:
    a, b, c = knowns["a"], knowns["b"], knowns["c"]
    if a + b <= c or a + c <= b or b + c <= a:
        return TriangleSolveResult(solutions=[], case="SSS",
                                     error="These three side lengths can't form a triangle "
                                            "(violates the triangle inequality).")
    A = math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
    B = math.degrees(math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
    C = 180.0 - A - B
    return TriangleSolveResult(solutions=[_build_solution(a, b, c, A, B, C)], case="SSS")


def _solve_sas(knowns: dict[str, float], side1: str, side2: str, angle: str) -> TriangleSolveResult:
    s1, s2, ang = knowns[side1], knowns[side2], knowns[angle]
    third_side_letter = ({"a", "b", "c"} - {side1, side2}).pop()
    third = math.sqrt(s1 ** 2 + s2 ** 2 - 2 * s1 * s2 * math.cos(math.radians(ang)))
    sides = {side1: s1, side2: s2, third_side_letter: third}
    a, b, c = sides["a"], sides["b"], sides["c"]
    # every angle via law of cosines, never asin -- see module docstring
    A = math.degrees(math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)))
    B = math.degrees(math.acos((a ** 2 + c ** 2 - b ** 2) / (2 * a * c)))
    C = 180.0 - A - B
    return TriangleSolveResult(solutions=[_build_solution(a, b, c, A, B, C)], case="SAS")


def _solve_asa_aas(knowns: dict[str, float]) -> TriangleSolveResult:
    given_angles = [ang for ang in _ANGLES if ang in knowns]
    side_letter = [s for s in _SIDES if s in knowns][0]
    A1, A2 = knowns[given_angles[0]], knowns[given_angles[1]]
    if A1 + A2 >= 180:
        return TriangleSolveResult(solutions=[], case="ASA/AAS",
                                     error="These two angles already sum to 180 degrees or more -- "
                                            "no valid triangle.")
    angles = dict(zip(given_angles, (A1, A2)))
    angles[({"A", "B", "C"} - set(given_angles)).pop()] = 180.0 - A1 - A2
    A, B, C = angles["A"], angles["B"], angles["C"]

    known_side_val = knowns[side_letter]
    known_side_angle = side_letter.upper()
    ratio = known_side_val / math.sin(math.radians(angles[known_side_angle]))
    sides = {side_letter: known_side_val}
    for s, ang_letter in zip(_SIDES, _ANGLES):
        if s not in sides:
            sides[s] = ratio * math.sin(math.radians(angles[ang_letter]))
    a, b, c = sides["a"], sides["b"], sides["c"]
    return TriangleSolveResult(solutions=[_build_solution(a, b, c, A, B, C)], case="ASA/AAS")


def _solve_ssa(knowns: dict[str, float], side1: str, side2: str, angle: str) -> TriangleSolveResult:
    """The genuinely ambiguous case: two sides and a NON-included angle
    can correspond to zero, one, or two valid triangles -- both must be
    found and returned when two exist, not silently collapsed to
    whichever one a naive computation happens to land on first."""
    # angle is opposite one of the two known sides; call that one s_opp,
    # the other (adjacent, unknown-opposite-angle) s_adj
    if angle == side1.upper():
        s_opp, s_adj, adj_letter = knowns[side1], knowns[side2], side2
    else:
        s_opp, s_adj, adj_letter = knowns[side2], knowns[side1], side1
    ang = knowns[angle]

    sin_adj_angle = s_adj * math.sin(math.radians(ang)) / s_opp
    if sin_adj_angle > 1 + 1e-9:
        return TriangleSolveResult(solutions=[], case="SSA",
                                     error="No triangle exists with these measurements (the given "
                                            "side is too short to reach across from the given angle).")
    sin_adj_angle = min(1.0, sin_adj_angle)  # clamp float noise right at the boundary case
    adj_angle_candidates = [math.degrees(math.asin(sin_adj_angle))]
    second = 180.0 - adj_angle_candidates[0]
    if second != adj_angle_candidates[0]:
        adj_angle_candidates.append(second)

    solutions = []
    for adj_angle in adj_angle_candidates:
        third_angle = 180.0 - ang - adj_angle
        if third_angle <= 1e-9:
            continue  # this branch isn't a valid triangle -- angles wouldn't sum right
        angles = {angle: ang, adj_letter.upper(): adj_angle,
                  ({"A", "B", "C"} - {angle, adj_letter.upper()}).pop(): third_angle}
        third_side_letter = ({"a", "b", "c"} - {side1, side2}).pop()
        ratio = s_opp / math.sin(math.radians(ang))
        third_side = ratio * math.sin(math.radians(angles[third_side_letter.upper()]))
        sides = {side1: knowns.get(side1, s_opp if side1 != adj_letter else s_adj),
                 side2: knowns.get(side2, s_opp if side2 != adj_letter else s_adj),
                 third_side_letter: third_side}
        sides[side1] = knowns[side1]
        sides[side2] = knowns[side2]
        a, b, c = sides["a"], sides["b"], sides["c"]
        solutions.append(_build_solution(a, b, c, angles["A"], angles["B"], angles["C"]))

    if not solutions:
        return TriangleSolveResult(solutions=[], case="SSA",
                                     error="No valid triangle found for these measurements.")
    return TriangleSolveResult(solutions=solutions, case="SSA")


def render_triangle(solution: TriangleSolution, title: str = "Triangle") -> go.Figure:
    """A labeled Plotly schematic: the triangle's edges, each vertex
    labeled with its letter, each side labeled with its length at the
    midpoint, and each angle marked with a small arc plus its degree
    value -- the actual "schematic with labelled angles" a geometry
    problem needs, not just the raw solved numbers."""
    V = solution.vertices
    order = ["A", "B", "C", "A"]
    xs = [V[p][0] for p in order]
    ys = [V[p][1] for p in order]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", fill="toself",
                               fillcolor="rgba(46,94,170,0.08)", line=dict(width=3, color="#2E5EAA"),
                               showlegend=False, hoverinfo="skip"))

    # vertex labels, offset slightly outward from the triangle's centroid
    # so the letter doesn't sit directly on top of the angle arc
    centroid = (sum(V[p][0] for p in "ABC") / 3, sum(V[p][1] for p in "ABC") / 3)
    for p in "ABC":
        vx, vy = V[p]
        dx, dy = vx - centroid[0], vy - centroid[1]
        norm = math.hypot(dx, dy) or 1.0
        label_x, label_y = vx + 0.12 * dx / norm, vy + 0.12 * dy / norm
        fig.add_annotation(x=label_x, y=label_y, text=f"<b>{p}</b>", showarrow=False,
                             font=dict(size=16, color="#2E5EAA"))

    # side length labels at each edge's midpoint
    side_of = {"A": ("b", "c"), "B": ("a", "c"), "C": ("a", "b")}  # unused; explicit pairs below
    edges = [("A", "B", "c"), ("B", "C", "a"), ("C", "A", "b")]
    for p1, p2, side_letter in edges:
        mx, my = (V[p1][0] + V[p2][0]) / 2, (V[p1][1] + V[p2][1]) / 2
        value = getattr(solution, side_letter)
        fig.add_annotation(x=mx, y=my, text=f"{side_letter} = {value:.3g}", showarrow=False,
                             font=dict(size=12, color="#333333"), yshift=-14)

    # angle arcs: a short arc at each vertex spanning the angle between
    # its two adjacent sides, with the degree value labeled at its midpoint
    scale = min(solution.a, solution.b, solution.c) * 0.22
    for p, (n1, n2) in [("A", ("B", "C")), ("B", ("A", "C")), ("C", ("A", "B"))]:
        vx, vy = V[p]
        a1 = math.atan2(V[n1][1] - vy, V[n1][0] - vx)
        a2 = math.atan2(V[n2][1] - vy, V[n2][0] - vx)
        diff = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        steps = [a1 + diff * t / 20 for t in range(21)]
        arc_x = [vx + scale * math.cos(t) for t in steps]
        arc_y = [vy + scale * math.sin(t) for t in steps]
        fig.add_trace(go.Scatter(x=arc_x, y=arc_y, mode="lines", line=dict(width=2, color="#C0392B"),
                                   showlegend=False, hoverinfo="skip"))
        mid_angle = a1 + diff / 2
        label_r = scale * 1.5
        angle_value = getattr(solution, p)
        fig.add_annotation(x=vx + label_r * math.cos(mid_angle), y=vy + label_r * math.sin(mid_angle),
                             text=f"{angle_value:.1f}°", showarrow=False,
                             font=dict(size=11, color="#C0392B"))

    fig.update_layout(title=title, xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False,
                                                 visible=False),
                        yaxis=dict(showgrid=False, zeroline=False, visible=False),
                        margin=dict(l=20, r=20, t=40, b=20), height=450)
    return fig


def _ssa_shared_frame(sol1: TriangleSolution, sol2: TriangleSolution):
    """Recovers the classic "swinging compass" construction shared by both
    SSA solutions: a fixed pivot vertex (where the given angle sits), a
    fixed neighbor (the far end of the given ADJACENT side), and a third
    vertex that can land in one of two positions along the SAME ray from
    the pivot -- exactly where a circle of radius s_opp (the given side
    OPPOSITE the angle) centered at the neighbor crosses that ray. Returns
    (pivot, neighbor, swinging, s_adj_len, angle_deg, r1, r2, s_opp_len) or
    None if sol1/sol2 don't actually share two sides and an angle (i.e.
    aren't a genuine SSA pair -- defensive; the only caller always passes
    a real pair, but this function doesn't assume that).

    Both solutions' OWN vertex placements (see _place_vertices) are NOT
    used here -- each one places A at the origin independently, so the two
    placements generally do NOT share a common frame when the differing
    ("third") side changes vertex B's position. This reconstructs a frame
    where the GENUINELY fixed quantities (pivot, neighbor, angle) are
    actually fixed, which is what an ambiguity animation needs: something
    held constant for the eye to anchor on while the rest visibly moves.
    """
    sides1 = {"a": sol1.a, "b": sol1.b, "c": sol1.c}
    sides2 = {"a": sol2.a, "b": sol2.b, "c": sol2.c}
    angles1 = {"A": sol1.A, "B": sol1.B, "C": sol1.C}
    angles2 = {"A": sol2.A, "B": sol2.B, "C": sol2.C}
    shared_sides = [k for k in "abc" if math.isclose(sides1[k], sides2[k], rel_tol=1e-6, abs_tol=1e-9)]
    shared_angle = next((k for k in "ABC" if math.isclose(angles1[k], angles2[k], rel_tol=1e-6, abs_tol=1e-9)),
                          None)
    if len(shared_sides) != 2 or shared_angle is None:
        return None

    pivot = shared_angle
    s_opp_letter = pivot.lower()  # the side opposite the pivot's own angle
    if s_opp_letter not in shared_sides:
        return None  # not the shape of a genuine SSA pair
    s_adj_letter = next(s for s in shared_sides if s != s_opp_letter)
    neighbor = next(v for v in "ABC" if v != pivot and v != s_adj_letter.upper())
    swinging = next(v for v in "ABC" if v not in (pivot, neighbor))

    s_adj_len = sides1[s_adj_letter]
    s_opp_len = sides1[s_opp_letter]
    ang = angles1[pivot]
    r1 = getattr(sol1, neighbor.lower())  # pivot-to-swinging distance in each solution
    r2 = getattr(sol2, neighbor.lower())
    return pivot, neighbor, swinging, s_adj_len, ang, r1, r2, s_opp_len


def build_ssa_ambiguity_animation(sol1: TriangleSolution, sol2: TriangleSolution,
                                    n_frames: int = 30) -> go.Figure | None:
    """Animates the morph between the two valid SSA triangles by sliding the
    swinging vertex from its first valid position to its second, along the
    ray fixed by the given angle -- with the pivot vertex, the neighbor
    vertex, and the constraining circle (radius = the given side opposite
    the angle) all drawn as fixed reference geometry. This is the SSA
    ambiguity made literal: the same two given sides and angle, and TWO
    genuinely different places the third vertex can land. Returns None when
    sol1/sol2 don't share the two-sides-plus-angle structure a real SSA
    pair has (see _ssa_shared_frame) -- the caller (both
    ui/results/summary.py and ui/geometry.py) simply skips the animation
    in that case rather than showing something built on a wrong premise.
    """
    frame_data = _ssa_shared_frame(sol1, sol2)
    if frame_data is None:
        return None
    pivot, neighbor, swinging, s_adj, ang, r1, r2, s_opp = frame_data

    P = (0.0, 0.0)
    Q = (s_adj, 0.0)
    ang_rad = math.radians(ang)
    ray_dir = (math.cos(ang_rad), math.sin(ang_rad))
    R1 = (r1 * ray_dir[0], r1 * ray_dir[1])
    R2 = (r2 * ray_dir[0], r2 * ray_dir[1])

    # the constraining circle: every point exactly s_opp away from the
    # neighbor -- both R1 and R2 lie exactly on it by construction, and
    # seeing that visually is the point of drawing it at all
    circle_t = [i * 2 * math.pi / 100 for i in range(101)]
    circle_x = [Q[0] + s_opp * math.cos(t) for t in circle_t]
    circle_y = [Q[1] + s_opp * math.sin(t) for t in circle_t]

    ray_len = max(r1, r2) * 1.15
    static_traces = [
        go.Scatter(x=circle_x, y=circle_y, mode="lines", line=dict(color="rgba(160,120,0,0.5)", dash="dot"),
                    name=f"circle: {swinging} is always {s_opp:.3g} from {neighbor}", hoverinfo="skip"),
        go.Scatter(x=[P[0], P[0] + ray_len * ray_dir[0]], y=[P[1], P[1] + ray_len * ray_dir[1]],
                    mode="lines", line=dict(color="rgba(100,100,100,0.4)", dash="dash"),
                    name=f"ray: where {swinging} must lie", hoverinfo="skip"),
        go.Scatter(x=[R1[0]], y=[R1[1]], mode="markers", marker=dict(symbol="x", size=10, color="#888"),
                    name="solution 1", hoverinfo="skip"),
        go.Scatter(x=[R2[0]], y=[R2[1]], mode="markers", marker=dict(symbol="x", size=10, color="#888"),
                    name="solution 2", hoverinfo="skip"),
    ]

    def moving_traces(f: float):
        Rf = (R1[0] + f * (R2[0] - R1[0]), R1[1] + f * (R2[1] - R1[1]))
        return [
            go.Scatter(x=[P[0], Q[0], Rf[0], P[0]], y=[P[1], Q[1], Rf[1], P[1]],
                        mode="lines", fill="toself", fillcolor="rgba(46,94,170,0.12)",
                        line=dict(width=3, color="#2E5EAA"), name="triangle", hoverinfo="skip"),
            go.Scatter(x=[P[0], Q[0], Rf[0]], y=[P[1], Q[1], Rf[1]], mode="markers+text",
                        marker=dict(size=12, color="#2E5EAA"),
                        text=[pivot, neighbor, swinging], textposition="top center",
                        name="vertices", hoverinfo="skip"),
        ]

    steps = [i / (n_frames - 1) for i in range(n_frames)]
    frames = [go.Frame(data=moving_traces(f), traces=[4, 5], name=f"{i}") for i, f in enumerate(steps)]

    fig = go.Figure(
        data=static_traces + moving_traces(0.0),
        layout=go.Layout(
            title=f"SSA ambiguity: {swinging} swinging between two valid positions",
            xaxis=dict(scaleanchor="y", showgrid=False, zeroline=False, visible=False),
            yaxis=dict(showgrid=False, zeroline=False, visible=False),
            margin=dict(l=20, r=20, t=40, b=20), height=450, showlegend=True,
            updatemenus=[dict(type="buttons", showactive=False, x=0.05, y=1.1, buttons=[
                dict(label="\u25b6 Play", method="animate",
                      args=[None, {"frame": {"duration": 60, "redraw": True},
                                     "fromcurrent": True, "transition": {"duration": 0}}]),
                dict(label="\u23f8 Pause", method="animate",
                      args=[[None], {"frame": {"duration": 0}, "mode": "immediate"}]),
            ])],
            sliders=[dict(currentvalue={"prefix": "solution 1 \u2192 2: "}, x=0.05, len=0.9, steps=[
                dict(method="animate", args=[[f"{i}"],
                      {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}],
                      label=f"{f:.2f}") for i, f in enumerate(steps)])],
        ),
        frames=frames,
    )
    return fig
