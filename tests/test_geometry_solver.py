"""Tests for modules/geometry_solver.py -- triangle solving (SSS, SAS,
ASA/AAS, SSA) and its labeled Plotly schematic."""
import math

from modules.geometry_solver import solve_triangle, render_triangle, build_ssa_ambiguity_animation, \
    TriangleSolution


# --------------------------------------------------------------------- SSS
def test_sss_known_triangle():
    result = solve_triangle({"a": 5, "b": 6, "c": 7})
    assert result.error is None
    assert result.case == "SSS"
    assert len(result.solutions) == 1
    sol = result.solutions[0]
    assert sol.verified
    assert abs(sol.A + sol.B + sol.C - 180.0) < 1e-9
    assert abs(sol.A - 44.415) < 0.01


def test_sss_triangle_inequality_violation_rejected():
    result = solve_triangle({"a": 1, "b": 1, "c": 10})
    assert result.error is not None
    assert "triangle inequality" in result.error.lower()


# --------------------------------------------------------------------- SAS
def test_sas_acute_case():
    result = solve_triangle({"a": 5, "b": 6, "C": 60})
    assert result.error is None
    sol = result.solutions[0]
    assert sol.verified
    assert abs(sol.c - 5.5678) < 0.01


def test_sas_obtuse_angle_computed_correctly_not_via_asin_ambiguity():
    """Regression test for a real bug class caught during development:
    naively using asin() to find a second angle after law-of-cosines
    silently returns the wrong angle (a value's supplement) whenever
    that angle is obtuse. a=10, b=3, C=20 degrees produces an obtuse
    angle A ~ 151.9 degrees -- asin() would incorrectly give ~28.1."""
    result = solve_triangle({"a": 10, "b": 3, "C": 20})
    assert result.error is None
    sol = result.solutions[0]
    assert sol.verified
    assert sol.A > 90  # the obtuse angle, correctly identified
    assert abs(sol.A - 151.868) < 0.01


# --------------------------------------------------------------------- ASA / AAS
def test_asa_included_side():
    result = solve_triangle({"A": 40, "B": 60, "c": 10})
    assert result.error is None
    sol = result.solutions[0]
    assert sol.verified
    assert abs(sol.C - 80.0) < 1e-9
    assert abs(sol.c - 10) < 1e-9


def test_aas_non_included_side():
    result = solve_triangle({"A": 40, "B": 60, "a": 10})
    assert result.error is None
    sol = result.solutions[0]
    assert sol.verified
    assert abs(sol.a - 10) < 1e-9


def test_two_angles_summing_past_180_rejected():
    result = solve_triangle({"A": 100, "B": 90, "c": 5})
    assert result.error is not None


# --------------------------------------------------------------------- SSA (ambiguous)
def test_ssa_two_valid_solutions():
    result = solve_triangle({"a": 8, "b": 10, "A": 40})
    assert result.error is None
    assert result.case == "SSA"
    assert len(result.solutions) == 2
    for sol in result.solutions:
        assert sol.verified
        assert abs(sol.a - 8) < 1e-6
        assert abs(sol.b - 10) < 1e-6
        assert abs(sol.A - 40) < 1e-6
    # the two solutions must actually be different triangles
    assert abs(result.solutions[0].B - result.solutions[1].B) > 1


def test_ssa_no_solution_side_too_short():
    result = solve_triangle({"a": 2, "b": 10, "A": 40})
    assert result.error is not None
    assert len(result.solutions) == 0


def test_ssa_single_solution_right_angle_case():
    """When b*sin(A)/a is exactly 1, there's exactly one (right-angle)
    triangle -- the two asin() branches coincide."""
    a_val = 10.0
    A_val = 40.0
    b_val = a_val / math.sin(math.radians(A_val))  # makes sin(B) exactly 1 -> B = 90
    result = solve_triangle({"a": a_val, "b": b_val, "A": A_val})
    assert result.error is None
    assert len(result.solutions) == 1
    assert abs(result.solutions[0].B - 90.0) < 0.01


# --------------------------------------------------------------------- input validation
def test_wrong_number_of_knowns_rejected():
    result = solve_triangle({"a": 5, "b": 6})
    assert result.error is not None


def test_no_side_known_rejected():
    result = solve_triangle({"A": 40, "B": 60, "C": 80})
    assert result.error is not None
    assert "shape but not its size" in result.error


def test_negative_side_rejected():
    result = solve_triangle({"a": -5, "b": 6, "c": 7})
    assert result.error is not None


def test_invalid_angle_rejected():
    result = solve_triangle({"a": 5, "b": 6, "C": 200})
    assert result.error is not None


# --------------------------------------------------------------------- rendering
def test_render_produces_valid_figure_with_correct_arc_geometry():
    """Since a rendered chart can't be visually inspected in tests, this
    verifies the actual GEOMETRY of what was drawn: each angle arc's
    true angular span (measured from its own plotted coordinates, from
    the vertex) must match the solved angle value -- not just that
    SOME arc was drawn near each vertex."""
    result = solve_triangle({"a": 5, "b": 6, "c": 7})
    sol = result.solutions[0]
    fig = render_triangle(sol)
    fig_dict = fig.to_dict()
    assert len(fig_dict["data"]) == 4  # triangle outline + 3 angle arcs
    assert len(fig_dict["layout"]["annotations"]) == 9  # 3 vertex + 3 side + 3 angle labels

    for i, vertex_letter in enumerate(["A", "B", "C"]):
        arc = fig_dict["data"][i + 1]
        vx, vy = sol.vertices[vertex_letter]
        a1 = math.atan2(arc["y"][0] - vy, arc["x"][0] - vx)
        a2 = math.atan2(arc["y"][-1] - vy, arc["x"][-1] - vx)
        span = abs(math.degrees(a2 - a1))
        span = min(span, 360 - span)
        expected = getattr(sol, vertex_letter)
        assert abs(span - expected) < 0.5


# --------------------------------------------------------------------- SSA ambiguity animation
def _fake_solution(a, b, c, A, B, C):
    return TriangleSolution(a=a, b=b, c=c, A=A, B=B, C=C, vertices={}, angle_sum_check=180.0,
                              law_of_cosines_residual=0.0, verified=True)


def test_ssa_ambiguity_animation_builds_for_a_real_ssa_pair():
    result = solve_triangle({"a": 8, "b": 10, "A": 40})
    assert len(result.solutions) == 2
    fig = build_ssa_ambiguity_animation(*result.solutions)
    assert fig is not None
    assert len(fig.frames) == 30
    assert "swinging between two valid positions" in fig.layout.title.text


def test_ssa_ambiguity_animation_keeps_pivot_and_neighbor_fixed_across_frames():
    """The whole point of the shared-frame reconstruction: the pivot (angle
    A's vertex) and the neighbor (far end of the given adjacent side) must
    be the SAME two points in every frame -- only the swinging vertex moves."""
    result = solve_triangle({"a": 8, "b": 10, "A": 40})
    fig = build_ssa_ambiguity_animation(*result.solutions)
    first_frame_verts = fig.frames[0].data[1]  # the "vertices" trace: [pivot, neighbor, swinging]
    last_frame_verts = fig.frames[-1].data[1]
    assert first_frame_verts.x[0] == last_frame_verts.x[0]  # pivot x unchanged
    assert first_frame_verts.y[0] == last_frame_verts.y[0]  # pivot y unchanged
    assert first_frame_verts.x[1] == last_frame_verts.x[1]  # neighbor x unchanged
    assert first_frame_verts.y[1] == last_frame_verts.y[1]  # neighbor y unchanged
    # the swinging vertex (index 2) DOES move between the first and last frame
    assert (first_frame_verts.x[2], first_frame_verts.y[2]) != (last_frame_verts.x[2], last_frame_verts.y[2])


def test_ssa_ambiguity_animation_endpoints_match_the_two_solutions():
    """The swinging vertex's position at frame 0 and frame -1 should land
    exactly on solution 1's and solution 2's own triangle -- i.e. the
    reconstructed frame's distances match each solution's actual side
    lengths, not some approximation."""
    result = solve_triangle({"a": 8, "b": 10, "A": 40})
    sol1, sol2 = result.solutions
    fig = build_ssa_ambiguity_animation(sol1, sol2)
    pivot_xy = (fig.frames[0].data[1].x[0], fig.frames[0].data[1].y[0])
    neighbor_xy = (fig.frames[0].data[1].x[1], fig.frames[0].data[1].y[1])
    start_xy = (fig.frames[0].data[1].x[2], fig.frames[0].data[1].y[2])
    end_xy = (fig.frames[-1].data[1].x[2], fig.frames[-1].data[1].y[2])

    def dist(p, q):
        return math.hypot(p[0] - q[0], p[1] - q[1])

    # pivot-to-swinging distance is side "c" for both (A and B are pivot/swinging here)
    assert abs(dist(pivot_xy, start_xy) - sol1.c) < 1e-6
    assert abs(dist(pivot_xy, end_xy) - sol2.c) < 1e-6
    # neighbor-to-swinging distance is the given (shared) side "a" for BOTH solutions
    assert abs(dist(neighbor_xy, start_xy) - sol1.a) < 1e-6
    assert abs(dist(neighbor_xy, end_xy) - sol2.a) < 1e-6


def test_ssa_ambiguity_animation_returns_none_for_a_non_ssa_pair():
    """Two triangles that don't share two sides and an angle (not a genuine
    SSA pair) -- the shared-frame reconstruction can't find a pivot, and
    the function returns None rather than building a misleading animation."""
    sol1 = _fake_solution(1, 2, 3, 10, 20, 150)
    sol2 = _fake_solution(9, 8, 7, 99, 88, 77)  # nothing in common with sol1
    assert build_ssa_ambiguity_animation(sol1, sol2) is None


def test_ssa_ambiguity_animation_returns_none_for_identical_solutions():
    """Not genuinely ambiguous (both "solutions" are the same triangle) --
    every side and angle matches, so there's no non-shared side/angle to
    identify a pivot from, and it should decline rather than draw a
    a zero-length morph as if it meant something."""
    sol = _fake_solution(8, 10, 12.42304969336998, 40, 53.46414901438847, 86.53585098561153)
    assert build_ssa_ambiguity_animation(sol, sol) is None
