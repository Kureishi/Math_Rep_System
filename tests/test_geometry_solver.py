"""Tests for modules/geometry_solver.py -- triangle solving (SSS, SAS,
ASA/AAS, SSA) and its labeled Plotly schematic."""
import math

from modules.geometry_solver import solve_triangle, render_triangle


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
