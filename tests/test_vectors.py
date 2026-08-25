import sympy as sp

from modules.equation_engine import build_model
from modules.vector_utils import (
    make_vector, dot, cross, magnitude, unit_vector, angle_between,
    angle_between_deg, distance, make_point, vector_summary,
)


# ---------------------------------------------------------------- pure vector_utils


def test_dot_product():
    u = make_vector([1, 2, 3])
    v = make_vector([4, 5, 6])
    assert dot(u, v) == 32


def test_cross_product_3d():
    u = make_vector([1, 0, 0])
    v = make_vector([0, 1, 0])
    result = cross(u, v)
    assert result == sp.Matrix([0, 0, 1])


def test_cross_product_2d_returns_scalar():
    u = make_vector([2, 0])
    v = make_vector([0, 3])
    assert cross(u, v) == 6  # counter-clockwise -> positive z-component


def test_magnitude():
    v = make_vector([3, 4])
    assert magnitude(v) == 5


def test_magnitude_of_scalar_falls_back_to_abs():
    assert magnitude(-7) == 7


def test_unit_vector():
    v = make_vector([3, 4])
    u = unit_vector(v)
    assert sp.simplify(u[0]) == sp.Rational(3, 5)
    assert sp.simplify(u[1]) == sp.Rational(4, 5)
    assert sp.simplify(magnitude(u)) == 1


def test_angle_between_perpendicular_vectors():
    u = make_vector([1, 0])
    v = make_vector([0, 1])
    assert angle_between(u, v) == sp.pi / 2
    assert angle_between_deg(u, v) == 90


def test_distance_between_vectors():
    a = make_vector([0, 0])
    b = make_vector([3, 4])
    assert distance(a, b) == 5


def test_distance_between_points():
    p1 = make_point(0, 0)
    p2 = make_point(3, 4)
    assert distance(p1, p2) == 5


def test_dot_dimension_mismatch_raises():
    u = make_vector([1, 2])
    v = make_vector([1, 2, 3])
    try:
        dot(u, v)
        assert False, "expected a ValueError for mismatched dimensions"
    except ValueError:
        pass


def test_vector_summary_with_all_knowns():
    subs = {sp.Symbol("Fx"): 3.0, sp.Symbol("Fy"): 4.0}
    summary = vector_summary("F", ["Fx", "Fy"], subs)
    assert summary["magnitude"] == 5.0
    assert summary["components"] == {"Fx": 3.0, "Fy": 4.0}


def test_vector_summary_missing_component_returns_none():
    subs = {sp.Symbol("Fx"): 3.0}
    assert vector_summary("F", ["Fx", "Fy"], subs) is None


def test_build_vector_plot_2d_and_3d():
    from modules.plotter import build_vector_plot
    fig2d = build_vector_plot([("F", [3.0, 4.0]), ("d", [5.0, 1.0])])
    assert len(fig2d.data) == 2
    fig3d = build_vector_plot([("F", [1.0, 2.0, 3.0])])
    assert len(fig3d.data) == 1


def test_build_vector_plot_rejects_bad_dimension():
    from modules.plotter import build_vector_plot
    try:
        build_vector_plot([("F", [1.0])])
        assert False, "expected a ValueError for a 1D vector"
    except ValueError:
        pass


def test_snapshot_vector_plot_2d_and_3d_produce_png_bytes():
    from modules.plot_snapshot import snapshot_vector_plot
    png2d = snapshot_vector_plot([("F", [3.0, 4.0]), ("d", [5.0, 1.0])])
    assert png2d.startswith(b"\x89PNG")
    png3d = snapshot_vector_plot([("F", [1.0, 2.0, 3.0])])
    assert png3d.startswith(b"\x89PNG")


# ---------------------------------------------------------------- wired into equation_engine


def _work_problem_payload():
    return {
        "problem_domain": "work-energy", "problem_type": "algebraic",
        "variables": [
            {"symbol": "Fx", "meaning": "force x-component", "known_value": "10", "unit": "N"},
            {"symbol": "Fy", "meaning": "force y-component", "known_value": "0", "unit": "N"},
            {"symbol": "dx", "meaning": "displacement x-component", "known_value": "5", "unit": "m"},
            {"symbol": "dy", "meaning": "displacement y-component", "known_value": "3", "unit": "m"},
            {"symbol": "F", "meaning": "applied force", "known_value": None, "unit": "N",
             "is_vector": True, "components": ["Fx", "Fy"]},
            {"symbol": "d", "meaning": "displacement", "known_value": None, "unit": "m",
             "is_vector": True, "components": ["dx", "dy"]},
            {"symbol": "W", "meaning": "work done", "known_value": None, "unit": "J"},
        ],
        "equations": [
            {"name": "work", "kind": "equation", "expression": "Eq(W, dot(F, d))",
             "derivation": "work is the dot product of force and displacement"},
        ],
        "solve_for": ["W"], "assumptions": [],
    }


def test_vector_variable_parses_and_binds_to_matrix():
    model = build_model(_work_problem_payload())
    f_var = next(v for v in model.variables if v.symbol == "F")
    assert f_var.is_vector
    assert f_var.components == ["Fx", "Fy"]


def test_dot_product_equation_reduces_to_scalar():
    model = build_model(_work_problem_payload())
    eq = model.equations[0]
    assert eq.parse_error is None
    # dot(F, d) should have collapsed to a plain scalar expression, no
    # leftover Matrix object in the parsed equation
    assert not eq.sympy_eq.rhs.has(sp.MatrixBase)
    assert eq.sympy_eq.rhs == sp.Symbol("Fx") * sp.Symbol("dx") + sp.Symbol("Fy") * sp.Symbol("dy")


def test_dot_product_equation_solves_correctly_end_to_end():
    model = build_model(_work_problem_payload())
    subs = {sp.Symbol(v.symbol): v.known_value for v in model.variables if v.known_value is not None}
    result = model.equations[0].sympy_eq.rhs.subs(subs)
    assert float(result) == 50.0  # 10*5 + 0*3


def test_cross_product_torque_equation():
    payload = {
        "problem_domain": "torque", "problem_type": "algebraic",
        "variables": [
            {"symbol": "rx", "meaning": "lever arm x", "known_value": "2", "unit": "m"},
            {"symbol": "ry", "meaning": "lever arm y", "known_value": "0", "unit": "m"},
            {"symbol": "Fx", "meaning": "force x", "known_value": "0", "unit": "N"},
            {"symbol": "Fy", "meaning": "force y", "known_value": "5", "unit": "N"},
            {"symbol": "r", "meaning": "lever arm", "known_value": None, "unit": "m",
             "is_vector": True, "components": ["rx", "ry"]},
            {"symbol": "F", "meaning": "force", "known_value": None, "unit": "N",
             "is_vector": True, "components": ["Fx", "Fy"]},
            {"symbol": "tau", "meaning": "torque", "known_value": None, "unit": "N*m"},
        ],
        "equations": [
            {"name": "torque", "kind": "equation", "expression": "Eq(tau, cross(r, F))",
             "derivation": "torque is r cross F"},
        ],
        "solve_for": ["tau"], "assumptions": [],
    }
    model = build_model(payload)
    assert model.equations[0].parse_error is None
    subs = {sp.Symbol(v.symbol): v.known_value for v in model.variables if v.known_value is not None}
    result = model.equations[0].sympy_eq.rhs.subs(subs)
    assert float(result) == 10.0  # 2*5 - 0*0
