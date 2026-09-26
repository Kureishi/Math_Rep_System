"""
Tests for modules/motion_diagram.py -- the SUVAT-style kinematics detector
that feeds ui/results/summary.py's animated motion diagram. Uses
build_model() with hand-written payloads (not the LLM) so each detection
rule is exercised precisely, and a VerificationReport with
sympy_numeric_answers set directly to stand in for a solved target's
verified numeric answer.
"""
from modules.equation_engine import build_model
from modules.motion_diagram import build_kinematics_trajectory
from modules.verifier import VerificationReport


def _report(answers):
    r = VerificationReport()
    r.sympy_numeric_answers = answers
    return r


def _model(variables, solve_for=()):
    return build_model({
        "problem_domain": "physics", "problem_type": "algebraic",
        "variables": variables, "equations": [], "solve_for": list(solve_for), "assumptions": [],
    })


def test_full_suvat_set_builds_a_trajectory():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    ])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj is not None
    assert traj.t_values[0] == 0.0 and traj.t_values[-1] == 6.0
    assert traj.x_values[0] == 0.0
    # x(6) = 8*6 + 0.5*2*36 = 48 + 36 = 84
    assert abs(traj.x_values[-1] - 84.0) < 1e-6
    assert traj.v_values[0] == 8.0
    assert abs(traj.v_values[-1] - 20.0) < 1e-6  # v = 8 + 2*6


def test_position_unit_is_not_velocitys_unit():
    """Regression test for a real bug caught during development: the
    position axis must not inherit velocity's "m/s" unit -- it should be
    derived (here: "m/s" stripped of its "/s" against the time unit)."""
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    ])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj.x_unit == "m"
    assert traj.t_unit == "s"


def test_explicit_displacement_variable_unit_takes_priority():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
        {"symbol": "d", "meaning": "displacement", "known_value": None, "unit": "ft"},
    ])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj.x_unit == "ft"


def test_derives_initial_velocity_from_final_velocity_when_only_that_is_known():
    model = _model([
        {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    ])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj is not None
    # v_i = v_f - a*t = 20 - 12 = 8
    assert abs(traj.v_values[0] - 8.0) < 1e-6


def test_uses_a_solved_target_from_the_report_not_just_known_values():
    """Acceleration is the SOLVED target (known_value=None), resolved via
    report.sympy_numeric_answers rather than the variable's own
    known_value -- this is the whole reason this module doesn't just read
    render_variables()'s edited_values (which defaults an unsolved target
    to 0.0, not its actual solved answer)."""
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
    ], solve_for=["a"])
    traj = build_kinematics_trajectory(model, _report({"a": 2.0}))
    assert traj is not None
    assert abs(traj.x_values[-1] - 84.0) < 1e-6


def test_initial_position_offsets_the_whole_trajectory():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
        {"symbol": "x0", "meaning": "initial position", "known_value": "10", "unit": "m"},
    ])
    traj = build_kinematics_trajectory(model, _report({}))
    assert traj.x_values[0] == 10.0
    assert abs(traj.x_values[-1] - 94.0) < 1e-6  # 10 + 84


# ---------------------------------------------------------------- conservative "return None" cases

def test_missing_acceleration_returns_none():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    ])
    assert build_kinematics_trajectory(model, _report({})) is None


def test_missing_time_returns_none():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
    ])
    assert build_kinematics_trajectory(model, _report({})) is None


def test_no_velocity_at_all_returns_none():
    model = _model([
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
    ])
    assert build_kinematics_trajectory(model, _report({})) is None


def test_unresolved_acceleration_returns_none():
    """Acceleration is the intended target but nothing in the report solved
    it -- resolve() returns None, and the function bails rather than
    guessing or crashing."""
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
    ], solve_for=["a"])
    assert build_kinematics_trajectory(model, _report({})) is None


def test_zero_time_returns_none():
    model = _model([
        {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        {"symbol": "t", "meaning": "time elapsed", "known_value": "0", "unit": "s"},
    ])
    assert build_kinematics_trajectory(model, _report({})) is None


def test_non_kinematics_problem_returns_none():
    model = _model([
        {"symbol": "F", "meaning": "force", "known_value": "10", "unit": "N"},
        {"symbol": "m", "meaning": "mass", "known_value": "2", "unit": "kg"},
    ])
    assert build_kinematics_trajectory(model, _report({})) is None
