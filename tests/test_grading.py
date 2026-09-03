import pytest
import sympy as sp

from modules.equation_engine import build_model
from modules.grading import grade_work, classify_mistake


def _kinematics_model():
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def test_correct_work_passes_all_checks():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "a = (20 - 8) / 6", "a = 2.0"], 2.0)
    assert result.formula_ok is True
    assert result.final_answer_ok is True
    assert "Correct" in result.summary


def test_rearranged_but_correct_formula_recognized():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a*t = v_f - v_i", "a = 12/6", "a = 2"], 2.0)
    assert result.formula_ok is True
    assert result.final_answer_ok is True


def test_wrong_formula_flagged_as_conceptual_error():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = v_f / t", "a = 20/6", "a = 3.33"], 2.0)
    assert result.formula_ok is False
    assert result.final_answer_ok is False
    assert "conceptual" in result.summary.lower()


def test_within_line_arithmetic_error_detected_and_pinpointed():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "12/6 = 3", "a = 3"], 2.0)
    assert result.formula_ok is True
    assert result.final_answer_ok is False
    assert "arithmetic" in result.summary.lower()
    assert "line 2" in result.summary.lower()
    bad_line = result.line_results[1]
    assert bad_line.arithmetic_ok is False


def test_correct_bare_target_definition_lines_are_marked_ok():
    """Lines like 'a = (20-8)/6' where the LHS is just the (still
    unknown) target symbol shouldn't be marked 'not checkable' just
    because the target itself isn't a known value -- regression test
    for a real bug found during development."""
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "a = (20 - 8) / 6", "a = 2.0"], 2.0)
    for lr in result.line_results:
        assert lr.arithmetic_ok is True


def test_empty_work_returns_error():
    model = _kinematics_model()
    result = grade_work(model, "a", ["", "   "], 2.0)
    assert result.error is not None


def test_unparseable_line_marked_unparsed_not_crash():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "this is not math @#$", "a = 2.0"], 2.0)
    kinds = [lr.kind for lr in result.line_results]
    assert "unparsed" in kinds


# ---------------------------------------------------------------- classify_mistake

def test_classify_correct_submission():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "a = (20 - 8) / 6", "a = 2.0"], 2.0)
    c = classify_mistake(result)
    assert c.category == "correct"
    assert c.subtype is None


def test_classify_wrong_formula():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = v_f / t", "a = 20/6", "a = 3.33"], 2.0)
    c = classify_mistake(result)
    assert c.category == "formula"
    assert c.subtype is None


def test_classify_arithmetic_error_with_sign_subtype():
    model = _kinematics_model()
    # a stray line re-asserting v_i with the wrong sign -- same
    # magnitude (8) but opposite sign -- should be recognized as a sign
    # error specifically, distinct from a generic arithmetic slip
    result = grade_work(
        model, "a", ["a = (v_f - v_i) / t", "v_i = -8", "a = -2.0"], 2.0)
    c = classify_mistake(result)
    assert c.category == "arithmetic"
    assert c.subtype == "sign_error"


def test_classify_arithmetic_error_falls_back_to_operator_guess():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "12/6 = 5", "a = 5"], 2.0)
    c = classify_mistake(result)
    assert c.category == "arithmetic"
    assert c.subtype == "division"


def test_classify_empty_work_error_is_unverified():
    model = _kinematics_model()
    result = grade_work(model, "a", ["", "   "], 2.0)
    c = classify_mistake(result)
    assert c.category == "unverified"


def test_line_with_undeclared_symbol_marked_not_checkable():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "q = 5", "a = 2.0"], 2.0)
    q_line = next(lr for lr in result.line_results if "q = 5" in lr.raw)
    assert q_line.arithmetic_ok is None


def test_final_answer_check_without_correct_value_is_undetermined():
    model = _kinematics_model()
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "a = 2.0"], None)
    assert result.final_answer_ok is None


def test_bare_expression_final_line_checked_correctly():
    model = _kinematics_model()
    # last line is a bare number, not "a = ..."
    result = grade_work(model, "a", ["a = (v_f - v_i) / t", "2.0"], 2.0)
    assert result.student_final_value == pytest.approx(2.0)
    assert result.final_answer_ok is True


def test_grading_unsupported_for_non_algebraic_target():
    model = build_model({
        "problem_domain": "growth", "problem_type": "ode", "independent_variable": "t",
        "variables": [
            {"symbol": "k", "meaning": "rate", "known_value": "0.03", "unit": "1/yr"},
            {"symbol": "P", "meaning": "population", "known_value": None, "unit": "people", "is_function": True},
        ],
        "equations": [
            {"name": "growth", "kind": "ode", "expression": "Eq(Derivative(P(t), t), k*P(t))", "derivation": ""},
        ],
        "initial_conditions": [{"expression": "P(0)", "value": "500", "note": ""}],
        "solve_for": ["P"], "assumptions": [],
    })
    result = grade_work(model, "P", ["P = 500*exp(0.03*t)"], None)
    assert result.error is not None
