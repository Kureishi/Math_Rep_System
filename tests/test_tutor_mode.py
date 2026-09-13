"""Tests for modules/tutor_mode.py -- final-answer guess grading."""
from modules.tutor_mode import check_final_answer_guess
from modules.verifier import VerificationReport


def test_exact_correct_guess():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("6", "F", report)
    assert result.correct
    assert result.relative_error == 0.0


def test_guess_within_tolerance_is_correct():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("6.05", "F", report, tolerance=0.02)
    assert result.correct


def test_guess_outside_tolerance_is_incorrect():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("10", "F", report)
    assert not result.correct
    assert result.relative_error > 0.5


def test_simple_expression_guess_is_evaluated():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("2*3", "F", report)
    assert result.correct


def test_unparseable_guess_reports_error_not_crash():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("not a number", "F", report)
    assert result.error is not None
    assert not result.correct


def test_unknown_target_reports_error():
    report = VerificationReport(passed=True, sympy_numeric_answers={"F": 6.0})
    result = check_final_answer_guess("5", "G", report)
    assert result.error is not None


def test_relative_tolerance_scales_with_magnitude():
    """The same 2% relative tolerance should accept a proportionally
    larger absolute error for a much larger answer -- confirms this
    isn't secretly using an absolute tolerance."""
    report_small = VerificationReport(passed=True, sympy_numeric_answers={"x": 3.0})
    report_large = VerificationReport(passed=True, sympy_numeric_answers={"x": 3_000_000.0})
    # 0.05 absolute off is fine for the large answer (tiny relative error)...
    assert check_final_answer_guess("3000000.05", "x", report_large).correct
    # ...but would fail for the small one at the same absolute offset if tolerance were absolute
    assert not check_final_answer_guess("3.5", "x", report_small).correct
