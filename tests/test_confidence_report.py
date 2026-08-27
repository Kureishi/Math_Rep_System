import pytest

from modules.verifier import VerificationReport, _infer_category


def test_infer_category_matches_known_labels():
    assert _infer_category("Equation parsing") == "Structural"
    assert _infer_category("Target variable present") == "Structural"
    assert _infer_category("Determinacy") == "Structural"
    assert _infer_category("Dimensional consistency") == "Dimensional consistency"
    assert _infer_category("Independent cross-check: a") == "Independent cross-check"
    assert _infer_category("Domain validity: eq1") == "Domain of validity"
    assert _infer_category("Linear system consistency") == "Matrix/system consistency"
    assert _infer_category("Inequality solved") == "Inequality"
    assert _infer_category("ODE solved correctly") == "Differential equations"
    assert _infer_category("Recurrence closed form") == "Recurrence relations"
    assert _infer_category("Critical point verified") == "Optimization"
    assert _infer_category("Something totally unrecognized") == "Other"


def test_confidence_report_all_passed_perfect_score():
    report = VerificationReport()
    report.add("Equation parsing", True, "ok", margin_ratio=0.0)
    report.add("Dimensional consistency", True, "ok", margin_ratio=0.0)
    cr = report.confidence_report()
    assert cr.passed is True
    assert cr.score == pytest.approx(1.0)
    assert cr.passed_count == 2
    assert cr.total_count == 2
    assert cr.critical_failures == []


def test_confidence_report_borderline_margin_lowers_score():
    report = VerificationReport()
    report.add("Equation parsing", True, "ok", margin_ratio=0.0)
    report.add("Independent cross-check: a", True, "close call", margin_ratio=0.9)
    cr = report.confidence_report()
    assert cr.passed is True
    assert cr.score == pytest.approx(0.1, abs=1e-9)  # 1 - worst_ratio
    assert cr.label == "borderline -- close to the tolerance limit"


def test_confidence_report_failure_caps_score_and_records_critical_failures():
    report = VerificationReport()
    report.add("Equation parsing", True, "ok", margin_ratio=0.0)
    report.add("Domain validity: eq1", False, "divide by zero")
    cr = report.confidence_report()
    assert cr.passed is False
    assert cr.score <= 0.45
    assert len(cr.critical_failures) == 1
    assert cr.critical_failures[0].label == "Domain validity: eq1"


def test_confidence_report_groups_by_category():
    report = VerificationReport()
    report.add("Equation parsing", True, "ok", margin_ratio=0.0)
    report.add("Target variable present", True, "ok", margin_ratio=0.0)
    report.add("Dimensional consistency", True, "ok", margin_ratio=0.0)
    report.add("Independent cross-check: a", True, "ok", margin_ratio=0.1)
    cr = report.confidence_report()
    assert cr.categories["Structural"].passed == 2
    assert cr.categories["Structural"].total == 2
    assert cr.categories["Structural"].all_passed is True
    assert cr.categories["Dimensional consistency"].total == 1
    assert cr.categories["Independent cross-check"].total == 1


def test_confidence_report_category_marks_not_all_passed():
    report = VerificationReport()
    report.add("Domain validity: eq1", True, "ok", margin_ratio=0.0)
    report.add("Domain validity: eq2", False, "bad")
    cr = report.confidence_report()
    cat = cr.categories["Domain of validity"]
    assert cat.passed == 1
    assert cat.total == 2
    assert cat.all_passed is False


def test_confidence_report_no_checks_ran():
    report = VerificationReport()
    cr = report.confidence_report()
    assert cr.total_count == 0
    assert cr.passed_count == 0
    assert cr.passed is True
    assert cr.score == pytest.approx(1.0)  # vacuously "essentially exact"
