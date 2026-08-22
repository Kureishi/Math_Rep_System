import json

from modules.equation_engine import build_model, extract_model
from modules.verifier import verify, confidence_label


def test_kinematics_verifies_cleanly(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "car problem")
    assert report.passed
    assert report.sympy_numeric_answers["a"] == 2.0


def test_buggy_equation_fails_cross_check(kinematics_buggy_json, fake_client_factory):
    """The missing '*t' term makes the derived answer wrong (12 instead of
    2); the independent re-solve (which got the right answer) should catch
    the disagreement."""
    model = build_model(json.loads(kinematics_buggy_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "car problem")
    assert not report.passed
    cross_check = next(c for c in report.checks if "cross-check" in c.label)
    assert not cross_check.passed


def test_numeric_balance_detects_inconsistency(fake_client_factory):
    """Every symbol is known but the equation doesn't actually balance --
    should fail even without an independent cross-check being possible."""
    model = build_model({
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "5", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": "10", "unit": None},
        ],
        "equations": [{"name": "bad", "kind": "equation", "expression": "Eq(x, y)", "derivation": "x"}],
        "solve_for": [], "assumptions": [],
    })
    report = verify(model, fake_client_factory(), "x")
    balance_check = next(c for c in report.checks if "Numeric balance" in c.label)
    assert not balance_check.passed


def test_dimensional_check_catches_unit_mismatch(fake_client_factory):
    """distance = velocity is numerically checkable but dimensionally
    nonsensical -- this is the whole point of having a separate check."""
    model = build_model({
        "variables": [
            {"symbol": "d", "meaning": "distance", "known_value": None, "unit": "m"},
            {"symbol": "v", "meaning": "velocity", "known_value": "5", "unit": "m/s"},
        ],
        "equations": [{"name": "wrong dims", "kind": "equation", "expression": "Eq(d, v)", "derivation": "x"}],
        "solve_for": ["d"], "assumptions": [],
    })
    report = verify(model, fake_client_factory(), "x")
    dim_check = next(c for c in report.checks if "Dimensional consistency" in c.label)
    assert not dim_check.passed


def test_dimensional_check_passes_consistent_equation(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    report = verify(model, fake_client_factory(final_answers={"a": 2.0}), "x")
    dim_check = next(c for c in report.checks if "Dimensional consistency" in c.label)
    assert dim_check.passed


def test_unrecognized_unit_skips_gracefully_instead_of_crashing(fake_client_factory):
    model = build_model({
        "variables": [{"symbol": "x", "meaning": "widgets", "known_value": "5", "unit": "sprockets"}],
        "equations": [{"name": "trivial", "kind": "equation", "expression": "Eq(x, 5)", "derivation": "x"}],
        "solve_for": [], "assumptions": [],
    })
    report = verify(model, fake_client_factory(), "x")
    assert report.passed  # unresolved unit should be skipped, not treated as a failure


def test_inequality_satisfied(inequality_json, fake_client_factory):
    model = build_model(json.loads(inequality_json))
    # substitute a known v that satisfies v <= limit
    for v in model.variables:
        if v.symbol == "v":
            v.known_value = 55.0
    report = verify(model, fake_client_factory(), "x")
    constraint_check = next(c for c in report.checks if "Constraint satisfied" in c.label)
    assert constraint_check.passed


def test_inequality_violated_fails(inequality_json, fake_client_factory):
    model = build_model(json.loads(inequality_json))
    for v in model.variables:
        if v.symbol == "v":
            v.known_value = 80.0  # exceeds the 65 mph limit
    report = verify(model, fake_client_factory(), "x")
    assert not report.passed
    constraint_check = next(c for c in report.checks if "Constraint satisfied" in c.label)
    assert not constraint_check.passed


def test_ode_solution_verified_symbolically(ode_json, fake_client_factory):
    model = build_model(json.loads(ode_json))
    report = verify(model, fake_client_factory(), "x")
    assert report.passed
    ode_check = next(c for c in report.checks if "ODE solution check" in c.label)
    assert ode_check.passed


def test_confidence_label_buckets():
    assert confidence_label(0.001) == "essentially exact"
    assert confidence_label(0.1) == "comfortable margin"
    assert confidence_label(0.5) == "adequate margin"
    assert confidence_label(0.95) == "borderline -- close to the tolerance limit"


def test_report_confidence_reflects_worst_check(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    # exact agreement -> high confidence
    exact_report = verify(model, fake_client_factory(final_answers={"a": 2.0}), "x")
    label, ratio = exact_report.confidence()
    assert label == "essentially exact"

    # 1.8% off against a 2% default tolerance -> should read as borderline
    borderline_report = verify(model, fake_client_factory(final_answers={"a": 2.036}), "x")
    label2, ratio2 = borderline_report.confidence()
    assert borderline_report.passed  # still within tolerance...
    assert label2 == "borderline -- close to the tolerance limit"  # ...but flagged as risky
    assert ratio2 > ratio


def test_two_target_system_solves_and_verifies_both(kinematics_two_target_json, fake_client_factory):
    model = build_model(json.loads(kinematics_two_target_json))
    client = fake_client_factory(final_answers={"a": 2.0, "d": 84.0})
    report = verify(model, client, "x")
    assert report.passed
    assert report.sympy_numeric_answers["a"] == 2.0
    assert report.sympy_numeric_answers["d"] == 84.0


def test_extract_model_end_to_end_with_workspace_context(fake_client_factory):
    """Confirms known_context actually reaches the prompt sent to the model
    -- this was a real bug (workspace values existed but were never wired
    into extraction calls) caught during manual testing."""
    client = fake_client_factory(payload_json=json.dumps({
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [{"symbol": "d", "meaning": "distance", "known_value": "84", "unit": "m"}],
        "equations": [], "solve_for": [], "assumptions": [],
    }))
    extract_model(client, "using d from the workspace, ...", known_context="- d = 84 m (previously solved)")
    assert len(client.calls) == 1
    _, user_msg = client.calls[0]
    assert "d = 84 m" in user_msg
