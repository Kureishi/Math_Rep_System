from modules.equation_engine import build_model
from modules.plausibility import check_plausibility
from modules.verifier import verify


def _kinematics_model(accel_unit="m/s^2"):
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": accel_unit},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def test_absurd_acceleration_flagged():
    model = _kinematics_model()
    notes = check_plausibility(model, {"a": 500})
    assert len(notes) == 1
    assert notes[0].symbol == "a"
    assert notes[0].category == "kinematics"
    assert "typical range" in notes[0].message


def test_normal_acceleration_not_flagged():
    model = _kinematics_model()
    notes = check_plausibility(model, {"a": 2.0})
    assert notes == []


def test_negative_mass_flagged_without_declared_domain():
    model = build_model({
        "problem_domain": "chemistry", "problem_type": "algebraic",
        "variables": [
            {"symbol": "m", "meaning": "measured mass", "known_value": None, "unit": "kg"},
        ],
        "equations": [], "solve_for": ["m"], "assumptions": [],
    })
    notes = check_plausibility(model, {"m": -5.0})
    assert len(notes) == 1
    assert "negative" in notes[0].message


def test_negative_mass_not_flagged_when_domain_already_declared():
    """physical_validity.py already owns a variable with a declared
    domain restriction -- this module should stay out of the way."""
    model = build_model({
        "problem_domain": "chemistry", "problem_type": "algebraic",
        "variables": [
            {"symbol": "m", "meaning": "measured mass", "known_value": None, "unit": "kg",
             "domain": "positive"},
        ],
        "equations": [], "solve_for": ["m"], "assumptions": [],
    })
    notes = check_plausibility(model, {"m": -5.0})
    assert notes == []


def test_unrecognized_unit_and_meaning_produce_no_notes():
    model = build_model({
        "problem_domain": "abstract algebra", "problem_type": "algebraic",
        "variables": [{"symbol": "x", "meaning": "widget count factor", "known_value": None, "unit": "gronk"}],
        "equations": [], "solve_for": ["x"], "assumptions": [],
    })
    notes = check_plausibility(model, {"x": 1e9})
    assert notes == []


def test_temperature_below_absolute_zero_flagged():
    model = build_model({
        "problem_domain": "thermodynamics", "problem_type": "algebraic",
        "variables": [{"symbol": "T", "meaning": "final temperature", "known_value": None, "unit": "K"}],
        "equations": [], "solve_for": ["T"], "assumptions": [],
    })
    notes = check_plausibility(model, {"T": -10})
    assert len(notes) == 1
    assert notes[0].category == "thermodynamics"


def test_domain_inferred_from_free_text_label():
    model = build_model({
        "problem_domain": "compound interest", "problem_type": "algebraic",
        "variables": [{"symbol": "r", "meaning": "growth rate", "known_value": None, "unit": "%"}],
        "equations": [], "solve_for": ["r"], "assumptions": [],
    })
    notes = check_plausibility(model, {"r": 250})
    assert len(notes) == 1
    assert notes[0].category == "finance"


def test_unknown_domain_falls_back_to_general_with_no_ranges():
    model = _kinematics_model(accel_unit="m/s^2")
    model.problem_domain = "underwater basket weaving"
    notes = check_plausibility(model, {"a": 999999})
    # no kinematics-range check applies once the domain string doesn't
    # match any known category -- and "acceleration" isn't a recognized
    # non-negativity meaning keyword, so nothing else fires either
    assert notes == []


def test_symbol_missing_from_model_variables_still_handled():
    model = _kinematics_model()
    # a stray symbol not present in model.variables at all shouldn't crash --
    # it just has no unit/meaning to look anything up by
    notes = check_plausibility(model, {"z": 42})
    assert notes == []


def test_wired_into_verify_report_as_advisory_only(fake_client_factory):
    # (v_f - v_i) / t = (2018 - 8) / 6 -- an absurd ~335 m/s^2, so SymPy's
    # own solve (not just the independent-check number) actually lands
    # out of range
    model = _kinematics_model()
    for v in model.variables:
        if v.symbol == "v_f":
            v.known_value = 2018.0
    client = fake_client_factory(final_answers={"a": 335.0})
    report = verify(model, client, "some absurd car problem")
    assert any(n.symbol == "a" for n in report.plausibility_notes)
    # advisory only -- a wildly-out-of-range magnitude alone must never
    # flip the overall pass/fail verdict the way a domain violation would
    assert report.passed is True
