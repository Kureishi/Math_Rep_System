from modules.equation_engine import build_model
from modules.extraction_diff import diff_extractions, diff_variables, diff_equations


def _kinematics_a():
    """Extraction A: symbols v_f/v_i/t/a."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _kinematics_b_same_shape_different_names():
    """Extraction B: same underlying problem, different symbol letters
    but matching meanings -- should diff as fully 'matched'."""
    return build_model({
        "problem_domain": "Kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "vf", "meaning": "Final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "vi", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "T", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
            {"symbol": "accel", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "e1", "kind": "equation", "expression": "Eq(accel, (vf - vi) / T)", "derivation": ""},
        ],
        "solve_for": ["accel"], "assumptions": [],
    })


def _kinematics_b_different_known_value():
    """Same variables/structure as A, but v_i was extracted as 5 instead
    of 8 -- should diff as 'changed', not 'matched'."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "5", "unit": "m/s"},
            {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _extra_variable_model():
    """Same as A, but with an extra 'wind_speed' variable that A doesn't have."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
            {"symbol": "w", "meaning": "wind speed", "known_value": "3", "unit": "m/s"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


def _different_equation_model():
    """Same variable meanings as A but a DIFFERENT equation shape
    (multiplication instead of subtraction-then-division)."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time elapsed", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "e", "kind": "equation", "expression": "Eq(a, v_f * v_i * t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })


# ---------------------------------------------------------------- variable matching

def test_variables_matched_by_meaning_despite_different_symbols():
    a = _kinematics_a()
    b = _kinematics_b_same_shape_different_names()
    entries = diff_variables(a, b)
    assert all(e.status == "matched" for e in entries)
    assert len(entries) == 4


def test_variable_with_different_known_value_reported_as_changed():
    a = _kinematics_a()
    b = _kinematics_b_different_known_value()
    entries = diff_variables(a, b)
    changed = [e for e in entries if e.status == "changed"]
    assert len(changed) == 1
    assert "known value" in changed[0].detail


def test_variable_present_only_in_one_extraction():
    a = _kinematics_a()
    b = _extra_variable_model()
    entries = diff_variables(a, b)
    only_in_b = [e for e in entries if e.status == "only_in_b"]
    assert len(only_in_b) == 1
    assert only_in_b[0].symbol_b == "w"


def test_variable_meaning_matching_is_case_and_whitespace_insensitive():
    a = _kinematics_a()
    b = _kinematics_b_same_shape_different_names()
    # "final velocity" vs "Final velocity" should still match
    entries = diff_variables(a, b)
    vf_entry = next(e for e in entries if e.symbol_a == "v_f")
    assert vf_entry.status == "matched"


# ---------------------------------------------------------------- equation matching

def test_equations_matched_despite_variable_renames():
    a = _kinematics_a()
    b = _kinematics_b_same_shape_different_names()
    entries = diff_equations(a, b)
    assert len(entries) == 1
    assert entries[0].status == "matched"


def test_different_equation_shape_reported_as_only_in_each():
    a = _kinematics_a()
    b = _different_equation_model()
    entries = diff_equations(a, b)
    statuses = {e.status for e in entries}
    assert statuses == {"only_in_a", "only_in_b"}


# ---------------------------------------------------------------- full diff_extractions

def test_full_diff_identical_shape_reports_high_similarity():
    a = _kinematics_a()
    b = _kinematics_b_same_shape_different_names()
    diff = diff_extractions(a, b)
    assert diff.equation_shape_similarity == 1.0
    assert diff.solve_for_matches
    assert all(e.status == "matched" for e in diff.variables)


def test_full_diff_domain_case_insensitive_match():
    a = _kinematics_a()  # "kinematics"
    b = _kinematics_b_same_shape_different_names()  # "Kinematics"
    diff = diff_extractions(a, b)
    assert diff.domain_matches


def test_full_diff_different_equations_reports_zero_similarity():
    a = _kinematics_a()
    b = _different_equation_model()
    diff = diff_extractions(a, b)
    assert diff.equation_shape_similarity == 0.0


def test_full_diff_solve_for_mismatch_detected():
    a = _kinematics_a()  # solve_for = ["a"] -> "acceleration"
    b = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time elapsed", "known_value": None, "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": "2", "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["t"], "assumptions": [],   # solving for TIME instead of acceleration
    })
    diff = diff_extractions(a, b)
    assert not diff.solve_for_matches


def test_identical_model_diffed_against_itself_is_fully_matched():
    a = _kinematics_a()
    diff = diff_extractions(a, a)
    assert diff.domain_matches
    assert diff.solve_for_matches
    assert diff.equation_shape_similarity == 1.0
    assert all(e.status == "matched" for e in diff.variables)
    assert all(e.status == "matched" for e in diff.equations)
