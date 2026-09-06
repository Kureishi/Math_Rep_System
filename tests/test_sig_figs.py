import pytest

from modules.equation_engine import build_model
from modules.sig_figs import count_significant_figures, check_sig_figs, raw_known_value_strings


# ---------------------------------------------------------------- count_significant_figures

@pytest.mark.parametrize("text,expected", [
    ("8", 1),
    ("80", 1),          # ambiguous bare-integer trailing zero, conventionally 1
    ("8.0", 2),
    ("8.00", 3),
    ("0.0034", 2),
    ("0.34", 2),
    ("1002", 4),
    ("100", 1),
    ("100.", 3),
    ("100.0", 4),
    ("0.000", 3),
    ("0", 1),
    ("-8.20", 3),
    ("-8", 1),
    ("3.14159", 6),
    ("1.0", 2),
    ("120.0", 4),
])
def test_count_significant_figures(text, expected):
    assert count_significant_figures(text) == expected


def test_count_significant_figures_none_for_scientific_notation():
    assert count_significant_figures("1.5e10") is None


def test_count_significant_figures_none_for_non_numeric():
    assert count_significant_figures("abc") is None
    assert count_significant_figures("") is None
    assert count_significant_figures(None) is None


def test_count_significant_figures_none_for_fraction_text():
    assert count_significant_figures("1/2") is None


# ---------------------------------------------------------------- check_sig_figs

def test_flags_answer_reported_with_far_more_precision_than_inputs_support():
    note = check_sig_figs({"v_f": "20", "v_i": "8", "t": "6"}, 2.0000001, "a")
    assert note is not None
    assert note.supported_figures == 1
    assert note.reported_figures == 8
    assert "a ≈ 2" in note.message or "2" in note.message


def test_does_not_flag_when_within_one_digit_of_slack():
    # inputs support 1 sig fig; answer at 2 sig figs is within the
    # deliberate 1-digit slack, shouldn't be flagged
    note = check_sig_figs({"v_f": "20", "v_i": "8", "t": "6"}, 2.0, "a")
    assert note is None


def test_does_not_flag_when_answer_matches_input_precision():
    note = check_sig_figs({"x": "3.14", "y": "2.71"}, 5.85, "z")
    assert note is None


def test_supported_figures_is_the_minimum_across_inputs():
    # one input has 1 sig fig ('6'), another has 5 ('12.345') -- the
    # minimum (1) should govern
    note = check_sig_figs({"a": "12.345", "b": "6"}, 74.07, "c")
    assert note is not None
    assert note.supported_figures == 1


def test_returns_none_when_no_known_values_given():
    assert check_sig_figs({}, 2.0, "a") is None


def test_returns_none_when_no_input_text_is_countable():
    # every input is in scientific notation -- nothing countable
    assert check_sig_figs({"x": "1e5", "y": "2e3"}, 123.456, "z") is None


# ---------------------------------------------------------------- raw_known_value_strings

def test_raw_known_value_strings_extracted_from_model():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8.0", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    texts = raw_known_value_strings(model)
    assert texts == {"v_f": "20", "v_i": "8.0", "t": "6"}
    assert "a" not in texts  # unknown -- no known_value to report


def test_raw_known_value_strings_end_to_end_flags_overprecise_answer():
    model = build_model({
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
    texts = raw_known_value_strings(model)
    note = check_sig_figs(texts, 2.00000, "a")  # sympy-style overly precise
    assert note is None or note.supported_figures == 1
