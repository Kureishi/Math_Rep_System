import pytest

from modules.equation_engine import build_model
from modules.adversarial_testing import (
    generate_edge_cases, run_edge_case, run_adversarial_suite, EdgeCaseVariant,
)


def _kinematics_model():
    """a = (v_f - v_i) / t -- t in the denominator makes it a natural
    division-by-zero target for the 'zero' category."""
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


def _sqrt_model():
    """t = sqrt(d) -- d in a square root makes it a natural negative-input
    target for the 'negative' category (no real solution)."""
    return build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "d", "meaning": "distance", "known_value": "9", "unit": "m"},
            {"symbol": "t", "meaning": "time", "known_value": None, "unit": "s"},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(t, sqrt(d))", "derivation": ""}],
        "solve_for": ["t"], "assumptions": [],
    })


# ---------------------------------------------------------------- generate_edge_cases

def test_generates_four_categories_per_known_variable():
    model = _kinematics_model()
    variants = generate_edge_cases(model)
    known_symbols = {"v_f", "v_i", "t"}  # 'a' is unknown, shouldn't be perturbed
    symbols_seen = {v.symbol for v in variants}
    assert symbols_seen == known_symbols
    categories_per_symbol = {}
    for v in variants:
        categories_per_symbol.setdefault(v.symbol, set()).add(v.category)
    for sym in known_symbols:
        assert categories_per_symbol[sym] == {"zero", "negative", "tiny", "huge"}


def test_skips_variables_without_a_known_value():
    model = _kinematics_model()
    variants = generate_edge_cases(model)
    assert all(v.symbol != "a" for v in variants)


def test_zero_base_value_still_produces_valid_negative_and_scaled_variants():
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x0", "meaning": "initial position", "known_value": "0", "unit": "m"},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "e", "kind": "equation", "expression": "Eq(y, x0 + 1)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    variants = generate_edge_cases(model)
    neg = next(v for v in variants if v.symbol == "x0" and v.category == "negative")
    assert neg.value == -1.0  # fallback since abs(0) == 0
    tiny = next(v for v in variants if v.symbol == "x0" and v.category == "tiny")
    assert tiny.value != 0.0


# ---------------------------------------------------------------- run_edge_case: solved / unsolvable

def test_zero_denominator_handled_gracefully_not_solved():
    model = _kinematics_model()
    variant = EdgeCaseVariant("t = 0 (zero)", "t", 0.0, "zero")
    outcome = run_edge_case(model, variant, "a")
    assert outcome.status in ("unsolvable", "exception")  # division by zero must not silently "solve"
    assert outcome.status != "solved"


def test_normal_perturbation_still_solves():
    model = _kinematics_model()
    variant = EdgeCaseVariant("v_f = 25 (huge-ish but fine)", "v_f", 25.0, "huge")
    outcome = run_edge_case(model, variant, "a")
    assert outcome.status == "solved"
    assert outcome.value == pytest.approx((25 - 8) / 6)


def test_negative_input_to_sqrt_does_not_crash():
    """sqrt of a negative distance has no real solution -- this must be
    reported as 'unsolvable', never an unhandled exception bubbling out."""
    model = _sqrt_model()
    variant = EdgeCaseVariant("d = -9 (negative)", "d", -9.0, "negative")
    outcome = run_edge_case(model, variant, "t")
    assert outcome.status == "unsolvable"


def test_missing_symbol_reported_as_exception_not_raised():
    model = _kinematics_model()
    variant = EdgeCaseVariant("bogus", "not_a_real_symbol", 1.0, "zero")
    outcome = run_edge_case(model, variant, "a")
    # run_edge_case itself must never raise -- the missing symbol is
    # reported as part of the outcome instead
    assert outcome.status == "exception"
    assert "not_a_real_symbol" in outcome.detail


def test_run_edge_case_never_raises_even_for_nonsense_target():
    model = _kinematics_model()
    variant = EdgeCaseVariant("t = 0 (zero)", "t", 0.0, "zero")
    # 'not_a_target' isn't in model.solve_for at all
    outcome = run_edge_case(model, variant, "not_a_target")
    assert outcome.status in ("unsolvable", "exception")


# ---------------------------------------------------------------- run_adversarial_suite

def test_run_adversarial_suite_returns_one_outcome_per_variant():
    model = _kinematics_model()
    outcomes = run_adversarial_suite(model, "a")
    variants = generate_edge_cases(model)
    assert len(outcomes) == len(variants)
    assert all(o.status in ("solved", "unsolvable", "timeout", "exception") for o in outcomes)


def test_run_adversarial_suite_never_raises():
    """The suite runner itself must survive every variant it generates,
    including the deliberately nasty ones, without propagating any
    exception up to the caller."""
    model = _kinematics_model()
    outcomes = run_adversarial_suite(model, "a")  # should not raise
    assert len(outcomes) > 0


def test_timeout_status_is_a_recognized_outcome(monkeypatch):
    """Forces the timeout path deterministically by making the
    underlying solve step artificially slow, rather than relying on
    system timing -- verifies the timeout branch is wired correctly,
    not just theoretically reachable."""
    import time
    import modules.adversarial_testing as at_module
    monkeypatch.setattr(at_module, "EDGE_CASE_TIMEOUT_SECONDS", 0.05)

    def _slow_solve(sample_model, target):
        time.sleep(0.5)
        return {}

    monkeypatch.setattr(at_module, "_solve_and_check_plausibility", _slow_solve)
    model = _kinematics_model()
    variant = EdgeCaseVariant("v_f = 25", "v_f", 25.0, "huge")
    outcome = run_edge_case(model, variant, "a")
    assert outcome.status == "timeout"
