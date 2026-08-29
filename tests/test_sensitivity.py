import pytest
import sympy as sp

from modules.equation_engine import build_model
from modules.sensitivity import sweep_input, tornado_analysis
from modules.verifier import _known_substitutions


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


# ---------------------------------------------------------------- sweep_input


def test_sweep_input_produces_correct_values():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    result = sweep_input(model, "a", "t", knowns, pct_range=0.2, n_points=5)
    assert result is not None
    assert result.symbol == "t"
    assert result.nominal_input == pytest.approx(6.0)
    assert result.nominal_target == pytest.approx(2.0)
    assert len(result.values) == 5
    assert result.values[0] == pytest.approx(4.8)
    assert result.values[-1] == pytest.approx(7.2)
    # a = (20-8)/4.8 = 2.5
    assert result.target_values[0] == pytest.approx(2.5, rel=1e-4)


def test_sweep_input_handles_zero_nominal_with_fallback_window():
    model = build_model({
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "0", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(y, x + 5)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    knowns = _known_substitutions(model)
    result = sweep_input(model, "y", "x", knowns, n_points=5)
    assert result is not None
    assert result.values[0] == pytest.approx(-1.0)
    assert result.values[-1] == pytest.approx(1.0)


def test_sweep_input_none_for_unrelated_symbol():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    assert sweep_input(model, "a", "nonexistent", knowns) is None


def test_sweep_input_none_for_nonexistent_target():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    assert sweep_input(model, "nonexistent_target", "t", knowns) is None


def test_sweep_input_none_when_symbol_not_known():
    model = _kinematics_model()
    # sweep the TARGET itself, which isn't a known
    knowns = _known_substitutions(model)
    assert sweep_input(model, "a", "a", knowns) is None


# ---------------------------------------------------------------- tornado_analysis


def test_tornado_analysis_ranks_by_swing_descending():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "a", knowns, pct_range=0.2)
    assert len(entries) == 3
    swings = [e.swing for e in entries]
    assert swings == sorted(swings, reverse=True)
    # v_f should have the largest swing (coefficient 1/t on both v_f, v_i,
    # but the sweep range in absolute terms is larger for v_f since it's
    # larger in magnitude than v_i)
    assert entries[0].symbol == "v_f"


def test_tornado_analysis_entries_have_consistent_signs():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "a", knowns)
    for e in entries:
        assert e.swing == pytest.approx(abs(e.high_target - e.low_target))


def test_tornado_analysis_empty_for_nonexistent_target():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    assert tornado_analysis(model, "nonexistent", knowns) == []


def test_tornado_analysis_only_includes_known_symbols_the_target_depends_on():
    model = build_model({
        "problem_domain": "test", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": "10", "unit": None},
            {"symbol": "unused", "meaning": "unused", "known_value": "99", "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [{"name": "eq", "kind": "equation", "expression": "Eq(y, x * 2)", "derivation": ""}],
        "solve_for": ["y"], "assumptions": [],
    })
    knowns = _known_substitutions(model)
    entries = tornado_analysis(model, "y", knowns)
    assert [e.symbol for e in entries] == ["x"]


def test_tornado_analysis_pct_range_affects_swing_magnitude():
    model = _kinematics_model()
    knowns = _known_substitutions(model)
    narrow = tornado_analysis(model, "a", knowns, pct_range=0.05)
    wide = tornado_analysis(model, "a", knowns, pct_range=0.4)
    narrow_by_symbol = {e.symbol: e.swing for e in narrow}
    wide_by_symbol = {e.symbol: e.swing for e in wide}
    for sym in narrow_by_symbol:
        assert wide_by_symbol[sym] > narrow_by_symbol[sym]
