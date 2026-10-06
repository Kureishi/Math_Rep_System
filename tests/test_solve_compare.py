"""Comparing two solves (modules/solve_compare.py) and the charts drawn from the result."""
import json

import pytest

import modules.chains as chains_module
import modules.history as history
from modules.compare_plots import build_category_chart, build_relative_change_chart
from modules.equation_engine import build_model
from modules.solve_compare import (
    SolveSnapshot, chart_rows, compare, compare_answers, compare_equations, compare_variables,
    comparison_markdown, deterministic_report, snapshot_from_history, targets_in_common, what_if_snapshot,
)
from modules.solver import compute_steps
from modules.verifier import verify
from tests.conftest import FakeClient, KINEMATICS_TWO_TARGET_JSON


def raw(equation="Eq(v, u + a*t)", u="8", extra_var=None, name="vel"):
    variables = [
        {"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
        {"symbol": "u", "meaning": "initial velocity", "known_value": u, "unit": "m/s"},
        {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
        {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
    ] + ([extra_var] if extra_var else [])
    return {"problem_domain": "kinematics", "problem_type": "algebraic", "variables": variables,
            "equations": [{"name": name, "kind": "equation", "expression": equation, "derivation": "x"}],
            "solve_for": ["a"], "assumptions": []}


def snap(label="A", **kw) -> SolveSnapshot:
    model = build_model(raw(**kw))
    return SolveSnapshot(label, "problem", model, deterministic_report(model))


# ------------------------------------------------------------------ the deterministic report

def test_the_deterministic_report_solves_and_checks_without_any_model_call():
    s = snap()
    assert s.report.passed
    assert s.report.sympy_numeric_answers["a"] == pytest.approx(2.0)
    assert all(not c.label.startswith("Independent cross-check") for c in s.report.checks)


def test_a_failing_model_is_reported_as_failed_not_raised():
    model = build_model(raw(equation="Eq(v, u + a)"))        # missing *t: still solvable, but a different answer
    report = deterministic_report(model)
    assert report.sympy_numeric_answers["a"] == pytest.approx(12.0)
    bad = build_model(raw(equation="Eq(v, u + ("))
    r2 = deterministic_report(bad)
    assert r2.passed is False and r2.failure_reason


# ------------------------------------------------------------------ what-if snapshots

def test_a_what_if_changes_only_what_was_named_and_re_solves():
    a = snap()
    b = what_if_snapshot(a, {"t": 10.0})
    assert b.report.sympy_numeric_answers["a"] == pytest.approx(1.2)
    assert {v.symbol: v.known_value for v in b.model.variables if v.symbol != "a"} == \
           {"v": 20.0, "u": 8.0, "t": 10.0}
    assert b.deterministic_only is True and a.deterministic_only is False
    assert a.model.variables[2].known_value == 6.0                # the original is untouched


def test_a_what_if_cannot_turn_an_unknown_into_a_given():
    with pytest.raises(ValueError, match="known value"):
        what_if_snapshot(snap(), {"a": 3.0})
    with pytest.raises(ValueError, match="nope"):
        what_if_snapshot(snap(), {"nope": 1.0})


# ------------------------------------------------------------------ answers

def test_identical_solves_have_no_differences():
    c = compare(snap(), snap("B"))
    assert [d.status for d in c.answers] == ["same"]
    assert all(v.status == "same" for v in c.variables)
    assert [e.status for e in c.equations] == ["identical"]
    assert c.summary == ["All 1 answer(s) are the same."]


def test_a_changed_answer_reports_both_values_and_the_relative_difference():
    a = snap()
    d, = compare_answers(a, what_if_snapshot(a, {"t": 10.0}))
    assert (d.target, d.status) == ("a", "changed")
    assert d.a == pytest.approx(2.0) and d.b == pytest.approx(1.2)
    assert d.abs_diff == pytest.approx(0.8)
    assert d.rel_diff == pytest.approx(0.4)              # 0.8 / max(|2.0|, |1.2|)
    assert d.unit == "m/s^2"


def test_an_answer_only_one_side_has_is_flagged_as_such():
    a = snap()
    b = snap("B")
    b.report.sympy_numeric_answers = {}
    d, = compare_answers(a, b)
    assert d.status == "only_a" and d.b is None and d.abs_diff is None
    d2, = compare_answers(b, a)
    assert d2.status == "only_b"


def test_two_zero_answers_are_the_same_and_have_no_relative_difference():
    a, b = snap(), snap("B")
    a.report.sympy_numeric_answers = {"a": 0.0}
    b.report.sympy_numeric_answers = {"a": 0.0}
    d, = compare_answers(a, b)
    assert d.status == "same" and d.rel_diff is None


def test_numerically_negligible_differences_count_as_the_same_number():
    a, b = snap(), snap("B")
    a.report.sympy_numeric_answers = {"a": 2.0}
    b.report.sympy_numeric_answers = {"a": 2.0 + 1e-12}
    assert compare_answers(a, b)[0].status == "same"


# ------------------------------------------------------------------ variables

def test_variable_changes_are_classified():
    a = snap()
    b = snap("B", u="9", extra_var={"symbol": "w", "meaning": "extra", "known_value": "1", "unit": "s"})
    by = {v.symbol: v.status for v in compare_variables(a, b)}
    assert by == {"v": "same", "u": "changed", "t": "same", "a": "same", "w": "only_b"}


def test_a_variable_that_becomes_known_changed_role():
    a = snap()
    model = build_model({**raw(), "variables": [dict(v, known_value="2") if v["symbol"] == "a" else v
                                                 for v in raw()["variables"]]})
    b = SolveSnapshot("B", "p", model, deterministic_report(model))
    assert {v.symbol: v.status for v in compare_variables(a, b)}["a"] == "role_changed"


def test_a_unit_change_counts_as_a_change():
    a = snap()
    model = build_model({**raw(), "variables": [dict(v, unit="km/h") if v["symbol"] == "u" else v
                                                 for v in raw()["variables"]]})
    b = SolveSnapshot("B", "p", model, deterministic_report(model))
    assert {v.symbol: v.status for v in compare_variables(a, b)}["u"] == "changed"


# ------------------------------------------------------------------ equations

def test_the_same_text_is_identical_whatever_the_spacing_or_case():
    a, b = snap(), snap("B", equation="Eq( v ,  u + a*t )")
    assert compare_equations(a, b)[0].status == "identical"


def test_a_rearranged_equation_is_equivalent_not_changed():
    a, b = snap(), snap("B", equation="Eq(v - u, a*t)", name="rearranged")
    e, = compare_equations(a, b)
    assert e.status == "equivalent" and (e.a_name, e.b_name) == ("vel", "rearranged")


def test_a_sign_flipped_or_rescaled_equation_is_equivalent():
    a = snap()
    assert compare_equations(a, snap("B", equation="Eq(u + a*t, v)"))[0].status == "equivalent"
    assert compare_equations(a, snap("B", equation="Eq(2*v, 2*u + 2*a*t)"))[0].status == "equivalent"


def test_a_genuinely_different_equation_over_the_same_symbols_is_changed():
    e, = compare_equations(snap(), snap("B", equation="Eq(v, u + a)"))
    assert e.status == "changed"


def test_an_unrelated_equation_is_reported_on_each_side_separately():
    a = snap()
    b = snap("B", equation="Eq(w, 3)", extra_var={"symbol": "w", "meaning": "w", "known_value": None, "unit": None})
    statuses = sorted(e.status for e in compare_equations(a, b))
    assert statuses == ["only_a", "only_b"]


def test_every_equation_on_both_sides_is_accounted_for_exactly_once():
    a = SolveSnapshot("A", "p", build_model(json.loads(KINEMATICS_TWO_TARGET_JSON)),
                      deterministic_report(build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))))
    b = snap("B")
    out = compare_equations(a, b)
    a_names = [e.a_name for e in out if e.a_name]
    b_names = [e.b_name for e in out if e.b_name]
    assert sorted(a_names) == sorted(eq.name for eq in a.model.equations)
    assert sorted(b_names) == sorted(eq.name for eq in b.model.equations)


# ------------------------------------------------------------------ confidence and the summary

def test_verification_differences_show_up_per_category_and_per_check():
    good = snap()
    bad_model = build_model(raw(equation="Eq(v, u + a)"))          # adds a velocity to an acceleration
    bad = SolveSnapshot("B", "p", bad_model, deterministic_report(bad_model))
    c = compare(good, bad)
    assert c.a_passed is True and c.b_passed is False
    assert any(k.a_passed is not False and k.b_passed is False for k in c.checks)
    dimensional = next(k for k in c.categories if k.category == "Dimensional consistency")
    assert (dimensional.a_passed, dimensional.b_passed) == (1, 0)
    assert any("Verification: A passed, B failed" in line for line in c.summary)


def test_the_summary_lists_each_changed_answer_and_input():
    a = snap()
    c = compare(a, what_if_snapshot(a, {"t": 10.0}))
    text = " ".join(c.summary)
    assert "a: 2 → 1.2" in text and "40.00%" in text
    assert "Inputs that differ: t." in text


def test_a_what_if_carries_a_caveat_that_it_was_not_cross_checked():
    a = snap()
    assert compare(a, a).caveats == []
    c = compare(a, what_if_snapshot(a, {"t": 10.0}))
    assert len(c.caveats) == 1 and "deterministic" in c.caveats[0]


def test_targets_in_common():
    a, b = snap(), snap("B")
    assert list(targets_in_common(a, b)) == ["a"]


# ------------------------------------------------------------------ history snapshots

def test_a_saved_problem_loads_as_a_snapshot(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "h.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "c.db")
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    report = verify(model, FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 84.0}), "p")
    hid = history.save("A car accelerates\nsecond line", model, report, compute_steps(model), [])
    s = snapshot_from_history(hid)
    assert s is not None and s.label == "A car accelerates" and s.deterministic_only is False
    assert s.report.sympy_numeric_answers == report.sympy_numeric_answers
    assert snapshot_from_history(hid + 1) is None


# ------------------------------------------------------------------ export and charts

def test_the_markdown_has_each_section_and_the_numbers():
    a = snap()
    md = comparison_markdown(compare(a, what_if_snapshot(a, {"t": 10.0})))
    for heading in ("## Summary", "## Answers", "## Variables", "## Equations", "## Verification"):
        assert heading in md
    assert "| a | 2 m/s^2 | 1.2 m/s^2 |" in md
    assert "deterministic checks only" in md


def test_chart_rows_give_percent_change_relative_to_a_and_skip_zero_baselines():
    a = snap()
    c = compare(a, what_if_snapshot(a, {"t": 10.0}))
    rows = chart_rows(c)
    assert rows["relative_change_pct"] == [("a", pytest.approx(-40.0))]
    zero = snap(); zero.report.sympy_numeric_answers = {"a": 0.0}
    assert chart_rows(compare(zero, a))["relative_change_pct"] == []


def test_the_charts_build_with_the_right_data():
    a = snap()
    rows = chart_rows(compare(a, what_if_snapshot(a, {"t": 10.0})))
    fig = build_relative_change_chart(rows["relative_change_pct"])
    assert list(fig.data[0].y) == ["a"] and list(fig.data[0].x) == [pytest.approx(-40.0)]
    cat = build_category_chart(rows["category_pass_fraction"])
    assert [t.name for t in cat.data] == ["A", "B"]
    assert list(cat.data[0].x) == [c for c, _, _ in rows["category_pass_fraction"]]
