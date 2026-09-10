"""Tests for modules/research_journal.py -- assembling a chosen set of
already-loaded history entries into one Markdown document."""
from modules.equation_engine import build_model
from modules.verifier import VerificationReport, CheckResult
from modules.research_journal import JournalEntry, build_journal_entry, generate_journal_markdown


def _model(equations, variables, domain="mechanics"):
    raw = {"problem_domain": domain, "variables": variables, "equations": equations,
           "solve_for": [], "assumptions": []}
    return build_model(raw)


def _newton_entry(entry_id=1):
    model = _model(
        [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "Newton 2nd law"}],
        [{"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"}],
    )
    report = VerificationReport(passed=True, checks=[CheckResult("Symbolic check", True, "ok")],
                                  sympy_numeric_answers={"F": 6.0})
    scenarios = [{"scenario": "A rocket accelerating in space", "mapping": "F=thrust"}]
    return JournalEntry(entry_id=entry_id, problem_text="A 2kg block accelerates at 3 m/s^2. Find the force.",
                          model=model, report=report, steps_by_target={}, scenarios=scenarios,
                          timestamp="2026-01-01T12:00:00")


def test_build_journal_entry_wraps_history_load_tuple():
    model = _model(
        [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "d"}],
        [{"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"}],
    )
    report = VerificationReport(passed=True)
    loaded = ("problem text", model, report, {}, [])
    entry = build_journal_entry(5, loaded, timestamp="2026-01-01")
    assert entry.entry_id == 5
    assert entry.problem_text == "problem text"
    assert entry.model is model
    assert entry.timestamp == "2026-01-01"


def test_journal_includes_concept_citation():
    md = generate_journal_markdown([_newton_entry()])
    assert "Newton's second law" in md


def test_journal_includes_equation_latex():
    md = generate_journal_markdown([_newton_entry()])
    assert "$$" in md
    assert "F = a m" in md or "F=am" in md.replace(" ", "")


def test_journal_includes_results_and_scenarios():
    md = generate_journal_markdown([_newton_entry()])
    assert "F = 6" in md
    assert "rocket accelerating in space" in md


def test_journal_concepts_summary_aggregates_across_entries():
    entry1 = _newton_entry(entry_id=1)
    model2 = _model(
        [{"name": "eq2", "expression": "Eq(KE, m*a**2/2)", "derivation": "d"}],
        [{"symbol": "KE", "meaning": "kinetic energy", "known_value": None, "unit": "J"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "speed", "known_value": 3.0, "unit": "m/s"}],
    )
    entry2 = JournalEntry(entry_id=2, problem_text="KE problem", model=model2,
                            report=VerificationReport(passed=True, sympy_numeric_answers={"KE": 9.0}))
    md = generate_journal_markdown([entry1, entry2], title="Combined Journal")
    assert "Combined Journal" in md
    assert "Concepts covered in this journal" in md
    assert "Newton's second law" in md
    assert "Kinetic energy" in md
    # both problems' individual sections should be present, in the given order
    assert md.index("Problem 1:") < md.index("Problem 2:")


def test_journal_flags_failed_verification():
    model = _model(
        [{"name": "eq1", "expression": "Eq(F, m*a)", "derivation": "d"}],
        [{"symbol": "F", "meaning": "force", "known_value": None, "unit": "N"},
         {"symbol": "m", "meaning": "mass", "known_value": 2.0, "unit": "kg"},
         {"symbol": "a", "meaning": "acceleration", "known_value": 3.0, "unit": "m/s^2"}],
    )
    report = VerificationReport(passed=False,
                                  checks=[CheckResult("Dimensional check", False, "units don't balance")])
    entry = JournalEntry(entry_id=1, problem_text="bad problem", model=model, report=report)
    md = generate_journal_markdown([entry])
    assert "did not pass" in md
    assert "units don't balance" in md


def test_empty_entry_list_produces_placeholder_not_crash():
    md = generate_journal_markdown([])
    assert "No entries selected" in md
