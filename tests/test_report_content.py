"""The structured report content every export format renders from (modules/report_content.py)."""
import pytest

from modules.report_content import (
    DEFAULT_PRESET, PRESETS, SECTION_DESCRIPTIONS, SECTION_IDS, SECTION_TITLES, SECTIONS, ReportExtras,
    build_sections, mc_run_dict, unknown_sections,
)
from tests.report_helpers import full_extras, kinematics_ctx, linear_system_ctx, snapshot


def sections(ctx=None, extras=None, include=None, **kw):
    ctx = ctx or kinematics_ctx()
    return build_sections(ctx.problem_text, ctx.model, ctx.report, ctx.steps_by_target, ctx.scenarios,
                          kw.get("snapshots", ctx.snapshots), extras, include)


def ids(secs):
    return [s.id for s in secs]


# ------------------------------------------------------------------ the registry

def test_every_section_has_an_id_a_title_and_a_description_and_ids_are_unique():
    assert len(SECTION_IDS) == len(set(SECTION_IDS)) == len(SECTIONS)
    assert set(SECTION_TITLES) == set(SECTION_DESCRIPTIONS) == set(SECTION_IDS)
    assert all(SECTION_TITLES[i] and SECTION_DESCRIPTIONS[i] for i in SECTION_IDS)


def test_the_titles_the_markdown_has_always_used_are_unchanged():
    assert SECTION_TITLES["equations"] == "Derived equations"
    assert SECTION_TITLES["steps"] == "Step-by-step solution"
    assert SECTION_TITLES["scenarios"] == "Where else this applies"
    assert SECTION_TITLES["verification"] == "Verification detail"


def test_every_preset_names_only_real_sections_and_the_default_exists():
    for name, chosen in PRESETS.items():
        assert unknown_sections(chosen) == [], name
    assert DEFAULT_PRESET in PRESETS and set(PRESETS[DEFAULT_PRESET]) == set(SECTION_IDS)


def test_presets_differ_in_the_ways_their_names_promise():
    handout, audit, review, quick = (set(PRESETS[n]) for n in ("Student handout", "Full audit", "Peer review", "Quick summary"))
    assert "steps" in handout and "verification" not in handout and "provenance" not in handout
    assert {"verification", "domain", "provenance", "sensitivity"} <= review and "steps" not in review
    assert quick < audit and "steps" not in quick and "results" in quick


# ------------------------------------------------------------------ building

def test_sections_come_out_in_report_order_with_empty_ones_left_out():
    got = ids(sections())
    assert got == [s for s in SECTION_IDS if s in got]                  # report order
    for absent in ("matrix", "vectors", "plots", "scenarios", "followups", "tutor", "provenance"):
        assert absent not in got                                        # nothing to put there
    assert {"problem", "results", "equations", "variables", "steps", "verification"} <= set(got)


def test_include_keeps_only_the_named_sections():
    assert ids(sections(include=["steps", "problem"])) == ["problem", "steps"]


def test_an_included_section_with_no_content_is_still_left_out():
    assert ids(sections(include=["matrix", "plots"])) == []


def test_an_unknown_section_is_an_error_not_a_silent_omission():
    with pytest.raises(ValueError, match="nope"):
        sections(include=["problem", "nope"])


def test_a_linear_system_gets_a_matrix_section():
    assert "matrix" in ids(sections(linear_system_ctx()))


def test_results_table_has_each_answer_with_its_unit():
    results = next(s for s in sections() if s.id == "results")
    table = results.blocks[0]
    assert table.headers == ["Target", "Value", "Unit"]
    assert table.rows == [["a", "2", "m/s^2"], ["d", "84", "m"]]


def test_derived_equations_carry_latex_and_a_style_hint():
    eq = next(s for s in sections() if s.id == "equations")
    equations = [b for b in eq.blocks if b.kind == "equation"]
    assert equations and all(b.style == "plain" for b in equations)
    step_eqs = [b for b in next(s for s in sections() if s.id == "steps").blocks if b.kind == "equation"]
    assert step_eqs and all(b.style == "step" for b in step_eqs)


def test_an_equation_that_never_parsed_is_marked_raw_so_formats_do_not_treat_it_as_math():
    ctx = kinematics_ctx()
    ctx.model.equations[0].sympy_eq = None
    eq = next(s for s in sections(ctx) if s.id == "equations")
    assert [b.style for b in eq.blocks if b.kind == "equation"][0] == "raw"


def test_steps_are_grouped_by_target_with_numbered_labels():
    steps = next(s for s in sections() if s.id == "steps")
    subs = [b.text for b in steps.blocks if b.kind == "subheading"]
    assert subs == ["Solving for a", "Solving for d"]
    labels = [b.text for b in steps.blocks if b.kind == "label"]
    assert labels[0].startswith("Step 1: ") and len(labels) == 14


def test_the_confidence_section_lists_each_category():
    conf = next(s for s in sections() if s.id == "confidence")
    assert conf.blocks[0].style == "bold" and "Overall score:" in conf.blocks[0].text
    assert conf.blocks[1].kind == "status" and conf.blocks[1].rows


def test_plots_become_image_blocks_with_their_titles_captions_and_figures():
    ctx = kinematics_ctx(snapshots=[snapshot("T", "C", figure_json='{"data":[]}')])
    plots = next(s for s in sections(ctx) if s.id == "plots")
    b = plots.blocks[0]
    assert (b.kind, b.label_text, b.text, b.figure_json) == ("image", "T", "C", '{"data":[]}')


def test_scenarios_with_an_error_entry_are_dropped_as_before():
    ctx = kinematics_ctx(scenarios=[{"scenario": "x", "mapping": "y"}, {"error": "boom"}])
    assert "scenarios" not in ids(sections(ctx))
    ok = kinematics_ctx(scenarios=[{"scenario": "a pump", "mapping": "flow"}])
    sc = next(s for s in sections(ok) if s.id == "scenarios")
    assert sc.blocks[0].items == ["a pump -- flow"]


# ------------------------------------------------------------------ sections that need extras

def test_without_extras_the_session_only_sections_are_absent():
    assert not {"followups", "tutor", "provenance"} & set(ids(sections()))


def test_followups_are_listed_question_then_answer_and_an_error_says_so():
    ex = full_extras()
    ex.followups.append(type(ex.followups[0])("bad?", "error", "no model"))
    fu = next(s for s in sections(extras=ex) if s.id == "followups")
    texts = [b.text for b in fu.blocks]
    assert texts[:2] == ["Q: what if t doubles?", "a becomes 1"]
    assert "(could not be answered) no model" in texts


def test_tutor_guesses_show_how_they_were_marked_and_how_far_the_steps_were_revealed():
    t = next(s for s in sections(extras=full_extras()) if s.id == "tutor").blocks[0]
    assert t.rows == [["a", "2", "correct", "Correct!", "3/7"]]
    ex = full_extras()
    ex.tutor[0].correct = None
    assert next(s for s in sections(extras=ex) if s.id == "tutor").blocks[0].rows[0][2] == "not marked"


def test_provenance_comes_from_the_stored_snapshot_not_from_todays_settings():
    prov = next(s for s in sections(extras=full_extras()) if s.id == "provenance").blocks[0]
    rows = dict(prov.rows)
    assert "test-model" in rows["Language model"] and "9.9" in rows["Solved"]
    assert rows["Exported"].startswith("2026-02-02")


def test_a_problem_with_no_stored_provenance_says_so_instead_of_inventing_it():
    ex = full_extras()
    ex.provenance = None
    rows = dict(next(s for s in sections(extras=ex) if s.id == "provenance").blocks[0].rows)
    assert "no record" in rows["Solved"] and "CURRENT" in rows["Solved"]


def test_provenance_can_be_switched_off():
    ex = full_extras()
    ex.include_provenance = False
    assert "provenance" not in ids(sections(extras=ex))


def test_monte_carlo_runs_are_reported_with_seed_samples_and_inputs():
    sens = next(s for s in sections(extras=full_extras()) if s.id == "sensitivity")
    mc = [b for b in sens.blocks if b.kind == "table" and "Seed" in b.headers][0]
    assert mc.rows[0][:2] == ["a", "2"] and 42 in mc.rows[0] and "u ~ N(8, 0.5)" in mc.rows[0]
    repro = dict(next(s for s in sections(extras=full_extras()) if s.id == "provenance").blocks[0].rows)
    assert "seed 42" in repro["Monte Carlo (a)"]


def test_the_sensitivity_table_ranks_inputs_by_swing_largest_first():
    sens = next(s for s in sections() if s.id == "sensitivity")
    table = next(b for b in sens.blocks if b.kind == "table")
    swings = [float(r[-1]) for r in table.rows]
    assert swings == sorted(swings, reverse=True)


def test_live_figures_are_only_built_when_asked_for():
    plain = next(s for s in sections(extras=full_extras()) if s.id == "sensitivity")
    assert not any(b.kind == "figure" for b in plain.blocks)
    ex = full_extras()
    ex.live_figures = True
    live = next(s for s in sections(extras=ex) if s.id == "sensitivity")
    figs = [b for b in live.blocks if b.kind == "figure"]
    assert len(figs) == 3                                              # a tornado, d tornado, a's histogram
    assert all(b.figure_json.startswith("{") for b in figs)


def test_mc_run_dict_reads_a_result():
    from modules.monte_carlo import MonteCarloResult
    r = MonteCarloResult(target="a", samples=[1.0], n_requested=10, n_failed=1, seed=3, mean=1.0, std=0.1, p5=0.9,
                         p95=1.1, inputs=["u ~ N(1, 0.1)"])
    d = mc_run_dict(r, "m")
    assert (d["target"], d["n"], d["seed"], d["unit"], d["inputs"]) == ("a", 10, 3, "m", ["u ~ N(1, 0.1)"])
