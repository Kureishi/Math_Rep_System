"""Report themes, automatic figure captions, the proof appendix with its replay manifest, and the answer card."""
import dataclasses
import io
import json

import plotly.graph_objects as go
import pytest

from modules import answer_card as ac
from modules import proof_card as pc
from modules import report_export as rx
from modules.report_content import PRESETS, SECTION_IDS, build_sections, figure_caption
from modules.report_theme import THEME_SECTIONS, THEMES, apply_theme
from modules.quantum_common import CheckList
from tests.report_helpers import kinematics_ctx, snapshot


def figure_json() -> str:
    fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 4, 9], name="y"))
    fig.update_layout(title="Parabola", xaxis_title="x", yaxis_title="y")
    return fig.to_json()


def ctx(**kw):
    return kinematics_ctx(snapshots=[snapshot("Parabola", "y against x", figure_json())], **kw)


def text_of(fmt, c, include=None):
    f = rx.render_export(fmt, c, include=include)
    return pc.text_of_report_file(f.name, f.data)


# ------------------------------------------------------------------------------------------ themes

def test_the_themes_and_their_presets_exist_and_agree():
    assert set(THEMES) == {"Lab report", "Homework solution", "One-page summary"} == set(THEME_SECTIONS)
    for name, sections in THEME_SECTIONS.items():
        assert PRESETS[name] == sections and set(sections) <= set(SECTION_IDS)


def test_a_theme_reorders_renames_and_titles():
    c = ctx()
    sections = build_sections(c.problem_text, c.model, c.report, c.steps_by_target, extras=dataclasses.replace(c.extras, auto_captions=True))
    themed = apply_theme(sections, THEMES["Lab report"])
    titles = [s.title for s in themed]
    assert titles[0] == "Objective" and titles.index("Theory") < titles.index("Results") and titles[-1].startswith("Appendix A")
    hw = apply_theme(build_sections(c.problem_text, c.model, c.report, c.steps_by_target, include=THEME_SECTIONS["Homework solution"]), THEMES["Homework solution"])
    assert [s.title for s in hw][:2] == ["Problem", "Given"] and "Answer" in [s.title for s in hw]


@pytest.mark.parametrize("fmt", rx.FORMAT_KEYS)
@pytest.mark.parametrize("theme", [None, *THEMES])
def test_every_format_renders_in_every_theme(fmt, theme):
    f = rx.render_export(fmt, dataclasses.replace(ctx(), theme=theme), include=THEME_SECTIONS.get(theme))
    assert len(f.data) > 500


def test_the_theme_title_replaces_the_report_title_in_each_format():
    c = dataclasses.replace(ctx(), theme="Homework solution")
    assert text_of("md", c).startswith("# Homework Solution")
    assert "Homework Solution" in text_of("pdf", c) and "Homework Solution" in text_of("docx", c) and "<title>Homework Solution</title>" in text_of("html", c)
    assert "Homework Solution" in text_of("tex", c)


def test_the_one_page_summary_is_one_page_and_short():
    from pypdf import PdfReader
    f = rx.render_export("pdf", dataclasses.replace(ctx(), theme="One-page summary"), include=THEME_SECTIONS["One-page summary"])
    assert len(PdfReader(io.BytesIO(f.data)).pages) == 1
    md = text_of("md", dataclasses.replace(ctx(), theme="One-page summary"), THEME_SECTIONS["One-page summary"])
    assert "BEGIN MRS REPLAY MANIFEST" not in md and "Proof id" in md and "## Proof reference" in md


def test_an_unknown_theme_is_refused():
    with pytest.raises(ValueError, match="Unknown report theme"):
        rx.render_export("md", dataclasses.replace(ctx(), theme="Poster"))


# ------------------------------------------------------------------------------------------ figure captions

def test_figure_captions_are_numbered_and_describe_the_chart():
    cap = figure_caption(2, "Parabola", "y against x", figure_json())
    assert cap.startswith("Figure 2. y against x.") and "x axis: x" in cap and "1 series: y" in cap
    assert figure_caption(1, "Plain", "", "") == "Figure 1."


def test_captions_follow_the_theme_and_the_override():
    lab = text_of("md", dataclasses.replace(ctx(), theme="Lab report"))
    assert "Figure 1. y against x." in lab
    plain = text_of("md", ctx())
    assert "Figure 1." not in plain
    forced = text_of("md", dataclasses.replace(ctx(), auto_captions=True))
    assert "Figure 1. y against x." in forced
    off = text_of("md", dataclasses.replace(ctx(), theme="Lab report", auto_captions=False))
    assert "Figure 1." not in off


def test_a_broken_figure_never_sinks_the_export():
    assert figure_caption(1, "x", "cap", "{not json") == "Figure 1. cap."


# ------------------------------------------------------------------------------------------ appendix and manifest

def test_the_appendix_holds_the_proof_card_and_the_manifest():
    c = ctx()
    sections = build_sections(c.problem_text, c.model, c.report, c.steps_by_target, include=["appendix"])
    kinds = [b.kind for b in sections[0].blocks]
    assert sections[0].id == "appendix" and "code" in kinds and "table" in kinds and "kv" in kinds


def test_the_manifest_round_trips_and_replays_from_every_format_that_carries_it():
    c = ctx()
    for fmt in ("md", "html", "tex", "pdf", "docx"):
        manifest = pc.extract_manifest(text_of(fmt, dataclasses.replace(c, theme="Lab report")))
        assert manifest is not None, fmt
        v = pc.verify_manifest(manifest)
        assert v.ok and v.headline.startswith("Re-verified"), fmt


def test_a_tex_zip_and_a_pptx_deal_with_the_manifest_sensibly():
    f = rx.render_export("tex", dataclasses.replace(ctx(), theme="Lab report"))
    assert pc.extract_manifest(pc.text_of_report_file(f.name, f.data)) is not None
    assert rx.render_export("pptx", dataclasses.replace(ctx(), theme="Lab report")).data[:2] == b"PK"


def test_an_edited_manifest_is_detected():
    c = dataclasses.replace(ctx(), theme="Lab report")
    manifest = pc.extract_manifest(text_of("md", c))
    manifest["content"]["problem_text"] = "A different problem"
    v = pc.verify_manifest(manifest)
    assert not v.ok and not v.integrity_ok and "edited" in v.integrity_detail


def test_a_manifest_whose_recorded_answers_no_longer_match_is_reported_as_differing():
    manifest = pc.build_manifest("p", ctx().model, ctx().report)
    manifest["content"]["verification"]["sympy_numeric_answers"] = {"a": 999.0}
    manifest["content_sha256"] = pc.content_hash(manifest["content"])
    v = pc.verify_manifest(manifest)
    assert v.integrity_ok and not v.ok and v.replay.status == "differs"


def test_garbage_is_not_a_manifest():
    assert pc.extract_manifest("nothing here") is None
    assert pc.extract_manifest(f"{pc.MANIFEST_BEGIN}\n!!!!\n{pc.MANIFEST_END}") is None
    assert pc.verify_manifest({"content": "x"}).headline == "Not a valid manifest"


def test_the_manifest_survives_a_reflow():
    enc = pc.encode_manifest(pc.build_manifest("p", ctx().model, ctx().report))
    head, *body, tail = enc.split("\n")
    reflowed = "\n".join([head, *(f"   > {line.replace('A', 'A ')}" for line in body), tail])      # quoted, indented and with stray spaces: still a manifest
    assert pc.extract_manifest(reflowed) is not None


def test_the_same_content_has_the_same_proof_id():
    a, b = pc.build_manifest("p", ctx().model, ctx().report), pc.build_manifest("p", ctx().model, ctx().report)
    assert a["content_sha256"] == b["content_sha256"] and pc.build_manifest("q", ctx().model, ctx().report)["content_sha256"] != a["content_sha256"]


# ------------------------------------------------------------------------------------------ answer card

def test_the_card_for_a_solved_problem_has_the_answers_with_units_and_the_proof_id():
    c = ctx()
    card = ac.from_solution(c.problem_text, c.model, c.report)
    assert card.answers[0].startswith("a = 2") and "m/s²" in card.answers[0] and card.level in ("verified", "caution")
    assert card.proof_id == pc.build_manifest(c.problem_text, c.model, c.report)["content_sha256"][:12]
    assert "language model" in card.footnote and card.proof_id in card.alt_text
    png = ac.render_png(card)
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    from PIL import Image
    assert Image.open(io.BytesIO(png)).size == (ac.WIDTH, ac.HEIGHT) and ac.render_png(card, dark=True) != png


def test_a_failing_result_is_not_dressed_up():
    ck = CheckList()
    ck.add("a", True, "fine")
    ck.add("b", False, "broken")
    card = ac.from_checks("Double slit", ["fringe spacing 3.3 mm"], ck, "λ = 550 nm")
    assert card.level == "failed" and card.verdict == "Not verified" and "1 of 2 checks passed" in card.verdict_detail and "✕" in card.caption
    assert "no language model" in card.footnote and ac.render_png(card)[:4] == b"\x89PNG"


def test_a_result_with_no_checks_is_not_called_verified():
    assert ac.from_checks("x", ["y"], CheckList()).level == "caution"


def test_a_long_problem_and_many_answers_still_render():
    ck = CheckList()
    ck.add("a", True, "fine")
    card = ac.from_checks("T", [f"answer {i} = {i * 1.2345678901:.10g} m/s²" for i in range(9)], ck, "word " * 200)
    assert len(card.answers) == ac.MAX_ANSWERS and ac.render_png(card)[:4] == b"\x89PNG"


def test_units_are_prettified():
    assert ac._pretty_unit("m/s**2") == " m/s²" and ac._pretty_unit("kg*m^-1") == " kg·m⁻¹" and ac._pretty_unit("unitless") == "" and ac._pretty_unit(None) == ""


def test_the_proof_id_on_a_lab_card_changes_with_the_result():
    a, b = CheckList(), CheckList()
    a.add("x", True, "")
    b.add("x", False, "")
    assert ac.from_checks("T", ["y"], a).proof_id != ac.from_checks("T", ["y"], b).proof_id
