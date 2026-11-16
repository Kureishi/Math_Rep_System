"""One failing part must not take the rest down: a missing optional package, a broken panel, a stage after the derivation, the housekeeping after a solve."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from modules import optional_deps, pipeline
from modules import report_export as rx
from modules.llm_client import LLMText, LMStudioClient
from tests.conftest import KINEMATICS_JSON, FakeClient
from tests.report_helpers import kinematics_ctx

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def without(monkeypatch):
    """Pretend the given top-level packages are not installed (importing them raises ImportError), the way they were on the machine that crashed."""
    def hide(*names):
        for n in names:
            for mod in [m for m in sys.modules if m == n or m.startswith(n + ".")]:
                monkeypatch.setitem(sys.modules, mod, None)
            monkeypatch.setitem(sys.modules, n, None)
        optional_deps.reset_cache()
    yield hide
    optional_deps.reset_cache()


# ------------------------------------------------------------------------------------------ the registry

def test_the_registry_knows_what_each_package_is_for():
    assert optional_deps.available("sympy") and not optional_deps.available("definitely_not_installed_xyz")
    gone = optional_deps.missing_for("definitely_not_installed_xyz")
    assert gone[0].dist == "definitely_not_installed_xyz"
    assert "pip install python-docx" in optional_deps.install_hint([d for d in optional_deps.DEPS if d.module == "docx"]) and "requirements.txt" in optional_deps.install_hint(gone)
    assert optional_deps.install_hint([]) == ""


def test_an_import_error_is_explained_with_the_package_that_fixes_it():
    msg = optional_deps.explain_import_error(ModuleNotFoundError("No module named 'docx'", name="docx"))
    assert "python-docx" in msg and "pip install python-docx" in msg and "Word" in msg
    assert "foo" in optional_deps.explain_import_error(ModuleNotFoundError("No module named 'foo'", name="foo.bar"))
    assert optional_deps.explain_import_error(ValueError("x")) is None


def test_every_registered_module_maps_to_a_requirements_line():
    reqs = {r[0].lower() for r in __import__("modules.health", fromlist=["x"]).parse_requirements(Path(__file__).parent.parent / "requirements.txt")}
    for d in optional_deps.DEPS:
        assert d.dist.lower() in reqs, d


# ------------------------------------------------------------------------------------------ the export formats

@pytest.mark.parametrize("package, fmt", [("docx", "docx"), ("pptx", "pptx"), ("fpdf", "pdf")])
def test_a_missing_report_package_disables_only_its_format(without, package, fmt):
    without(package)
    for mod in ("modules.report_office", "modules.exporter", "modules.report_export"):         # importing must still work
        __import__(mod)
    avail = rx.format_availability()
    assert avail[fmt][0] is False and f"pip install" in avail[fmt][1]
    assert all(ok for k, (ok, _) in avail.items() if k != fmt)
    ctx = kinematics_ctx()
    with pytest.raises(rx.ExportUnavailable, match="pip install"):
        rx.render_export(fmt, ctx)
    for other in (k for k in rx.FORMAT_KEYS if k != fmt):
        assert len(rx.render_export(other, ctx).data) > 100, other


def test_the_builders_themselves_raise_a_clear_import_error(without, monkeypatch):
    without("docx", "pptx", "fpdf")
    from modules import exporter, report_office
    for module, name in ((report_office, "Document"), (report_office, "Presentation"), (exporter, "FPDF")):     # forget what an earlier test already loaded
        monkeypatch.setattr(module, name, None)
    with pytest.raises(ImportError, match="python-docx"):
        report_office.render_docx([])
    with pytest.raises(ImportError, match="python-pptx"):
        report_office.render_pptx([])
    with pytest.raises(ImportError, match="fpdf2"):
        exporter.render_pdf([])


def test_all_formats_are_available_in_a_complete_install():
    assert all(ok for ok, _ in rx.format_availability().values())


# ------------------------------------------------------------------------------------------ pages

def _solved_app():
    ctx = kinematics_ctx()
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["model"], at.session_state["report"] = ctx.model, ctx.report
    at.session_state["steps"], at.session_state["problem_text"] = ctx.steps_by_target, ctx.problem_text
    return at, ctx


def test_the_results_still_show_when_python_docx_is_not_installed(without):
    """The reported failure: ModuleNotFoundError 'docx' replaced the whole results view."""
    without("docx")
    at, _ = _solved_app()
    at.run()
    assert not at.exception, [str(e.value)[:200] for e in at.exception]
    assert any("Domain" in h.value for h in at.subheader) and any("Share" in b.label for b in at.button)
    assert not [w for w in at.warning if "could not be shown" in w.value]


def test_one_failing_panel_shows_a_notice_and_the_rest_still_render(monkeypatch):
    import ui.results as results

    def boom(model):
        raise ValueError("matrix exploded")
    monkeypatch.setattr(results, "render_matrix_view", boom)
    at, _ = _solved_app()
    at.run()
    assert not at.exception
    notices = [w.value for w in at.warning if "Matrix view" in w.value]
    assert notices and "matrix exploded" in notices[0] and "rest of this page is not affected" in notices[0]
    assert any("Domain" in h.value for h in at.subheader)            # panels after the broken one still rendered


def test_a_missing_package_in_a_panel_names_the_pip_command(monkeypatch):
    import ui.results as results

    def boom(*a, **k):
        raise ModuleNotFoundError("No module named 'pptx'", name="pptx")
    monkeypatch.setattr(results, "render_proof_card", boom)
    at, _ = _solved_app()
    at.run()
    assert any("pip install python-pptx" in w.value for w in at.warning)


def test_streamlits_own_control_flow_passes_through_the_guards():
    def page():
        import streamlit as st
        from ui.safe import guarded, run_panel
        with guarded("a panel"):
            st.write("before")
            st.stop()
        st.write("AFTER-GUARDED")
        run_panel("b", lambda: None)
    at = AppTest.from_function(page, default_timeout=30).run()
    assert not at.exception and [m.value for m in at.markdown] == ["before"]


def test_run_panel_returns_its_default_on_failure_and_its_value_otherwise():
    def page():
        import streamlit as st
        from ui.safe import run_panel
        st.write(str(run_panel("fine", lambda x: x * 2, 21)))
        st.write(run_panel("broken", lambda: 1 / 0, default="fallback"))
    at = AppTest.from_function(page, default_timeout=30).run()
    assert [m.value for m in at.markdown] == ["42", "fallback"] and any("could not be shown" in w.value for w in at.warning)


# ------------------------------------------------------------------------------------------ the solve page

class _EmptyReplyClient(FakeClient):
    def chat(self, system, user, temperature=0.0, json_mode=False, model=None):
        from modules.llm_client import LLMOutputError
        raise LLMOutputError("The model returned an empty response (no text content) -- try again or check LM Studio's loaded model.", raw_output="", kind="empty",
                              diagnostics={"finish_reason": "length", "completion_tokens": 2048, "max_tokens": 2048, "reasoning_chars": 0, "model": "m"})


def _patched_solve_page(monkeypatch, chat):
    monkeypatch.setattr(LMStudioClient, "is_available", lambda self: (True, "ok"))
    monkeypatch.setattr(LMStudioClient, "list_models", lambda self: ["m"])
    monkeypatch.setattr(LMStudioClient, "chat", chat)
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.text_area[0].set_value("A car accelerates uniformly from 8 m/s to 20 m/s over 6 seconds. What is its acceleration?")
    next(b for b in at.button if "Represent" in b.label).click()
    at.run()
    return at


def test_an_empty_model_reply_is_explained_with_causes_and_the_exact_setting(monkeypatch):
    at = _patched_solve_page(monkeypatch, lambda self, *a, **k: _EmptyReplyClient().chat(*a, **k))
    assert not at.exception
    errors = " ".join(e.value for e in at.error)
    assert "empty reply" in errors
    body = " ".join(m.value for m in at.markdown)
    assert "What probably happened" in body and "Max response tokens" in body and "thinking" in body
    assert at.session_state["model"] is None


def test_a_connection_failure_mid_solve_is_explained_not_a_traceback(monkeypatch):
    import httpx
    from openai import APIConnectionError

    def chat(self, *a, **k):
        raise APIConnectionError(request=httpx.Request("POST", "http://localhost:1234/v1"))
    at = _patched_solve_page(monkeypatch, chat)
    assert not at.exception and any("Lost contact with LM Studio" in e.value for e in at.error)
    assert any("Deriving equations" in c.value for c in at.caption)         # which stage it happened in


def test_history_failing_does_not_cost_the_result(monkeypatch):
    from modules import history

    class Good(FakeClient):
        pass
    good = FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0})
    monkeypatch.setattr(history, "save", lambda *a, **k: (_ for _ in ()).throw(OSError("database is locked")))
    at = _patched_solve_page(monkeypatch, lambda self, *a, **k: good.chat(*a, **k))
    assert not at.exception
    assert at.session_state["model"] is not None
    assert any("could not be saved to history" in w.value for w in at.warning)
    assert any("Domain" in h.value for h in at.subheader)


# ------------------------------------------------------------------------------------------ the pipeline's own stages

def test_a_failing_step_trace_keeps_the_derivation_and_reports_a_warning(monkeypatch):
    monkeypatch.setattr(pipeline, "compute_steps", lambda model: (_ for _ in ()).throw(RuntimeError("steps exploded")))
    result = pipeline.run_pipeline(FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}), "A car accelerates from 8 m/s to 20 m/s in 6 s.")
    assert result.report.passed and result.steps == {} and any("steps exploded" in w and "step-by-step" in w for w in result.warnings)


def test_a_failing_scenario_stage_is_a_warning_too(monkeypatch):
    monkeypatch.setattr(pipeline, "generate_alternative_scenarios", lambda client, model: (_ for _ in ()).throw(RuntimeError("scenarios exploded")))
    result = pipeline.run_pipeline(FakeClient(payload_json=KINEMATICS_JSON, final_answers={"a": 2.0}), "x")
    assert result.steps and any("Alternative scenarios" in w for w in result.warnings)


def test_a_derivation_failure_still_raises_and_is_tagged_with_its_stage():
    class Prose(FakeClient):
        def chat(self, *a, **k):
            return LLMText("I think the answer is 2.")
    from modules.llm_client import LLMOutputError
    with pytest.raises(LLMOutputError) as info:
        pipeline.run_pipeline(Prose(payload_json=KINEMATICS_JSON, final_answers={}), "x")
    assert info.value.pipeline_stage == "Deriving equations from the problem statement..."


def test_batch_rows_get_the_explanation_and_the_first_fix():
    from modules.batch_solver import solve_one
    r = solve_one(_EmptyReplyClient(payload_json=KINEMATICS_JSON, final_answers={}), 0, "x")
    assert r.error and "empty reply" in r.error and "Try:" in r.error and "Max response tokens" in r.error


def test_the_unverified_banner_offers_the_explanation():
    ctx = kinematics_ctx()
    ctx.report.add("Dimensional consistency: accel", False, "m/s vs m/s^2")
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    at.session_state["model"], at.session_state["report"] = ctx.model, ctx.report
    at.session_state["steps"], at.session_state["problem_text"] = ctx.steps_by_target, ctx.problem_text
    at.run()
    assert not at.exception
    assert any("Why didn't this verify" in e.label for e in at.expander)
    assert any("did not pass its checks" in w.value for w in at.warning)
