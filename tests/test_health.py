"""The health check: each failure is diagnosed with a fix, no check can crash the page, and nothing is changed."""
import time

import pytest

from modules import health, history, templates
from modules.llm_client import pick_embedding_model


class FakeClient:
    def __init__(self, available=(True, "Connected to LM Studio."), loaded=("qwen2.5-14b-instruct", "qwen2-vl-7b-instruct"),
                 reply="ok", chat_error=None, delay=0.0):
        self.available, self.loaded, self.reply, self.chat_error, self.delay = available, list(loaded), reply, chat_error, delay

    def is_available(self):
        return self.available

    def list_models(self):
        return self.loaded

    def validate_model(self, name):
        if not self.available[0]:
            return self.available
        return (name in self.loaded, f"'{name}' is loaded and ready." if name in self.loaded else f"'{name}' isn't currently loaded.")

    def chat(self, system, user, temperature):
        time.sleep(self.delay)
        if self.chat_error:
            raise self.chat_error
        return self.reply


@pytest.fixture(autouse=True)
def isolated_databases(tmp_path, monkeypatch):
    """The health check opens the real history/templates databases; keep that in a temporary folder."""
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(templates, "DB_PATH", tmp_path / "templates.db")


def run(**kw):
    return health.run_checks(data_dir=kw.pop("data_dir", None), **kw)


# ------------------------------------------------------------------------------------------ the local checks

def test_a_healthy_install_without_lm_studio_has_no_failures_and_says_the_model_was_not_checked(tmp_path):
    report = run(data_dir=tmp_path / "data")
    assert report.overall == "ok" and report.by_id("lm_studio").status == "info"
    assert {c.id for c in report.checks} >= {"python", "packages", "sympy", "pipeline", "data", "databases", "gpu", "latex"}


def test_the_self_test_problem_gets_the_right_answer():
    c = health.check_pipeline()
    assert c.status == "ok" and c.seconds is not None and c.seconds < 10


def test_the_symbolic_engine_check_passes():
    assert health.check_symbolic_engine().status == "ok"


def test_an_unwritable_data_folder_is_a_failure_with_a_fix(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")                                           # a file where the folder should be
    c = health.check_data_folder(blocker / "data")
    assert c.status == "fail" and "not writable" in c.detail and c.fix


def test_low_disk_space_is_a_warning(tmp_path, monkeypatch):
    class Usage:
        free = 10 * 2 ** 20
    monkeypatch.setattr(health.shutil, "disk_usage", lambda p: Usage)
    c = health.check_data_folder(tmp_path)
    assert c.status == "warn" and "10 MB" in c.detail


def test_the_data_folder_probe_leaves_nothing_behind(tmp_path):
    health.check_data_folder(tmp_path)
    assert list(tmp_path.iterdir()) == []


def test_old_python_is_a_warning(monkeypatch):
    monkeypatch.setattr(health.sys, "version_info", (3, 9, 0, "final", 0))
    c = health.check_python()
    assert c.status == "warn" and "older than" in c.detail


# ------------------------------------------------------------------------------------------ packages

def test_requirements_are_parsed_with_their_operators(tmp_path):
    f = tmp_path / "req.txt"
    f.write_text("sympy==1.14.0  # pinned\n# a comment\nnumpy>=1.0\n-r other.txt\nstreamlit\n")
    assert health.parse_requirements(f) == [("sympy", "==", "1.14.0"), ("numpy", ">=", "1.0")]


def test_a_missing_package_fails_with_the_install_command(tmp_path):
    f = tmp_path / "req.txt"
    f.write_text("definitely-not-a-real-package==1.0\n")
    c = health.check_packages(f)
    assert c.status == "fail" and "definitely-not-a-real-package" in c.detail and "pip install" in c.fix


def test_a_different_pinned_or_too_old_package_is_only_a_warning(tmp_path):
    f = tmp_path / "req.txt"
    f.write_text("sympy==0.0.1\nnumpy>=999.0\n")
    c = health.check_packages(f)
    assert c.status == "warn" and "sympy" in c.detail and "numpy" in c.detail


def test_the_projects_own_requirements_are_satisfied():
    assert health.check_packages().status in ("ok", "warn")


# ------------------------------------------------------------------------------------------ LM Studio

def test_unreachable_lm_studio_is_a_failure_that_says_how_to_start_it():
    report = run(client=FakeClient(available=(False, "Cannot reach LM Studio at http://localhost:1234/v1.")))
    c = report.by_id("lm_studio")
    assert c.status == "fail" and "Developer tab" in c.fix and report.overall == "fail"
    assert not any(x.id == "reasoning_model" for x in report.checks)               # nothing else to say about a model


def test_a_missing_reasoning_model_fails_but_a_missing_vision_model_only_warns():
    report = run(client=FakeClient(loaded=["some-other-model"]))
    assert report.by_id("reasoning_model").status == "fail" and report.by_id("vision_model").status == "warn"
    assert "Photo input is unavailable" in report.by_id("vision_model").detail
    only_reasoning = run(client=FakeClient(loaded=["qwen2.5-14b-instruct"]))
    assert only_reasoning.by_id("vision_model").status == "warn" and only_reasoning.overall == "warn"


def test_a_fully_loaded_setup_is_ok():
    report = run(client=FakeClient())
    assert report.by_id("reasoning_model").status == "ok" and report.by_id("vision_model").status == "ok"
    assert report.overall == "ok"


def test_the_embedding_model_is_reported_as_optional():
    assert run(client=FakeClient()).by_id("embedding_model").status == "info"
    with_embed = run(client=FakeClient(loaded=["qwen2.5-14b-instruct", "text-embedding-nomic-embed-text-v1.5"]))
    c = with_embed.by_id("embedding_model")
    assert c.status == "ok" and "nomic" in c.detail


def test_embedding_models_are_picked_by_name_and_a_configured_one_is_never_swapped(monkeypatch):
    monkeypatch.delenv("LM_EMBEDDING_MODEL", raising=False)
    assert pick_embedding_model(["qwen2.5-14b", "nomic-embed-text-v1.5"]) == "nomic-embed-text-v1.5"
    assert pick_embedding_model(["qwen2.5-14b", "llama-3"]) is None
    assert pick_embedding_model(["a", "my-embedder"], "my-embedder") == "my-embedder"
    assert pick_embedding_model(["a", "nomic-embed-text"], "not-loaded") is None


# ------------------------------------------------------------------------------------------ the optional round trip

def test_the_round_trip_is_not_run_unless_asked():
    assert not any(c.id == "llm_round_trip" for c in run(client=FakeClient()).checks)


def test_the_round_trip_passes_fails_and_warns():
    assert run(client=FakeClient(), include_llm_round_trip=True).by_id("llm_round_trip").status == "ok"
    bad = run(client=FakeClient(chat_error=RuntimeError("context length is 0")), include_llm_round_trip=True).by_id("llm_round_trip")
    assert bad.status == "fail" and "context length is 0" in bad.detail
    odd = run(client=FakeClient(reply="Certainly! Here is a long answer"), include_llm_round_trip=True).by_id("llm_round_trip")
    assert odd.status == "warn"


def test_a_slow_model_is_a_warning_not_a_failure(monkeypatch):
    monkeypatch.setattr(health, "SLOW_LLM_SECONDS", 0.05)
    c = run(client=FakeClient(delay=0.12), include_llm_round_trip=True).by_id("llm_round_trip")
    assert c.status == "warn" and "took" in c.detail


def test_the_round_trip_is_skipped_when_the_model_is_not_even_loaded():
    report = run(client=FakeClient(loaded=[]), include_llm_round_trip=True)
    assert not any(c.id == "llm_round_trip" for c in report.checks)


# ------------------------------------------------------------------------------------------ robustness and the report

def test_a_check_that_blows_up_becomes_a_failed_check_and_the_others_still_run(monkeypatch):
    def explode():
        raise ZeroDivisionError("boom")
    monkeypatch.setattr(health, "check_symbolic_engine", explode)
    report = run()
    c = report.by_id("sympy")
    assert c.status == "fail" and "boom" in c.detail and "the check itself failed" in c.detail
    assert report.by_id("pipeline").status == "ok"


def test_an_unreadable_data_folder_does_not_crash_the_database_check(monkeypatch):
    monkeypatch.setattr(history, "_connect", lambda: (_ for _ in ()).throw(OSError("disk I/O error")))
    c = health.check_databases()
    assert c.status == "fail" and "disk I/O error" in c.detail


def test_the_text_report_is_pasteable_and_carries_the_fixes():
    report = run(client=FakeClient(loaded=[]))
    text = report.text()
    assert "Math Representation System health report" in text and "[FAIL]" in text and "fix:" in text
    assert report.headline().startswith(("1 problem", "2 problem", "3 problem"))
