"""One lean check that the Health page runs and reports, with and without a reachable LM Studio."""
from tests.test_ui_smoke import _app

MODE = "🩺 Health check"


def test_the_page_runs_the_checks_and_offers_a_pasteable_report():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value(MODE)
    at.run(timeout=120)
    assert not at.exception
    text = " ".join(m.value for m in at.markdown)
    assert "Symbolic engine" in text and "Solve-and-verify pipeline" in text and "LM Studio" in text
    assert any("health report" in c.value for c in at.code)
