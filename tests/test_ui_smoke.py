"""End-to-end UI tests, driven through Streamlit's own AppTest harness.

Everything else in the suite tests modules/ directly; ui/ (and app.py) was
only ever checked by hand, so a crash that only shows up once a real page is
rendered had nothing to catch it. These tests run the REAL app.py script
in-process and read what Streamlit would have drawn.

What is covered:
  * every sidebar mode renders its landing state without an exception
  * the three places the type-checker found a latent crash in the UI
    (a chain that vanishes mid-session, a best-fit family with no R²,
    the Explore tab's plot-target selector)

The four SQLite databases (history / chains / templates / settings profiles)
are redirected to a temp dir, so these tests never read or write the app's
real data. (data/app.log is the one exception, and it isn't specific to this
file: modules/app_logging attaches its file handler when it is first
imported, before any fixture can run, so the log file is created by every
test run.)
"""
import json
import re

import pytest
from streamlit.testing.v1 import AppTest

import modules.chains as chains_module
import modules.history as history_module
import modules.settings_profiles as profiles_module
import modules.templates as templates_module
from modules.command_palette import MODE_LABELS
from modules.curve_fitting import FitResult
from modules.equation_engine import build_model
from modules.solver import compute_steps
from modules.verifier import verify
from tests.conftest import FakeClient, KINEMATICS_TWO_TARGET_JSON

APP_PATH = str(__import__("pathlib").Path(__file__).resolve().parent.parent / "app.py")
TIMEOUT = 120


@pytest.fixture(autouse=True)
def isolated_data(tmp_path, monkeypatch):
    monkeypatch.setattr(history_module, "DB_PATH", tmp_path / "history.db")
    monkeypatch.setattr(chains_module, "DB_PATH", tmp_path / "chains.db")
    monkeypatch.setattr(templates_module, "DB_PATH", tmp_path / "templates.db")
    monkeypatch.setattr(profiles_module, "DB_PATH", tmp_path / "profiles.db")


def _app() -> AppTest:
    return AppTest.from_file(APP_PATH, default_timeout=TIMEOUT)


def _exceptions(at: AppTest) -> list[str]:
    return [str(e.value)[:200] for e in at.exception]


def _markdown_text(at: AppTest) -> str:
    return "\n".join(m.value for m in at.markdown)


# ------------------------------------------------------------------ every mode renders

def test_app_starts_without_exceptions():
    at = _app().run()
    assert _exceptions(at) == []


@pytest.mark.parametrize("mode", MODE_LABELS)
def test_every_sidebar_mode_renders_its_landing_state(mode):
    at = _app()
    at.session_state["app_mode"] = mode
    at.run()
    assert _exceptions(at) == [], f"mode {mode!r} raised on first render"


# ------------------------------------------------------------------ chains: a chain deleted mid-session

def test_chains_page_survives_the_selected_chain_disappearing(monkeypatch):
    """load_chain() returns None for an id that no longer exists (deleted from
    another tab between the list query and the load). The page used to read
    `.steps` off that None and crash with an AttributeError."""
    chains_module.create_chain("Doomed chain")
    monkeypatch.setattr(chains_module, "load_chain", lambda chain_id: None)

    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "chain" in m.lower())
    at.run()

    assert _exceptions(at) == []
    assert any("no longer exists" in w.value for w in at.warning)
    assert at.session_state["active_chain_id"] is None


# ------------------------------------------------------------------ curve fitting: best-fit family with no R²

def test_best_fit_ranking_skips_a_family_without_r_squared(monkeypatch):
    """Defence in depth: best_fit() promises successful fits only, and the real
    one keeps that promise -- but FitResult.r_squared / .rmse / .expr are
    typed Optional (they are None on the failure path), and the ranking
    sorted on r_squared and formatted it with :.5f, so ONE result breaking
    that promise would take the whole page down with a TypeError. Such a
    result is now dropped before ranking; the genuine fits still show."""
    from modules.curve_fitting import fit_curve
    good = fit_curve([1.0, 2.0, 3.0, 4.0], [2.1, 3.9, 6.2, 7.8], "linear")
    assert good.error is None and good.r_squared is not None        # a real fit, not a stand-in
    failed = FitResult(family="power", expr=None, param_values={}, r_squared=None, rmse=None,
                       residuals=[], error="needs positive data")
    import ui.curve_fitting as curve_fitting_ui
    monkeypatch.setattr(curve_fitting_ui, "best_fit", lambda xs, ys: {"power": failed, "linear": good})

    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "urve" in m)
    at.run()
    next(t for t in at.text_area if t.key == "fit_csv_paste").set_value("x,y\n1,2.1\n2,3.9\n3,6.2\n4,7.8\n").run()
    next(s for s in at.selectbox if s.key == "fit_family").set_value("best fit (try all)").run()
    next(b for b in at.button if "fit" in b.label.lower()).click().run()    # the result is gated on the Fit click

    assert _exceptions(at) == []
    text = _markdown_text(at)
    assert f"**linear**: R² = {good.r_squared:.5f}" in text          # the genuine fit is still ranked
    assert "**power**" not in text                                    # the broken one is dropped, not shown
    assert any("Best fit: **linear**" in s.value for s in at.success)


# ------------------------------------------------------------------ explore tab: plot-target selector

def _seed_solved_kinematics(at: AppTest) -> None:
    model = build_model(json.loads(KINEMATICS_TWO_TARGET_JSON))
    client = FakeClient(KINEMATICS_TWO_TARGET_JSON, final_answers={"a": 2.0, "d": 100.0})
    at.session_state["model"] = model
    at.session_state["report"] = verify(model, client, "test problem")
    at.session_state["steps"] = compute_steps(model)
    at.session_state["scenarios"] = []
    at.session_state["pdf_bytes"] = None
    at.session_state["plot_snapshots"] = {}


def _plotly_y_titles(at: AppTest) -> list[str | None]:
    titles = []
    for chart in at.get("plotly_chart"):
        title = json.loads(chart.proto.spec).get("layout", {}).get("yaxis", {}).get("title", {})
        titles.append(title.get("text") if isinstance(title, dict) else title)
    return titles


def test_explore_tab_y_axis_target_really_plots_the_solved_variable():
    """Regression for the bug the real UI exposed: the sliders include one for
    the Y-axis target itself, and substituting that value before solving made
    the solve fail silently, so the plot showed the equation's RESIDUAL no
    matter what target was picked. Picking a target must plot that variable."""
    at = _app()
    _seed_solved_kinematics(at)
    at.run()
    assert _exceptions(at) == []

    next(s for s in at.selectbox if s.key == "line_x").set_value("t").run()
    y_target = next(s for s in at.selectbox if "Y-axis target" in s.label)
    assert "a" in y_target.options
    y_target.set_value("a").run()

    assert _exceptions(at) == []
    titles = _plotly_y_titles(at)
    assert "a" in titles                                              # the solved curve a(t)
    assert not any(t and "residual" in t for t in titles)             # ...not the residual fallback
    assert any("slider above is ignored" in c.value for c in at.caption)
