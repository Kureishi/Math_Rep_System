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

import numpy as np
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


# ================================================================== time-resolved views (ODE page)

PNG_MAGIC = b"\x89PNG\r\n\x1a\n"


def _ode_script(which):
    """Renders the ODE results section on its own (no LLM or verification
    pipeline needed) for one of the models in tests/test_ode_trajectories."""
    import streamlit as st
    from modules.workspace import Workspace
    from ui.results.solutions import render_ode_solution
    import tests.test_ode_trajectories as models

    for key, default in [("plot_snapshots", {}), ("pdf_bytes", None), ("problem_text", "")]:
        st.session_state.setdefault(key, default)
    model = {"spiral": models.spiral_model, "decay": models.decay_model,
             "decay_no_ic": lambda: models.decay_model(ics=())}[which]()
    render_ode_solution(Workspace(st.session_state), model)


def _ode_page(which):
    at = AppTest.from_function(_ode_script, args=(which,), default_timeout=TIMEOUT).run()
    assert _exceptions(at) == []
    return at


def _specs(at):
    return [json.loads(c.proto.spec) for c in at.get("plotly_chart")]


def _values(arr):
    """A trace's x/y as a list. Plotly serialises numpy arrays as base64
    typed arrays ({"dtype": "f8", "bdata": ...}) and plain lists as lists."""
    if isinstance(arr, dict) and "bdata" in arr:
        import base64
        return np.frombuffer(base64.b64decode(arr["bdata"]), dtype=np.dtype(arr["dtype"])).tolist()
    return list(arr)


def _title(spec):
    title = spec.get("layout", {}).get("title", {})
    return (title.get("text") if isinstance(title, dict) else title) or ""


def _expander_labels(at):
    return [e.label for e in at.expander]


def test_coupled_ode_page_offers_error_over_time_animated_flow_and_linked_view():
    at = _ode_page("spiral")
    labels = _expander_labels(at)
    assert any("numerical integration over time" in lab for lab in labels)
    assert any("Phase portrait" in lab for lab in labels)

    specs = _specs(at)
    error = next(s for s in specs if "Numerical integration and closed form" in _title(s))
    assert "agree" in _title(error) and "DISAGREE" not in _title(error)
    flow = next(s for s in specs if _title(s).startswith("Phase portrait flow"))
    assert len(flow["frames"]) >= 2 and len(flow["frames"][0]["data"]) == 2
    linked = next(s for s in specs if _title(s).startswith("t = ") and s.get("frames"))
    assert len(linked["frames"]) >= 2 and len(linked["frames"][0]["data"]) == 3
    assert "t = " in linked["frames"][5]["layout"]["title"]["text"]


def test_extra_starting_points_slider_changes_how_many_paths_flow():
    at = _ode_page("spiral")
    flow = lambda: next(s for s in _specs(at) if _title(s).startswith("Phase portrait flow"))
    heads_with_ring = len(flow()["frames"][2]["data"][1]["x"])
    assert heads_with_ring == 7                                                # the solved path + the default ring of 6
    next(s for s in at.slider if s.key == "phase_extra_x_y").set_value(0).run()
    assert _exceptions(at) == []
    assert len(flow()["frames"][2]["data"][1]["x"]) == 1                      # just the solved trajectory


def test_error_window_slider_extends_the_comparison():
    at = _ode_page("spiral")
    slider = next(s for s in at.slider if s.key == "odeerr_end_x, y")
    default_end = slider.value
    slider.set_value(default_end * 3).run()
    assert _exceptions(at) == []
    error = next(s for s in _specs(at) if "Numerical integration and closed form" in _title(s))
    assert _values(error["data"][0]["x"])[-1] == pytest.approx(default_end * 3)        # the plot really covers the longer span
    assert any("Agreement everywhere shown" in s.value for s in at.success)


def test_error_plot_can_be_included_in_the_report():
    at = _ode_page("spiral")
    next(b for b in at.button if b.key == "include_snap_odeerr_x, y").click().run()
    assert _exceptions(at) == []
    snap = at.session_state["plot_snapshots"]["odeerr_x, y"]
    assert snap.png_bytes[:8] == PNG_MAGIC and "numerical integration" in snap.title


def test_single_ode_gets_the_error_view_but_no_phase_portrait():
    labels = _expander_labels(_ode_page("decay"))
    assert any("numerical integration over time: N" in lab for lab in labels)
    assert not any("Phase portrait" in lab for lab in labels)


def test_ode_without_initial_conditions_gets_no_comparison_and_no_crash():
    labels = _expander_labels(_ode_page("decay_no_ic"))
    assert not any("numerical integration over time" in lab for lab in labels)


# ================================================================== time-resolved views (Transforms & series mode)

def _transforms_page():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "Transforms" in m)
    at.run()
    assert _exceptions(at) == []
    return at


def _fill(at, key, value):
    next(t for t in at.text_input if t.key == key).set_value(value).run()


def _click(at, key):
    next(b for b in at.button if b.key == key).click().run()


def test_taylor_result_offers_a_convergence_animation():
    at = _transforms_page()
    _fill(at, "series_expr", "sin(x)")
    _click(at, "series_button")
    assert _exceptions(at) == []
    assert any("Watch the approximation converge" in lab for lab in _expander_labels(at))
    anim = next(s for s in _specs(at) if "through x^" in _title(s))
    assert len(anim["frames"]) == 3                                            # x, x^3, x^5 -- not one frame per power
    assert "sin(x)" in _title(anim) and "RMS error" in _title(anim)

    next(s for s in at.slider if s.key == "series_anim_halfwidth").set_value(10.0).run()
    assert _exceptions(at) == []
    wide = next(s for s in _specs(at) if "through x^" in _title(s))
    xs_wide = _values(wide["data"][0]["x"])
    assert xs_wide[0] == pytest.approx(-10.0) and xs_wide[-1] == pytest.approx(10.0)


def test_asymptotic_result_has_no_convergence_animation():
    at = _transforms_page()
    next(r for r in at.radio if r.key == "series_kind").set_value("Asymptotic (x \u2192 \u221e)").run()
    _fill(at, "series_expr", "1/(x+1)")
    _click(at, "series_button")
    assert _exceptions(at) == []
    assert at.session_state["series_expand_result"].kind == "asymptotic"     # the click really produced an asymptotic result
    assert not any("Watch the approximation converge" in lab for lab in _expander_labels(at))


def test_unexpandable_taylor_result_shows_no_animation_and_no_crash():
    at = _transforms_page()
    _fill(at, "series_expr", "exp(1/x)")                                      # essential singularity at 0: no series
    _click(at, "series_button")
    assert _exceptions(at) == []
    assert not any("Watch the approximation converge" in lab for lab in _expander_labels(at))


def test_fourier_series_tab_animates_a_square_wave():
    at = _transforms_page()
    _fill(at, "fseries_expr", "Piecewise((-1, x < 0), (1, True))")
    _click(at, "fseries_button")
    assert _exceptions(at) == []
    assert any("never increased" in s.value for s in at.success)
    anim = next(s for s in _specs(at) if "N = " in _title(s))
    assert len(anim["frames"]) == 12
    coeffs = at.dataframe[0].value
    assert coeffs.loc[1, "b\u2099"] == pytest.approx(4 / np.pi, abs=2e-3)   # b1 of a square wave

    next(s for s in at.slider if s.key == "fseries_N").set_value(20).run()
    _click(at, "fseries_button")
    assert _exceptions(at) == []
    assert len(next(s for s in _specs(at) if "N = " in _title(s))["frames"]) == 20


def test_fourier_series_tab_reports_a_bad_function():
    at = _transforms_page()
    _fill(at, "fseries_expr", "1/x")
    _click(at, "fseries_button")
    assert _exceptions(at) == []
    assert any("must be finite everywhere" in e.value for e in at.error)
