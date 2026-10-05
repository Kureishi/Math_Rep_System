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
import functools
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


@pytest.fixture
def gif_stub(monkeypatch):
    """Swaps a GIF renderer, as imported into a UI module, for a recorder that returns a tiny fake GIF.

    The UI tests that press a "GIF" button are about the WIRING -- the button hands the renderer the right
    arguments, the result reaches the download button, nothing raises -- and rendering 40 matplotlib frames to find
    that out costs 5-15 seconds a test. The renderers themselves are exercised for real (frame counts, valid GIF
    bytes) in test_gif_export.py and test_time_plots*.py, with the same arguments.
    Usage: calls = gif_stub("ui.results.summary", "snapshot_motion_diagram_gif"); ...; (_, args, kwargs), = calls"""
    def install(module: str, name: str) -> list:
        calls: list = []

        def fake(*args, **kwargs):
            calls.append((name, args, kwargs))
            return b"GIF89a-stub"
        monkeypatch.setattr(f"{module}.{name}", fake)
        return calls
    return install


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
    from ui.results.time_views import render_ode_time_views
    import tests.test_ode_trajectories as models
    import tests.test_time_uncertainty as fan_models

    for key, default in [("plot_snapshots", {}), ("pdf_bytes", None), ("problem_text", "")]:
        st.session_state.setdefault(key, default)
    model = {"spiral": models.spiral_model, "decay": models.decay_model,
             "decay_no_ic": lambda: models.decay_model(ics=()),
             "oscillator": fan_models._symbolic_oscillator}[which]()
    render_ode_solution(Workspace(st.session_state), model)
    render_ode_time_views(model)


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


@functools.lru_cache(maxsize=None)
def _shared_ode_page(which):
    """A rendered ODE page that READ-ONLY tests share, so each page is booted once per run rather than once per
    test (a boot is several seconds, nearly all of it building Plotly figures). A test that uses this must not touch
    a widget; anything that needs set_value() or click() boots its own page with _ode_page()."""
    return _ode_page(which)


def _expander_labels(at):
    return [e.label for e in at.expander]


def test_coupled_ode_page_offers_error_over_time_animated_flow_and_linked_view():
    at = _shared_ode_page("spiral")
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


def test_coupled_ode_page_controls():
    """One boot for the page's three controls (each used to boot its own): the extra-starting-points slider, the
    comparison window, and putting the error plot in the report."""
    at = _ode_page("spiral")
    flow = lambda: next(s for s in _specs(at) if _title(s).startswith("Phase portrait flow"))
    assert len(flow()["frames"][2]["data"][1]["x"]) == 7                       # the solved path + the default ring of 6
    next(s for s in at.slider if s.key == "phase_extra_x_y").set_value(0).run()
    assert _exceptions(at) == []
    assert len(flow()["frames"][2]["data"][1]["x"]) == 1                       # just the solved trajectory

    slider = next(s for s in at.slider if s.key == "odeerr_end_x, y")
    default_end = slider.value
    slider.set_value(default_end * 3).run()
    assert _exceptions(at) == []
    error = next(s for s in _specs(at) if "Numerical integration and closed form" in _title(s))
    assert _values(error["data"][0]["x"])[-1] == pytest.approx(default_end * 3)    # the plot really covers the longer span
    assert any("Agreement everywhere shown" in s.value for s in at.success)

    next(b for b in at.button if b.key == "include_snap_odeerr_x, y").click().run()
    assert _exceptions(at) == []
    snap = at.session_state["plot_snapshots"]["odeerr_x, y"]
    assert snap.png_bytes[:8] == PNG_MAGIC and "numerical integration" in snap.title


def test_single_ode_gets_the_error_view_but_no_phase_portrait():
    labels = _expander_labels(_shared_ode_page("decay"))
    assert any("numerical integration over time: N" in lab for lab in labels)
    assert not any("Phase portrait" in lab for lab in labels)


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


def test_results_that_cannot_be_animated_show_no_animation_and_do_not_crash():
    at = _transforms_page()
    _fill(at, "series_expr", "exp(1/x)")                                      # essential singularity at 0: no Taylor series
    _click(at, "series_button")
    assert _exceptions(at) == []
    assert not any("Watch the approximation converge" in lab for lab in _expander_labels(at))

    next(r for r in at.radio if r.key == "series_kind").set_value("Asymptotic (x \u2192 \u221e)").run()
    _fill(at, "series_expr", "1/(x+1)")
    _click(at, "series_button")
    assert _exceptions(at) == []
    assert at.session_state["series_expand_result"].kind == "asymptotic"     # the click really produced an asymptotic result
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


# ================================================================== uncertainty fan and parameter morph (ODE page)

def _render_errors(at):
    """Errors from a failed GIF render. (The app's own 'Could not reach LM Studio' banner is an st.error too,
    and is expected in tests, so only the GIF button's message counts.)"""
    return [e.value for e in at.error if "render this animation" in e.value]


def _set(at, kind, key, value):
    next(w for w in getattr(at, kind) if w.key == key).set_value(value).run()
    assert _exceptions(at) == []


def _chart(at, starts_with):
    return next(s for s in _specs(at) if _title(s).startswith(starts_with))


def test_ode_page_offers_a_fan_and_a_morph_when_the_solution_is_fully_determined():
    labels = _expander_labels(_shared_ode_page("decay"))
    assert "🌫️ Uncertainty over time: N" in labels and "🎞️ Parameter morph: N" in labels
    labels = _expander_labels(_shared_ode_page("spiral"))
    assert "🌫️ Uncertainty over time: x, y" in labels and "🎞️ Parameter morph: x, y" in labels


def test_ode_without_initial_conditions_gets_no_comparison_fan_or_morph():
    labels = _expander_labels(_shared_ode_page("decay_no_ic"))        # the solution still contains an integration constant
    assert not any("numerical integration over time" in lab or "Uncertainty over time" in lab
                   or "Parameter morph" in lab for lab in labels)


def test_fan_defaults_to_five_percent_and_reports_the_sampled_bands():
    at = _shared_ode_page("decay")
    fan = _chart(at, "Uncertainty over time")
    assert "800 samples" in _title(fan) and "seed 12345" in _title(fan)
    stds = {n.key: n.value for n in at.number_input if n.key and n.key.startswith("fan_sd_")}
    assert stds == {"fan_sd_N_k_0.5": pytest.approx(0.025), "fan_sd_N_N(0)_100": pytest.approx(5.0)}
    text = "\n".join(m.value for m in at.markdown)
    assert "**N** at t = 10: nominal" in text and "90% of samples in" in text
    assert any("Guaranteed bound" in c.value for c in at.caption)           # the envelope is on by default


def test_fan_controls_change_what_is_drawn_and_the_fan_can_go_in_the_report():
    at = _ode_page("decay")
    _set(at, "number_input", "fan_seed_N", 7)
    assert "seed 7" in _title(_chart(at, "Uncertainty over time"))
    _set(at, "slider", "fan_n_N", 300)
    assert "300 samples" in _title(_chart(at, "Uncertainty over time"))

    n_traces = len(_chart(at, "Uncertainty over time")["data"])
    _set(at, "checkbox", "fan_env_N", False)
    assert len(_chart(at, "Uncertainty over time")["data"]) == n_traces - 2        # the two envelope edges are gone
    assert not any("Guaranteed bound" in c.value for c in at.caption)

    next(b for b in at.button if b.key == "include_snap_timefan_N").click().run()   # the report records THESE settings
    assert _exceptions(at) == []
    snap = at.session_state["plot_snapshots"]["timefan_N"]
    assert snap.png_bytes[:8] == PNG_MAGIC and "seed 7" in snap.caption and "300 samples" in snap.caption
    assert "k \u00b10.025" in snap.caption

    _set(at, "number_input", "fan_sd_N_k_0.5", 0.0)                              # now make every input certain
    _set(at, "number_input", "fan_sd_N_N(0)_100", 0.0)
    assert any("at least one input a standard deviation" in i.value for i in at.info)
    assert not any(_title(s).startswith("Uncertainty over time") for s in _specs(at))


def test_morph_controls_range_errors_and_gif_wiring(gif_stub):
    calls = gif_stub("ui.results.time_views", "snapshot_parameter_morph_gif")
    at = _ode_page("decay")
    morph = _chart(at, "N(0) =")                                           # the first parameter alphabetically is the initial value
    assert len(morph["frames"]) == 40
    _set(at, "selectbox", "morph_param_N", "k")
    morph = _chart(at, "k =")
    assert len(morph["frames"]) == 40 and all("monotone" in f["layout"]["title"]["text"] for f in morph["frames"])
    assert any("doesn't change character" in c.value for c in at.caption)  # pure decay never starts to oscillate
    _set(at, "slider", "morph_n_N", 20)
    assert len(_chart(at, "k =")["frames"]) == 20

    next(b for b in at.button if b.key == "gif_morph_N_k").click().run()    # the GIF button gets the family currently on screen
    assert _exceptions(at) == [] and not _render_errors(at)
    (_, args, _kwargs), = calls
    assert args[0].param_name == "k" and len(args[0].values) == 20

    _set(at, "number_input", "morph_hi_N_k_0.5", -5.0)                      # "to" below "from"
    assert any("range for k" in w.value for w in at.warning)
    _set(at, "number_input", "fan_tend_N", -3.0)                            # an end time before the start
    assert any("end time must be after" in w.value for w in at.warning)


def test_morph_reports_where_a_solution_starts_to_oscillate():
    at = _ode_page("oscillator")                                           # "x" is the function shown by default
    _set(at, "selectbox", "morph_param_x, y", "w")
    assert any("The shape first changes between w =" in c.value and "visible turning points" in c.value
               for c in at.caption)
    frames = _chart(at, "w =")["frames"]
    assert "monotone" in frames[0]["layout"]["title"]["text"] or "turning point" in frames[0]["layout"]["title"]["text"]
    assert "turning points" in frames[-1]["layout"]["title"]["text"]


# ================================================================== bifurcation diagram and cobweb (recurrence page)

def _map_script(which):
    import streamlit as st
    from modules.equation_engine import build_model
    from modules.workspace import Workspace
    from ui.results.solutions import render_recurrence_solution
    from ui.results.time_views import render_map_views

    for key, default in [("plot_snapshots", {}), ("pdf_bytes", None), ("problem_text", "")]:
        st.session_state.setdefault(key, default)

    def model(expression, known, ic):
        return build_model({
            "problem_domain": "p", "problem_type": "recurrence", "independent_variable": "n",
            "variables": [{"symbol": sym, "meaning": sym, "known_value": val, "unit": None, "is_function": False}
                          for sym, val in known]
                         + [{"symbol": "a", "meaning": "a", "known_value": None, "unit": None, "is_function": True}],
            "equations": [{"name": "map", "kind": "recurrence", "expression": expression, "derivation": ""}],
            "solve_for": ["a"], "assumptions": [],
            "initial_conditions": [{"expression": "a(0)", "value": ic}]})

    m = {
        "logistic": lambda: model("Eq(a(n+1), r*a(n)*(1-a(n)))", [("r", 3.2)], 0.3),
        "linear": lambda: model("Eq(a(n+1), r*a(n))", [("r", 1.5)], 2.0),
        "two_params": lambda: model("Eq(a(n+1), r*s*a(n)*(1-a(n)))", [("r", 3.2)], 0.3),
        "no_params": lambda: model("Eq(a(n+1), a(n)**2)", [], 0.5),
        "second_order": lambda: model("Eq(a(n+2), a(n+1) + a(n))", [], 1.0),
    }[which]()
    render_recurrence_solution(Workspace(st.session_state), m)
    render_map_views(m)


def _map_page(which):
    at = AppTest.from_function(_map_script, args=(which,), default_timeout=TIMEOUT).run()
    assert _exceptions(at) == []
    return at


@functools.lru_cache(maxsize=None)
def _shared_map_page(which):
    """Read-only twin of _map_page -- see _shared_ode_page. Do not touch a widget on it."""
    return _map_page(which)


def test_logistic_map_gets_a_bifurcation_diagram_and_a_cobweb_despite_having_no_closed_form():
    at = _shared_map_page("logistic")
    labels = _expander_labels(at)
    assert "🌿 Bifurcation diagram for a" in labels and "🕸️ Cobweb diagram for a" in labels
    bif = _chart(at, "Bifurcation diagram")
    assert _title(bif) == "Bifurcation diagram (x\u2080 = 0.3)"             # the initial condition is the start value
    xr = bif["layout"]["xaxis"]["range"]
    assert xr == [pytest.approx(1.6), pytest.approx(4.0)]                    # 0.5 r to 1.25 r around the known r = 3.2
    marker = [sh for sh in bif["layout"]["shapes"] if sh["line"]["dash"] == "dash"]
    assert len(marker) == 1 and marker[0]["x0"] == pytest.approx(3.2)        # "current" marks the known value
    regimes = next(c.value for c in at.caption if c.value.startswith("Regime changes:"))
    assert "period 2" in regimes and "period 4" in regimes


def test_bifurcation_controls_report_and_cobweb_gif_wiring(gif_stub):
    calls = gif_stub("ui.results.time_views", "snapshot_cobweb_gif")
    at = _map_page("logistic")
    _set(at, "number_input", "bif_lo_a_r", 2.5)
    _set(at, "number_input", "bif_hi_a_r", 3.4)
    bif = _chart(at, "Bifurcation diagram")
    assert bif["layout"]["xaxis"]["range"] == [pytest.approx(2.5), pytest.approx(3.4)]
    _set(at, "number_input", "bif_x0_a", 0.6)
    assert "x\u2080 = 0.6" in _title(_chart(at, "Bifurcation diagram"))
    _set(at, "slider", "bif_n_a", 200)
    assert len(_chart(at, "Bifurcation diagram")["data"]) == 1
    assert any("period 2" in c.value for c in at.caption if c.value.startswith("Regime changes:"))   # 3.0 is inside 2.5-3.4

    next(b for b in at.button if b.key == "gif_cobweb2_a").click().run()      # a cobweb for a map with no closed form has a GIF
    assert _exceptions(at) == [] and not _render_errors(at)
    (_, args, kwargs), = calls
    assert args[1] == pytest.approx(0.3) and kwargs["x_label"] == "a"        # starts from the initial condition, labelled by the function

    next(b for b in at.button if b.key == "include_snap_bifurcation_a").click().run()
    assert _exceptions(at) == []
    assert at.session_state["plot_snapshots"]["bifurcation_a"].png_bytes[:8] == PNG_MAGIC

    _set(at, "number_input", "bif_lo_a_r", 2.0)
    _set(at, "number_input", "bif_hi_a_r", 2.8)                             # a single stable fixed point throughout
    assert any(c.value == "One long-run behaviour throughout this range." for c in at.caption)
    _set(at, "number_input", "bif_hi_a_r", 1.0)                             # below "from"
    assert any("range for r" in w.value for w in at.warning)


def test_a_map_with_a_closed_form_keeps_its_one_cobweb_and_gains_a_bifurcation_diagram():
    at = _shared_map_page("linear")
    labels = _expander_labels(at)
    assert labels.count("🕸️ Cobweb diagram for a(n)") == 1                  # the original one, with the sequence plot
    assert "🕸️ Cobweb diagram for a" not in labels                          # not duplicated by the no-closed-form version
    assert "🌿 Bifurcation diagram for a" in labels


def test_a_second_parameter_without_a_value_is_asked_for():
    at = _shared_map_page("two_params")
    assert any("Give s a value to draw this diagram" in c.value for c in at.caption)
    assert "🕸️ Cobweb diagram for a" not in _expander_labels(at)             # the cobweb needs s too and is simply skipped
    assert not any(_title(s).startswith("Bifurcation diagram") for s in _specs(at))


@pytest.mark.parametrize("which", ["no_params", "second_order"])
def test_maps_without_something_to_vary_or_without_a_one_step_form_get_no_bifurcation(which):
    labels = _expander_labels(_shared_map_page(which))
    assert not any("Bifurcation" in lab for lab in labels)


# ================================================================== PDE evolution (heat and wave)

def _pde_page():
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "PDE" in m)
    at.run()
    assert _exceptions(at) == []
    return at


def test_heat_solution_is_shown_as_a_profile_with_the_whole_evolution_beneath(gif_stub):
    calls = gif_stub("ui.pde", "snapshot_space_time_gif")
    at = _pde_page()
    _set(at, "text_input", "heat_ic", "x*(1-x)")
    _click(at, "heat_button")
    assert _exceptions(at) == []
    chart = next(s for s in _specs(at) if "max|u(x,t)|" in _title(s))
    assert [d["type"] for d in chart["data"]] == ["scatter", "heatmap", "scatter"]    # profile, heatmap, time cursor
    assert len(chart["frames"]) == 60
    assert chart["frames"][0]["layout"]["title"]["text"].startswith("t = 0")
    heat = chart["data"][1]
    assert "Inferno" in str(heat["colorscale"]) or heat["colorscale"][0][1].lower() != heat["colorscale"][-1][1].lower()

    next(b for b in at.button if b.key == "gif_heat_d_anim").click().run()          # the GIF button is given the same field
    assert _exceptions(at) == [] and not _render_errors(at)
    (_, args, _kwargs), = calls
    assert args[0].error is None and args[0].u.shape[0] > 10 and args[0].u_max > 0


def test_wave_solution_gets_a_diverging_colour_scale_because_it_changes_sign():
    at = _pde_page()
    _set(at, "text_input", "wave_ic_disp", "sin(pi*x)")
    _click(at, "wave_button")
    assert _exceptions(at) == []
    chart = next(s for s in _specs(at) if "max|u(x,t)|" in _title(s))
    assert chart["data"][1]["zmid"] == 0                                    # symmetric about zero: blue/red, not one-sided


# ================================================================== solve-order replay, chain flow, motion upgrades

def _annotation_text(spec):
    notes = spec.get("layout", {}).get("annotations") or [{}]
    return notes[0].get("text", "")


def _motion_spec(at):
    return next(s for s in _specs(at) if _annotation_text(s).startswith("motion"))


@functools.lru_cache(maxsize=None)
def _shared_kinematics_page():
    """A solved kinematics problem rendered once for the read-only tests below. Do not touch a widget on it."""
    at = _app()
    _seed_solved_kinematics(at)
    at.run()
    assert _exceptions(at) == []
    return at


def test_dependency_graph_can_be_replayed_in_solve_order():
    at = _shared_kinematics_page()
    replay = next(s for s in _specs(at) if _title(s).startswith("Givens:"))
    assert _title(replay) == "Givens: t, u, v"
    assert len(replay["frames"]) == 3                                    # the givens, then one frame per equation
    assert replay["frames"][1]["layout"]["title"]["text"].startswith("Step 1 of 2: solve a from final velocity equation")
    assert replay["frames"][2]["layout"]["title"]["text"].startswith("Step 2 of 2: solve d from displacement equation")
    text = "\n".join(m.value for m in at.markdown)
    assert "1. solve a from final velocity equation (needs t, u, v)" in text
    assert "2. solve d from displacement equation (needs a, t, u)" in text
    assert any("dependencies" in c.value and "rather than a trace" in c.value for c in at.caption)


def test_gif_buttons_on_the_kinematics_page(gif_stub):
    """The solve-order GIF is only three frames, so it is rendered for real -- one genuine end-to-end click through
    gif_download_button. The motion GIF is 40 frames, so it is stubbed and checked for what it is HANDED."""
    motion_calls = gif_stub("ui.results.summary", "snapshot_motion_diagram_gif")
    at = _app()
    _seed_solved_kinematics(at)
    at.run()
    next(b for b in at.button if b.key == "gif_solve_order").click().run()
    assert _exceptions(at) == [] and not _render_errors(at)

    _set(at, "checkbox", "motion_show_a", False)                             # the button must reflect the CURRENT controls
    _set(at, "slider", "motion_strobe_n", 5)
    next(b for b in at.button if b.key == "gif_motion_diagram").click().run()
    assert _exceptions(at) == [] and not _render_errors(at)
    (_, args, kwargs), = motion_calls
    assert len(args) == 3 and len(args[0]) > 10                              # time, position, velocity
    assert kwargs["a_values"] is None and kwargs["n_strobes"] == 5


def _two_step_chain():
    import modules.chains as chains_module
    from modules.chains import InputBinding
    from tests.test_chains import _downstream_model, _kinematics_model
    cid = chains_module.create_chain("kinematics")
    chains_module.add_step(cid, "car accelerates", _kinematics_model(), "a")
    chains_module.add_step(cid, "then cruises", _downstream_model(), "d",
                           bindings=[InputBinding("a", "upstream", None, 0, "a")])
    chains_module.resolve_chain(cid)
    return cid


def _chains_page(cid):
    at = _app()
    at.session_state["app_mode"] = next(m for m in MODE_LABELS if "chain" in m.lower())
    at.session_state["active_chain_id"] = cid
    at.run()
    assert _exceptions(at) == []
    return at


def test_chain_page_shows_the_value_flow_between_steps(gif_stub):
    calls = gif_stub("ui.chains", "snapshot_chain_flow_gif")
    cid = _two_step_chain()
    at = _chains_page(cid)
    assert "🌊 Value flow through the chain" in _expander_labels(at)
    flow = next(s for s in _specs(at) if "Step 1 receives" in _title(s))
    assert len(flow["frames"]) == 2
    assert flow["frames"][1]["layout"]["title"]["text"] == "Step 2 receives a = 2 from step 1; result: d = 20."
    link_labels = flow["frames"][1]["data"][2]["text"]
    assert list(link_labels) == ["a = 2"]                                # the value that travelled, written on the link
    text = "\n".join(m.value for m in at.markdown)
    assert "- Step 1 receives no overridden inputs; result: a = 2." in text

    next(b for b in at.button if b.key == f"gif_chain_flow_{cid}").click().run()
    assert _exceptions(at) == [] and not _render_errors(at)
    (_, args, _kwargs), = calls
    assert [s.symbol for s in args[0].steps] == ["a", "d"]               # the GIF is given the flow that is on screen


def test_a_one_step_chain_has_no_value_flow():
    import modules.chains as chains_module
    from tests.test_chains import _kinematics_model
    cid = chains_module.create_chain("solo")
    chains_module.add_step(cid, "car accelerates", _kinematics_model(), "a")
    assert "🌊 Value flow through the chain" not in _expander_labels(_chains_page(cid))


def test_motion_diagram_shows_acceleration_and_a_strobe_trail_by_default():
    motion = _motion_spec(_shared_kinematics_page())
    assert _annotation_text(motion) == "motion \u2014 red: velocity, green: acceleration (each scaled separately)"
    assert len(motion["data"]) == 9                                      # 5 original + acceleration + ghosts + 2 ghost arrows
    ghosts_by_frame = [len(f["data"][6]["x"]) for f in motion["frames"]]
    assert ghosts_by_frame == sorted(ghosts_by_frame) and ghosts_by_frame[-1] == 8     # default of 8 strobe positions


def test_motion_diagram_controls_switch_the_upgrades_off_and_on():
    at = _app()
    _seed_solved_kinematics(at)
    at.run()
    _set(at, "checkbox", "motion_show_a", False)
    motion = _motion_spec(at)
    assert "acceleration" not in _annotation_text(motion) and len(motion["data"]) == 7     # strobes remain, no accel arrows
    _set(at, "checkbox", "motion_show_strobes", False)
    motion = _motion_spec(at)
    assert _annotation_text(motion) == "motion" and len(motion["data"]) == 5               # back to the original diagram
    assert next(s for s in at.slider if s.key == "motion_strobe_n").disabled is True
    _set(at, "checkbox", "motion_show_strobes", True)
    _set(at, "slider", "motion_strobe_n", 3)
    assert [len(f["data"][5]["x"]) for f in _motion_spec(at)["frames"]][-1] == 3




# ================================================================== reruns reuse unchanged work (ui/cache.py)

def _counting(monkeypatch, module, name):
    """Wraps module.name so every call is recorded; returns the list of calls. The real function still runs."""
    import importlib
    mod = importlib.import_module(module)
    real = getattr(mod, name)
    calls: list = []

    def wrapper(*args, **kwargs):
        calls.append(1)
        return real(*args, **kwargs)
    monkeypatch.setattr(mod, name, wrapper)
    return calls


def test_an_unrelated_widget_does_not_rebuild_what_it_has_nothing_to_do_with(monkeypatch):
    """Streamlit re-runs the whole script on every widget change. Moving the fan's sample slider used to rebuild the
    phase portrait, the flow animation, the linked view and the error plot and re-solve the ODE (about 2 s). Now only
    the fan -- whose input actually changed -- is rebuilt, and a control that feeds one figure rebuilds only that one."""
    solve = _counting(monkeypatch, "ui.results.solutions", "solve_ode")
    symbolic_solve = _counting(monkeypatch, "ui.results.time_views", "solve_ode")
    portrait = _counting(monkeypatch, "ui.results.solutions", "build_phase_portrait")
    trail = _counting(monkeypatch, "ui.results.solutions", "build_phase_trail_animation")
    linked = _counting(monkeypatch, "ui.results.solutions", "build_time_linked_view")
    error = _counting(monkeypatch, "ui.results.solutions", "build_ode_error_plot")
    fan = _counting(monkeypatch, "ui.results.time_views", "build_uncertainty_fan")
    morph = _counting(monkeypatch, "ui.results.time_views", "build_parameter_morph")

    at = _ode_page("spiral")
    built = lambda: (len(solve), len(symbolic_solve), len(portrait), len(trail), len(linked), len(error), len(fan), len(morph))
    assert built() == (1, 1, 1, 1, 1, 1, 1, 1)

    _set(at, "slider", "fan_n_x, y", 900)                         # only the fan depends on its own sample count
    assert built() == (1, 1, 1, 1, 1, 1, 2, 1)
    assert "900 samples" in _title(_chart(at, "Uncertainty over time"))

    _set(at, "slider", "phase_extra_x_y", 3)                      # feeds the flow animation, not the portrait or the linked view
    assert built() == (1, 1, 1, 2, 1, 1, 2, 1)
    flow = next(s for s in _specs(at) if _title(s).startswith("Phase portrait flow"))
    assert len(flow["frames"][2]["data"][1]["x"]) == 4            # the solved path + 3 extra: the figure really changed

    _set(at, "number_input", "fan_seed_x, y", 99)                 # and the seed, only the fan
    assert built() == (1, 1, 1, 2, 1, 1, 3, 1) and "seed 99" in _title(_chart(at, "Uncertainty over time"))


def test_changing_the_model_re_solves_instead_of_showing_the_old_solution():
    """The cache is keyed by the model's content, so an equal model reuses the solution and a different one does not."""

    def script():
        import streamlit as st
        from modules.workspace import Workspace
        from ui.results.solutions import render_ode_solution
        import tests.test_ode_trajectories as models

        for key, default in [("plot_snapshots", {}), ("pdf_bytes", None), ("problem_text", "")]:
            st.session_state.setdefault(key, default)
        initial = st.session_state.get("initial_value", 100.0)
        render_ode_solution(Workspace(st.session_state), models.decay_model(ics=(("N(0)", initial),)))

    at = AppTest.from_function(script, default_timeout=TIMEOUT).run()
    assert _exceptions(at) == []
    shown = lambda: " ".join(l.value for l in at.latex)
    assert "100.0" in shown() and "50.0" not in shown()

    at.session_state["initial_value"] = 50.0
    at.run()
    assert _exceptions(at) == []
    assert "50.0" in shown() and "100.0" not in shown()           # a different model: solved afresh, not served from the cache
