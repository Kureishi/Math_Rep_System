"""Accessible charts and status cues in the running app, and the GPU/CPU chip with what really ran."""
import json
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules import gpu, palette
from modules import sm_lattice as lat
from modules import sm_randommatrix as rm
from ui import theme
from tests.test_ui_smoke import _app

ROOT = str(Path(__file__).resolve().parent.parent)
FAKE = gpu.Backend(np, "Pretend GPU", True, "numpy")

CHART = f'''
import sys; sys.path.insert(0, {ROOT!r})
import streamlit as st, plotly.graph_objects as go
from ui.a11y import show_figure
fig = go.Figure(go.Scatter(x=[0, 1, 2], y=[1, 3, 2], name="demo"))
fig.update_layout(title="Demo chart", xaxis_title="t", yaxis_title="value")
show_figure(fig, width="stretch", **ALT)
'''


def chart_app(alt=None, **state):
    at = AppTest.from_string(CHART.replace("**ALT", f"**{({'alt': alt} if alt else {})!r}"))
    for k, v in state.items():
        at.session_state[k] = v
    at.run()
    assert not at.exception
    return at


def layout(at):
    return json.loads(at.get("plotly_chart")[0].proto.spec)["layout"]


# ------------------------------------------------------------------------------------------ show_figure

def test_a_chart_gets_the_colour_blind_safe_colours_a_transparent_background_and_no_streamlit_theme_override():
    at = chart_app()
    lay = layout(at)
    assert lay["colorway"][:4] == palette.CATEGORICAL[:4] and lay["paper_bgcolor"] == "rgba(0,0,0,0)"
    assert at.get("plotly_chart")[0].proto.theme == ""                       # theme=None: ours is not overridden


def test_the_description_is_in_the_page_for_screen_readers_but_not_on_screen():
    at = chart_app()
    (html,) = at.get("html")
    assert 'class="mrs-sr-only"' in html.proto.body and "Demo chart" in html.proto.body and "x axis: t" in html.proto.body
    assert not at.caption


def test_the_person_can_choose_to_see_the_description_as_text_instead():
    at = chart_app(a11y_show_descriptions=True)
    assert [c.value for c in at.caption][0].startswith("📊 Chart: Demo chart") and not at.get("html")


def test_a_custom_description_replaces_the_generated_one_and_cannot_inject_markup():
    at = chart_app(alt="Rises then falls <script>alert(1)</script>")
    body = at.get("html")[0].proto.body
    assert "Rises then falls" in body and "<script>" not in body and "&lt;script&gt;" in body


def test_the_chart_follows_the_apps_own_dark_mode_switch():
    light, dark = layout(chart_app())["template"]["layout"], layout(chart_app(dark_mode=True))["template"]["layout"]
    assert dark["paper_bgcolor"] != light["paper_bgcolor"] and dark["paper_bgcolor"] == "rgb(17,17,17)"


def test_switching_theme_does_not_damage_a_figure_that_is_reused():
    import plotly.graph_objects as go
    from ui import a11y
    fig = go.Figure(go.Scatter(x=[1, 2, 3], y=[3, 2, 1]))
    for dark in (False, True, False):
        palette.apply_theme(fig, dark)
    assert list(fig.data[0].y) == [3, 2, 1] and fig.layout.colorway == tuple(palette.CATEGORICAL)
    assert a11y.is_dark() is False                                            # no session: light, never an exception


def test_show_figure_returns_the_selection_state_that_click_to_act_depends_on():
    """Regression: the first version dropped st.plotly_chart's return value, silently disabling every click-to-act panel."""
    script = f'''
import sys; sys.path.insert(0, {ROOT!r})
import streamlit as st, plotly.graph_objects as go
from ui.a11y import show_figure
event = show_figure(go.Figure(go.Bar(x=["a", "b"], y=[1, 2])), key="pick", on_select="rerun", selection_mode=("points",))
st.write("got-selection-object" if event is not None and hasattr(event, "selection") else "got-nothing")
'''
    at = AppTest.from_string(script)
    at.run()
    assert not at.exception and any(m.value == "got-selection-object" for m in at.markdown)


def test_a_real_page_describes_its_charts():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("🧭 Discover equations")
    at.run(timeout=120)
    next(b for b in at.button if b.key == "discover_run").click()
    at.run(timeout=180)
    assert not at.exception
    bodies = " ".join(h.proto.body for h in at.get("html"))
    assert "mrs-sr-only" in bodies and "Data against the discovered system" in bodies


def test_the_sidebar_offers_the_visible_descriptions_option():
    at = _app()
    at.run(timeout=120)
    assert any(c.key == "a11y_show_descriptions" for c in at.sidebar.checkbox)


# ------------------------------------------------------------------------------------------ status never depends on colour alone

@pytest.mark.parametrize("text, kind, expected", [
    ("Verified", "pass", "✓ Verified"), ("2 terms", "pass", "✓ 2 terms"), ("Not sure", "warn", "! Not sure"),
    ("diverges", "fail", "✕ diverges"), ("✅ Converged", "pass", "✅ Converged"), ("🟡 Not converged", "warn", "🟡 Not converged"),
    ("plain note", "neutral", "plain note"), ("", "pass", ""),
])
def test_a_status_symbol_is_added_unless_the_text_already_has_one(text, kind, expected):
    assert theme._with_symbol(text, kind) == expected


def test_badges_carry_their_kind_as_a_class_and_use_the_accessible_colours():
    html = theme.badge("Verified", "pass")
    bg, fg = palette.BADGE_COLORS["pass"]
    assert "mrs-badge-pass" in html and bg in html and fg in html and "✓ Verified" in html


# ------------------------------------------------------------------------------------------ the GPU / CPU chip

def chip_text(name):
    at = AppTest.from_string(f'''
import sys; sys.path.insert(0, {ROOT!r})
from ui.theme import backend_chip
backend_chip({name!r})
''')
    at.run()
    assert not at.exception
    return " ".join(m.value for m in at.markdown)


def test_the_chip_says_gpu_or_cpu_and_highlights_only_the_gpu():
    gpu_html, cpu_html = chip_text("GPU: NVIDIA RTX (CuPy)"), chip_text("CPU (NumPy)")
    assert "⚡ GPU: NVIDIA RTX (CuPy)" in gpu_html and "mrs-badge-pass" in gpu_html
    assert "🖥️ CPU (NumPy)" in cpu_html and "mrs-badge-neutral" in cpu_html


def test_attempt_records_the_backend_that_actually_produced_the_result():
    gpu.reset_last()
    assert gpu.last_used() is None
    gpu.attempt(FAKE, lambda: 1, lambda: 2)
    assert gpu.last_used() == "Pretend GPU"
    gpu.attempt(gpu.CPU, lambda: 1, lambda: 2)
    assert gpu.last_used() == "CPU (NumPy)"
    gpu.attempt(FAKE, lambda: (_ for _ in ()).throw(RuntimeError("out of memory")), lambda: 2)
    assert gpu.last_used() == "CPU (NumPy)"                                    # fell back: the chip must say CPU


def test_each_gpu_capable_result_reports_where_it_was_computed():
    with gpu.pretend_gpu(FAKE):
        assert rm.analyse("GOE", 24, 4, seed=1).backend == "Pretend GPU"
        assert lat.run_single(lat.LatticeModel("ising", 8), 2.0, 30, 10, seed=1).backend == "Pretend GPU"
        scan = lat.scan_temperatures(lat.LatticeModel("ising", 8), [2.0, 3.0], 5, 100, seed=1)
        assert scan.backend == "Pretend GPU"
    assert rm.analyse("GOE", 24, 4, seed=1).backend in ("CPU (NumPy)",) or gpu.available().is_gpu
    with gpu.pretend_gpu(FAKE):
        assert rm.analyse("Poisson", 24, 4, seed=1).backend == "CPU (NumPy)"    # no eigenvalue solve, so nothing ran on the GPU


def test_a_langevin_result_on_the_real_page_shows_where_it_was_computed():
    at = _app()
    at.run(timeout=120)
    at.sidebar.radio[0].set_value("🔥 Statistical mechanics")
    at.run(timeout=120)
    before = sum("🖥️ CPU" in m.value for m in at.markdown)            # the random-matrix tab computes on load
    next(b for b in at.button if b.key == "sst_run").click()
    at.run(timeout=300)
    assert not at.exception and sum("🖥️ CPU" in m.value for m in at.markdown) == before + 1
