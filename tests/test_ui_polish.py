"""The final UI polish round: grouped mode navigation, history (stars, filter, day groups, undo), the persistent status line, short export captions, the motion-diagram slider."""
import sqlite3
from datetime import date
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import modules.history as history
from modules.mode_registry import REGISTRY
from modules.mode_spec import GROUP_ORDER, ModeSpec
from tests.report_helpers import kinematics_ctx

APP = str(Path(__file__).parent.parent / "app.py")


@pytest.fixture
def db(tmp_path, monkeypatch):
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "history.db")
    return tmp_path / "history.db"


def seed(text="A car accelerates from 8 m/s to 20 m/s in 6 s."):
    ctx = kinematics_ctx()
    return history.save(text, ctx.model, ctx.report, ctx.steps_by_target, [])


def _stamp(entry_id, iso):
    with sqlite3.connect(history.DB_PATH) as conn:
        conn.execute("UPDATE problems SET timestamp = ? WHERE id = ?", (iso, entry_id))


# ------------------------------------------------------------------------------------------ 1. grouped modes

def test_every_mode_is_in_exactly_one_known_section():
    grouped = REGISTRY.grouped()
    flat = [label for _, labels in grouped for label in labels]
    assert sorted(flat) == sorted(REGISTRY.labels) and len(flat) == len(set(flat))
    assert [g for g, _ in grouped] == [g for g in GROUP_ORDER if g in dict(grouped)]
    assert set(dict(grouped)) == set(GROUP_ORDER)                      # the built-ins use the four sections, no stray one


def test_the_default_mode_is_first_in_the_first_section():
    assert REGISTRY.grouped()[0][0] == "Solve" and REGISTRY.grouped()[0][1][0] == REGISTRY.labels[0]


def test_a_plugin_group_of_its_own_gets_a_section_after_the_built_in_ones():
    from modules import mode_registry
    extra = ModeSpec(label="🧩 Mine", group="My stuff", order=1)
    reg = mode_registry.Registry(specs=list(REGISTRY.specs) + [extra])
    assert reg.grouped()[-1] == ("My stuff", ["🧩 Mine"])


def test_an_empty_group_is_a_problem():
    assert "group must be a non-empty string" in ModeSpec(label="x", group=" ").problems()


def test_the_sidebar_lists_the_modes_as_grouped_buttons_and_the_selected_one_is_highlighted():
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    assert not at.exception
    assert not at.sidebar.radio                                         # the flat radio list is gone
    nav = {b.key: b for b in at.sidebar.button if (b.key or "").startswith("nav_")}
    assert len(nav) == len(REGISTRY.labels)
    assert nav[f"nav_{REGISTRY.labels[0]}"].proto.type == "primary"
    assert [e.label for e in at.sidebar.expander][:4] == [f"{g} · {len(l)}" for g, l in REGISTRY.grouped()]


def test_clicking_a_mode_button_switches_mode_and_a_bad_value_falls_back():
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    target = next(l for l in REGISTRY.labels if "Chemistry" in l)
    next(b for b in at.sidebar.button if b.key == f"nav_{target}").click()
    at.run()
    assert not at.exception and at.session_state["app_mode"] == target
    at.session_state["app_mode"] = "not a mode"
    at.run()
    assert not at.exception and at.session_state["app_mode"] == REGISTRY.labels[0]


# ------------------------------------------------------------------------------------------ 2. undo a delete

def test_a_deleted_entry_comes_back_exactly_as_it_was(db):
    entry_id = seed()
    history.rename(entry_id, "Keep me")
    history.set_pinned(entry_id, True)
    before = history.snapshot(entry_id)
    saved = history.delete_with_undo(entry_id)
    assert history.list_recent() == [] and saved == before
    assert history.restore(saved) is True
    assert history.snapshot(entry_id) == before and history.load(entry_id) is not None


def test_restore_refuses_a_taken_id_and_junk(db):
    entry_id = seed()
    assert history.restore(history.snapshot(entry_id)) is False
    assert history.restore({}) is False and history.restore("x") is False       # type: ignore[arg-type]
    assert history.delete_with_undo(9999) is None


def test_the_sidebar_delete_shows_an_undo_bar_and_undo_restores(db):
    entry_id = seed()
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    next(b for b in at.sidebar.button if b.key == f"delhist_{entry_id}").click()
    at.run()
    assert not at.exception and history.list_recent() == []
    assert any(b.key == "history_undo" for b in at.sidebar.button) and "Deleted" in " ".join(c.value for c in at.sidebar.caption)
    next(b for b in at.sidebar.button if b.key == "history_undo").click()
    at.run()
    assert not at.exception and [e["id"] for e in history.list_recent()] == [entry_id]
    assert not any(b.key == "history_undo" for b in at.sidebar.button)


# ------------------------------------------------------------------------------------------ 3. day groups, filter, stars

def test_day_labels():
    today = date(2026, 11, 22)
    assert [history.day_label(date(2026, 11, d), today) for d in (22, 21, 19)] == ["Today", "Yesterday", "Thursday"]
    assert history.day_label(date(2026, 11, 1), today) == "1 Nov 2026" and history.day_label(None, today) == "Earlier"


def test_sections_put_starred_first_then_group_by_day():
    e = lambda i, ts, pinned=False: {"id": i, "timestamp": ts, "pinned": pinned}        # noqa: E731
    entries = [e(5, "2026-11-22T10:00:00"), e(4, "2026-11-22T09:00:00"), e(3, "2026-11-21T09:00:00", True), e(2, "2026-11-10T09:00:00")]
    got = history.sections(entries, date(2026, 11, 22))
    assert [(h, [x["id"] for x in g]) for h, g in got] == [("⭐ Starred", [3]), ("Today", [5, 4]), ("10 Nov 2026", [2])]
    assert history.sections([]) == []


def test_the_filter_needs_every_word_in_the_name_domain_or_text():
    entry = {"title": "Motorway car", "domain": "kinematics", "problem_text": "A car accelerates from 8 m/s"}
    assert history.matches(entry, "") and history.matches(entry, "MOTORWAY accelerates") and history.matches(entry, "kinem")
    assert not history.matches(entry, "motorway pendulum")


def test_starred_entries_survive_pruning_and_do_not_count_toward_the_cap(db, monkeypatch):
    monkeypatch.setattr(history, "MAX_HISTORY_RECORDS", 3)
    first = seed("old favourite")
    history.set_pinned(first, True)
    for i in range(5):
        seed(f"problem {i}")
    ids = [e["id"] for e in history.list_with_pinned(3)]
    assert first in ids and len(history.all_ids()) == 4                         # the star plus the 3 newest
    assert history.is_pinned(first) and not history.is_pinned(history.all_ids()[0])
    history.set_pinned(first, False)
    seed("one more")
    assert first not in history.all_ids()                                       # unstarred, it is pruned like any other


def test_a_star_survives_a_project_bundle_round_trip(db, tmp_path, monkeypatch):
    from modules import project_bundle
    from modules.workspace import Workspace
    entry_id = seed()
    history.set_pinned(entry_id, True)
    bundle = project_bundle.export_bundle(workspace_entries={})
    monkeypatch.setattr(history, "DB_PATH", tmp_path / "fresh.db")
    project_bundle.import_bundle(bundle, Workspace({}))
    assert [e["pinned"] for e in history.list_recent()] == [True]


def test_a_database_without_the_pinned_column_is_upgraded(db, monkeypatch):
    monkeypatch.setattr(history, "_MIGRATIONS", history._MIGRATIONS[:4])
    old_id = seed("saved before stars existed")
    monkeypatch.undo()
    monkeypatch.setattr(history, "DB_PATH", db)
    assert history.list_recent()[0]["pinned"] is False
    assert history.set_pinned(old_id, True) and history.list_recent()[0]["pinned"] is True


def test_the_sidebar_filters_stars_and_groups(db):
    a, b = seed("A car accelerates from 8 m/s"), seed("A ball is thrown upward")
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    text = lambda: " ".join(m.value for m in at.sidebar.markdown)               # noqa: E731
    assert "**Today**" in text()
    next(btn for btn in at.sidebar.button if btn.key == f"pinhist_{a}").click()
    at.run()
    assert "**⭐ Starred**" in text() and history.is_pinned(a)
    at.sidebar.text_input(key="history_filter").set_value("ball").run()
    caps = " ".join(c.value for c in at.sidebar.caption)
    assert "1 of 2 match" in caps and not any(btn.key == f"pinhist_{a}" for btn in at.sidebar.button)
    assert any(btn.key == f"pinhist_{b}" for btn in at.sidebar.button)
    at.sidebar.text_input(key="history_filter").set_value("zzz").run()
    assert "Nothing matches" in " ".join(c.value for c in at.sidebar.caption)


# ------------------------------------------------------------------------------------------ 6. the status line

@pytest.mark.parametrize("hid, here, warned, saved, expected", [
    (12, True, False, True, "Solved · saved to history · #12"),
    (12, True, True, True, "Solved with warnings · saved to history · #12"),
    (5, False, False, True, "Opened from history · #5"),
    (12, True, False, False, "Solved · no longer in history"),
    (None, True, False, True, "Solved · not saved to history"),
])
def test_status_line_text(hid, here, warned, saved, expected):
    assert history.status_line(hid, here, warned, saved) == expected


def test_a_problem_opened_from_history_says_so_and_a_deleted_one_says_it_was_deleted(db):
    entry_id = seed()
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    next(b for b in at.sidebar.button if b.key == f"load_{entry_id}").click()
    at.run()
    assert not at.exception and f"Opened from history · #{entry_id}" in " ".join(c.value for c in at.caption)
    next(b for b in at.sidebar.button if b.key == f"delhist_{entry_id}").click()
    at.run()
    status = next(c.value for c in at.caption if "no longer in history" in c.value)
    assert status.endswith("Opened from history · no longer in history") and "#" not in status


# ------------------------------------------------------------------------------------------ 7. export captions

def test_a_missing_package_gets_a_short_caption_and_a_copyable_command():
    from modules import optional_deps
    docx = [d for d in optional_deps.DEPS if d.dist == "python-docx"]
    assert optional_deps.short_need(docx) == "Needs python-docx" and optional_deps.install_command(docx) == "pip install python-docx"
    assert optional_deps.short_need([]) == "" and optional_deps.install_command([]) == ""
    assert len(optional_deps.short_need(docx)) < len(optional_deps.install_hint(docx)) / 3


def test_the_share_dialog_shows_the_short_form_when_a_format_is_unavailable(monkeypatch):
    from modules import optional_deps, report_export
    monkeypatch.setattr(optional_deps, "available", lambda m: m != "docx")
    assert [d.dist for d in report_export.format_missing("docx")] == ["python-docx"]
    assert report_export.format_missing("md") == []


# ------------------------------------------------------------------------------------------ 5. the motion diagram's slider

def test_the_time_readout_has_clear_space_below_the_axis_title():
    import numpy as np
    from modules import plotter
    t = np.linspace(0, 6, 61)
    fig = plotter.build_motion_diagram(t, 8 * t + t ** 2, 8 + 2 * t, x_label="position", x_unit="m", t_unit="s")
    assert fig.layout.sliders[0].pad.t >= 40                                   # Plotly's default is 20: "t = 0" sat ~10 px under "time (s)"
    assert fig.layout.margin.b >= plotter.MOTION_SLIDER_PAD_TOP + 60            # tick labels + axis title, then the padding, then the track
    assert fig.layout.height - fig.layout.margin.t - fig.layout.margin.b == plotter.MOTION_PLOT_HEIGHT   # the panels themselves are no smaller


# ------------------------------------------------------------------------------------------ animation speed and reduced motion

def _play_ms(fig):
    return [b.args[1]["frame"]["duration"] for m in fig.layout.updatemenus for b in m.buttons if b.args and len(b.args) > 1 and isinstance(b.args[1], dict)]


def _animated():
    import numpy as np
    from modules import plotter
    t = np.linspace(0, 6, 61)
    return plotter.build_motion_diagram(t, 8 * t + t ** 2, 8 + 2 * t)


def test_speed_scales_the_play_duration_and_leaves_the_original_alone():
    from modules import motion_prefs as mp
    fig = _animated()
    base = _play_ms(fig)
    assert _play_ms(mp.apply(fig, 2.0))[0] == base[0] // 2 and _play_ms(mp.apply(fig, 0.5))[0] == base[0] * 2
    assert _play_ms(fig) == base                                              # a cached figure is never edited
    assert mp.apply(fig) is fig and mp.clamp_speed(99) == 4.0 and mp.clamp_speed("x") == 1.0 and mp.clamp_speed(0) == 0.25
    assert _play_ms(mp.apply(fig, 4.0))[1] == 0                               # Pause keeps a duration of 0


def test_reduce_motion_removes_play_but_keeps_every_frame_and_the_slider():
    from modules import motion_prefs as mp
    fig = _animated()
    calm = mp.apply(fig, 1.0, True)
    assert not mp.has_animation(calm) and len(calm.frames) == len(fig.frames) and len(calm.layout.sliders) == 1
    assert mp.has_animation(fig) and mp.has_animation(mp.apply(fig, 1.0, False))   # turning it off again brings Play back


def test_a_figure_without_frames_is_untouched_whatever_the_setting():
    import plotly.graph_objects as go
    from modules import motion_prefs as mp
    fig = go.Figure(go.Scatter(x=[0, 1], y=[0, 1]))
    assert mp.apply(fig, 4.0, True) is fig and not mp.has_animation(fig)


def test_show_figure_draws_with_the_chosen_speed_and_reduce_motion(monkeypatch):
    import streamlit as st
    from ui import a11y
    drawn = []
    monkeypatch.setattr(st, "plotly_chart", lambda fig, **k: drawn.append(fig))
    monkeypatch.setattr(a11y, "announce", lambda *a, **k: None)
    monkeypatch.setattr(st, "session_state", {a11y.SPEED_KEY: 2.0})
    fig = _animated()
    a11y.show_figure(fig)
    assert _play_ms(drawn[-1])[0] == _play_ms(fig)[0] // 2
    monkeypatch.setattr(st, "session_state", {a11y.REDUCE_MOTION_KEY: True})
    a11y.show_figure(fig)
    from modules import motion_prefs as mp
    assert not mp.has_animation(drawn[-1])


def test_the_advanced_settings_offer_reduce_motion_and_a_speed():
    at = AppTest.from_file(APP, default_timeout=200)
    at.run()
    assert not at.exception
    assert at.sidebar.checkbox(key="reduce_motion").value is False and at.sidebar.select_slider(key="animation_speed").value == 1.0
    at.sidebar.checkbox(key="reduce_motion").check().run()
    assert not at.exception and at.sidebar.select_slider(key="animation_speed").disabled


# ------------------------------------------------------------------------------------------ the pendulum's first-use spinner

def test_the_pendulum_explains_its_first_use_pause_and_only_then(monkeypatch):
    import streamlit as st
    from modules import dynamics as dy
    from ui import dynamics as ui_dy
    shown = []

    class Spin:
        def __init__(self, text, **k): shown.append(text)
        def __enter__(self): return self
        def __exit__(self, *a): return False

    derived = {"done": False}
    monkeypatch.setattr(st, "spinner", Spin)
    monkeypatch.setattr(dy, "pendulum_is_derived", lambda: derived["done"])
    monkeypatch.setattr(dy, "pendulum_sympy", lambda: derived.update(done=True))
    ui_dy._ensure_pendulum_derived()
    ui_dy._ensure_pendulum_derived()
    assert shown == [ui_dy.DERIVING_MESSAGE] and "first use only" in ui_dy.DERIVING_MESSAGE and derived["done"]


def test_pendulum_is_derived_tracks_the_cache():
    from modules import dynamics as dy
    dy.pendulum_sympy.cache_clear()
    assert dy.pendulum_is_derived() is False
