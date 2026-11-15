"""The three simulation labs are registered, reachable by asking, render in the app, and hand trajectories to equation discovery."""
from pathlib import Path

import numpy as np
import pytest
from streamlit.testing.v1 import AppTest

from modules import mode_registry
from modules import next_steps as ns
from modules import router as rt
from modules.command_palette import MODE_LABELS

WAVES, DYNAMICS, CIRCUITS = "🌊 Waves & optics lab", "🌀 Dynamics & chaos lab", "🔌 Circuits & control lab"


def test_the_labs_are_registered_and_need_no_language_model():
    reg = mode_registry.discover()
    for label in (WAVES, DYNAMICS, CIRCUITS):
        assert label in reg.labels and label in MODE_LABELS and reg.spec(label).needs_client is False


@pytest.mark.parametrize("query, mode", [
    ("double slit interference fringe spacing", WAVES), ("fraunhofer diffraction of a circular aperture airy disk", WAVES), ("ripple tank with two sources", WAVES),
    ("doppler effect of an ambulance siren", WAVES), ("mach cone of a supersonic jet", WAVES),
    ("figure-eight three-body orbit with a symplectic integrator", DYNAMICS), ("double pendulum chaos", DYNAMICS), ("lyapunov exponent of the lorenz butterfly", DYNAMICS),
    ("leapfrog versus rk4 on the figure-eight orbit", DYNAMICS),
    ("series rlc circuit damping ratio and resonant frequency", CIRCUITS), ("bode plot of a transfer function", CIRCUITS), ("pid tuning with ziegler-nichols", CIRCUITS),
    ("phase margin and gain margin of the loop", CIRCUITS),
])
def test_the_labs_are_reachable_by_asking(query, mode):
    assert rt.route(query).mode == mode


@pytest.mark.parametrize("query", ["A car accelerates from 8 m/s to 20 m/s in 6 s. How far does it go?", "pH of 0.01 M HCl", "partition function of a two level system"])
def test_existing_requests_are_not_stolen_by_the_labs(query):
    assert rt.route(query).mode not in (WAVES, DYNAMICS, CIRCUITS)


def test_a_trajectory_is_handed_to_equation_discovery_with_the_data_prefilled():
    t = np.linspace(0, 5, 1000)
    steps = ns.for_trajectory(t, np.stack([np.sin(t), np.cos(t)], axis=1), ["x", "y"])
    assert len(steps) == 1 and steps[0].mode == ns.mode_for(ns.PAGE_DISCOVER)
    paste = steps[0].prefill["discover_paste"]
    assert paste.splitlines()[0] == "t,x,y" and len(paste.splitlines()) == 401
    assert steps[0].prefill["discover_source"] == "Paste or upload data"


def test_a_malformed_trajectory_hands_off_nothing():
    assert ns.for_trajectory(np.arange(3.0), np.zeros((4, 2)), ["a", "b"]) == [] and ns.for_trajectory(np.arange(3.0), np.zeros((3, 2)), ["a"]) == []


def open_lab(label, timeout=240):
    at = AppTest.from_file(str(Path(__file__).parent.parent / "app.py"), default_timeout=timeout)
    at.run()
    at.sidebar.radio[0].set_value(label)
    at.run()
    assert not at.exception, [str(e.value)[:300] for e in at.exception]
    return at


def verdicts(at):
    return [m.value for m in at.success if "checks passed" in m.value]


def test_the_waves_lab_renders_every_tab_with_passing_checks():
    at = open_lab(WAVES)
    assert len(at.tabs) == 4 and len(verdicts(at)) >= 3 and not [e for e in at.error if "checks" in e.value]


def test_the_waves_lab_explains_bad_input_instead_of_crashing():
    at = open_lab(WAVES)
    at.number_input(key="wv_wl").set_value(550.0)
    at.radio(key="wv_kind").set_value("double slit")
    at.number_input(key="wv_d").set_value(40.0)               # closer than 1.2 x the 50 µm slit width
    at.run()
    assert not at.exception and any("1.2 times" in e.value for e in at.error)


def test_the_dynamics_lab_renders_and_a_chaotic_preset_fails_its_energy_check_visibly():
    at = open_lab(DYNAMICS)
    assert len(at.tabs) == 3 and len(verdicts(at)) >= 3
    at.selectbox(key="dy_preset").set_value("Pythagorean 3-4-5 (chaotic)")
    at.run()
    assert not at.exception and any("checks passed" in e.value for e in at.error)


def test_the_circuits_lab_renders_and_an_unstable_loop_is_flagged():
    at = open_lab(CIRCUITS)
    assert len(at.tabs) == 3 and len(verdicts(at)) >= 3
    at.selectbox(key="ck_plant_kind").set_value("First order with delay (FOPDT)")
    at.run()
    at.number_input(key="ck_kp").set_value(10.0)
    at.number_input(key="ck_ki").set_value(0.0)
    at.run()
    assert not at.exception and any("UNSTABLE" in i.value for i in at.info)


def test_the_circuits_lab_tuning_buttons_fill_in_the_gains():
    at = open_lab(CIRCUITS)
    at.selectbox(key="ck_plant_kind").set_value("First order with delay (FOPDT)")
    at.run()
    at.button(key="ck_simc").click()
    at.run()
    assert not at.exception and at.number_input(key="ck_kp").value == pytest.approx(1.256, rel=0.05) and any("SIMC" in i.value for i in at.info)
