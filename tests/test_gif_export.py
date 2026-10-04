"""
Tests for the "lower effort" additions: modules.plotter.add_camera_rotation
(an auto-orbit animation wrapper for any 3D Plotly figure) and the three
new GIF-export functions in modules/plot_snapshot.py -- the animated
counterpart to that module's existing static PNG/SVG/PDF snapshots.
"""
import io
import math

import numpy as np
import pytest
import sympy as sp
from PIL import Image

from modules.equation_engine import Equation
from modules.plotter import add_camera_rotation, build_surface_plot
from modules.plot_snapshot import (
    snapshot_cobweb_gif, snapshot_motion_diagram_gif, snapshot_rotating_surface_gif,
)

x, y, z = sp.symbols("x y z")


def _surface_eq():
    return Equation(name="e0", raw_expression="Eq(z, x**2+y**2)", derivation="", kind="equation",
                      sympy_eq=sp.Eq(z, x ** 2 + y ** 2))


# ---------------------------------------------------------------- add_camera_rotation

def test_add_camera_rotation_produces_one_frame_per_step():
    fig = build_surface_plot(_surface_eq(), "x", "y", {}, (-3, 3), (-3, 3), z_target="z", resolution=15)
    rotated = add_camera_rotation(fig, n_frames=24)
    assert len(rotated.frames) == 24


def test_add_camera_rotation_keeps_elevation_and_radius_constant():
    """The camera should sweep in a circle at fixed height and distance --
    only the azimuth (x/y mix) changes, z stays put, and x^2+y^2 (the
    horizontal distance from the scene center) stays constant across
    every frame."""
    fig = build_surface_plot(_surface_eq(), "x", "y", {}, (-3, 3), (-3, 3), z_target="z", resolution=15)
    rotated = add_camera_rotation(fig, n_frames=20, elevation_deg=25.0, radius=1.9)
    zs = [f.layout.scene.camera.eye["z"] for f in rotated.frames]
    horiz = [math.hypot(f.layout.scene.camera.eye["x"], f.layout.scene.camera.eye["y"])
              for f in rotated.frames]
    assert all(abs(zi - zs[0]) < 1e-9 for zi in zs)
    assert all(abs(hi - horiz[0]) < 1e-9 for hi in horiz)


def test_add_camera_rotation_adds_play_pause_buttons():
    fig = build_surface_plot(_surface_eq(), "x", "y", {}, (-3, 3), (-3, 3), z_target="z", resolution=15)
    rotated = add_camera_rotation(fig, n_frames=10)
    assert len(rotated.layout.updatemenus) >= 1


def test_add_camera_rotation_preserves_existing_updatemenus():
    """A figure that already has its own updatemenus (unlikely for a plain
    surface plot today, but the function's docstring promises this works
    for any 3D scene) shouldn't have them clobbered."""
    fig = build_surface_plot(_surface_eq(), "x", "y", {}, (-3, 3), (-3, 3), z_target="z", resolution=15)
    fig.update_layout(updatemenus=[dict(type="buttons", buttons=[dict(label="existing", method="skip")])])
    rotated = add_camera_rotation(fig, n_frames=10)
    labels = [b["label"] for menu in rotated.layout.updatemenus for b in menu["buttons"]]
    assert "existing" in labels


# ---------------------------------------------------------------- snapshot_rotating_surface_gif

def test_rotating_surface_gif_is_a_valid_multi_frame_gif():
    data = snapshot_rotating_surface_gif(_surface_eq(), "x", "y", {}, (-3, 3), (-3, 3),
                                            z_target="z", resolution=10, n_frames=3)
    assert data[:6] == b"GIF89a"
    img = Image.open(io.BytesIO(data))
    assert img.n_frames == 3


def test_rotating_surface_gif_raises_on_unparsed_equation():
    bad = Equation(name="bad", raw_expression="garbage", derivation="", kind="equation", sympy_eq=None)
    with pytest.raises(ValueError, match="no parsed sympy expression"):
        snapshot_rotating_surface_gif(bad, "x", "y", {}, (-3, 3), (-3, 3))


# ---------------------------------------------------------------- snapshot_motion_diagram_gif

def test_motion_diagram_gif_is_a_valid_multi_frame_gif():
    ts = np.linspace(0, 6, 60)
    xs = 8 * ts + 0.5 * 2 * ts ** 2
    vs = 8 + 2 * ts
    data = snapshot_motion_diagram_gif(ts, xs, vs, x_label="position", x_unit="m", t_unit="s", n_frames=4)
    assert data[:6] == b"GIF89a"
    img = Image.open(io.BytesIO(data))
    assert img.n_frames == 4


def test_motion_diagram_gif_handles_negative_velocity():
    ts = np.linspace(0, 10, 60)
    xs = 20 * ts - 0.5 * 5 * ts ** 2
    vs = 20 - 5 * ts
    data = snapshot_motion_diagram_gif(ts, xs, vs, n_frames=3)
    assert data[:6] == b"GIF89a"


# ---------------------------------------------------------------- snapshot_cobweb_gif

def test_cobweb_gif_is_a_valid_multi_frame_gif():
    g = lambda xx: 2.8 * xx * (1 - xx)
    data = snapshot_cobweb_gif(g, 0.2, (0, 1), n_steps=4)
    assert data[:6] == b"GIF89a"
    img = Image.open(io.BytesIO(data))
    assert img.n_frames == 5  # n_steps + 1


def test_cobweb_gif_linear_map():
    g = lambda xx: xx + 500
    data = snapshot_cobweb_gif(g, 0, (0, 5000), n_steps=2)
    assert data[:6] == b"GIF89a"
