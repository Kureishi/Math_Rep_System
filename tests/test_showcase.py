"""The visual showpieces: each picture is checked against a number that must come out right."""
import math

import numpy as np
import pytest

from modules import gpu
from modules import showcase as sc
from modules import showcase_plots as sp


# ------------------------------------------------------------------------------------------ black-hole image

@pytest.fixture(scope="module")
def image():
    return sc.render_lens_image(160)


def test_the_shadow_radius_matches_the_exact_critical_impact_parameter(image):
    assert image.shadow_error is not None and image.shadow_error < 0.01          # 3 sqrt(3) M, to within one percent
    assert image.rgb.shape == (160, 160, 3) and image.rgb.dtype == np.uint8


def test_the_shadow_is_black_and_centred_and_the_sky_outside_is_lit(image):
    c = image.n // 2
    assert image.rgb[c, c].sum() == 0 and image.rgb[0, 0].sum() > 0
    assert (image.status == sc.CAPTURED).sum() > 1000


def test_the_hole_actually_distorts_the_sky(image):
    """Same sky, with and without the hole: far from the centre they agree closely, near the shadow they do not."""
    diff = np.abs(image.rgb.astype(int) - image.rgb_flat.astype(int)).sum(axis=2)
    n = image.n
    yy, xx = np.mgrid[:n, :n]
    rho = np.hypot(xx - n / 2, yy - n / 2) / (n / 2)
    assert diff[rho > 0.95].mean() < diff[(rho > 0.4) & (rho < 0.7)].mean()


def test_critical_impact_parameter_separates_capture_from_escape():
    D = 100.0
    ang = lambda b: math.asin(b * math.sqrt(1 - 2 / D) / D)
    _, status, _ = sc.trace_rays(np.array([ang(5.0), ang(5.4)]), 1.0, D)
    assert list(status) == [sc.CAPTURED, sc.ESCAPED]                              # b_c = 5.196


def test_weak_field_deflection_agrees_with_the_first_order_formula():
    D, b = 2000.0, 200.0
    a = math.asin(b * math.sqrt(1 - 2 / D) / D)
    phi, status, _ = sc.trace_rays(np.array([a]), 1.0, D)
    measured = phi[0] - (math.pi - math.asin(b / D))
    assert status[0] == sc.ESCAPED and measured == pytest.approx(sc.weak_deflection(b, 1.0, D), rel=0.03)


def test_rays_just_outside_the_critical_value_wind_around_more_than_once():
    D = 100.0
    a = math.asin(5.2 * math.sqrt(1 - 2 / D) / D)
    phi, status, _ = sc.trace_rays(np.array([a]), 1.0, D)
    assert status[0] in (sc.ESCAPED, sc.WINDING) and (status[0] == sc.WINDING or phi[0] > 1.5 * math.pi)


def test_mass_scales_the_shadow():
    big, small = sc.render_lens_image(120, M=1.0), sc.render_lens_image(120, M=0.5)
    assert big.shadow_px_expected == pytest.approx(2 * small.shadow_px_expected, rel=0.05)


def test_the_gpu_path_gives_the_identical_image():
    cpu = sc.render_lens_image(96)
    with gpu.pretend_gpu(gpu.Backend(np, "Pretend GPU", True, "numpy")):
        dev = sc.render_lens_image(96)
    assert dev.backend == "Pretend GPU" and np.array_equal(cpu.rgb, dev.rgb)


def test_a_failing_gpu_falls_back_to_the_same_image():
    class Broken:
        def __getattr__(self, name):
            raise RuntimeError("out of memory")
    cpu = sc.render_lens_image(96)
    with gpu.pretend_gpu(gpu.Backend(Broken(), "Broken", True, "numpy")):
        again = sc.render_lens_image(96)
    assert again.backend == "CPU (NumPy)" and np.array_equal(cpu.rgb, again.rgb)


@pytest.mark.parametrize("kwargs, message", [
    (dict(n=10), "image size"), (dict(fov_deg=1), "field of view"), (dict(D=2.0), "outside the horizon"),
])
def test_bad_view_parameters_are_refused(kwargs, message):
    with pytest.raises(sc.ShowcaseError, match=message):
        sc.render_lens_image(**{"n": 64, **kwargs})


def test_a_shadow_wider_than_the_view_is_noted_rather_than_mismeasured():
    img = sc.render_lens_image(64, fov_deg=8.0)
    assert img.shadow_px_measured is None and any("larger than the field of view" in n for n in img.notes)


def test_lens_figures_build():
    img = sc.render_lens_image(64)
    assert len(sp.lens_figure(img).layout.shapes) == 1 and len(sp.lens_figure(img, with_hole=False).layout.shapes) == 0


# ------------------------------------------------------------------------------------------ orbit and embedding

@pytest.mark.parametrize("p, e", [(60.0, 0.3), (200.0, 0.2)])
def test_perihelion_advance_matches_the_known_expansion(p, e):
    o = sc.orbit(p, e, n_orbits=4)
    second_order = 6 * math.pi / p * (1 + (18 + e * e) / (4 * p))
    assert o.precession == pytest.approx(second_order, rel=0.015)
    assert o.precession == pytest.approx(o.precession_weak_field, rel=0.12)


def test_the_conserved_quantity_is_constant_along_the_orbit():
    assert sc.orbit(30.0, 0.5, n_orbits=3).drift < 1e-9


def test_the_orbit_starts_at_periapsis_and_stays_between_peri_and_apoapsis():
    o = sc.orbit(40.0, 0.4)
    assert o.r[0] == pytest.approx(40.0 / 1.4, rel=1e-9)
    assert o.r.min() == pytest.approx(40.0 / 1.4, rel=1e-3) and o.r.max() == pytest.approx(40.0 / 0.6, rel=1e-2)


def test_the_scale_of_M_changes_lengths_but_not_the_shape():
    a, b = sc.orbit(50.0, 0.2, M=1.0), sc.orbit(50.0, 0.2, M=3.0)
    assert np.allclose(b.r, 3 * a.r, rtol=1e-9) and b.precession == pytest.approx(a.precession, rel=1e-9)


@pytest.mark.parametrize("p, e, message", [(6.0, 0.0, "plunges"), (7.0, 0.9, "plunges"), (20.0, 1.2, "eccentricity"), (20.0, -0.1, "eccentricity")])
def test_orbits_that_are_not_stable_and_bound_are_refused(p, e, message):
    with pytest.raises(sc.ShowcaseError, match=message):
        sc.orbit(p, e)


def test_flamms_paraboloid_has_the_spatial_geometry_of_the_hole():
    r = np.linspace(4.0, 40, 4000)                    # away from the horizon, where the slope is infinite
    slope = np.gradient(sc.flamm_z(r, 1.0), r)
    assert np.allclose(1 + slope ** 2, r / (r - 2.0), rtol=2e-3)                  # 1 + z'^2 = g_rr
    assert sc.flamm_z(2.0, 1.0) == 0.0


def test_the_orbit_figure_builds_with_a_body_and_frames():
    fig = sp.orbit_figure(sc.orbit(40.0, 0.3, n_orbits=1))
    assert len(fig.data) == 3 and len(fig.frames) > 10


# ------------------------------------------------------------------------------------------ critical movie

@pytest.fixture(scope="module")
def movie():
    return sc.critical_movie(L=24, sweeps=120, every=6, seed=3)


def test_the_movie_runs_three_temperatures_around_the_exact_tc(movie):
    assert movie.tc == pytest.approx(2 / math.log(1 + math.sqrt(2)))
    assert movie.temperatures == pytest.approx([0.85 * movie.tc, movie.tc, 1.25 * movie.tc])
    assert len(movie.runs) == 3 and "critical" in movie.labels[1]


def test_below_tc_orders_and_above_tc_does_not(movie):
    cold, hot = movie.runs[0], movie.runs[2]
    assert cold.order[-30:].mean() > 0.8 and hot.order[-30:].mean() < 0.4


def test_movie_figures_build(movie):
    fig = sp.critical_figure(movie)
    assert len(fig.data) == 3 and len(fig.frames) > 5 and len(sp.magnetisation_figure(movie).data) == 3


@pytest.mark.parametrize("kwargs", [dict(L=8), dict(sweeps=5), dict(sweeps=3000, every=2)])
def test_bad_movie_settings_are_refused(kwargs):
    with pytest.raises(sc.ShowcaseError):
        sc.critical_movie(**kwargs)
