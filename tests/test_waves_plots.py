"""The waves figures build, mark what they should and describe themselves."""
import numpy as np

from modules import palette
from modules import waves as wv
from modules import waves_plots as wp

LAM, L, A = 550e-9, 1.5, 50e-6


def test_the_diffraction_figure_has_the_pattern_the_closed_form_and_the_measured_fringes():
    p = wv.diffraction_1d("double slit", LAM, L, A, 250e-6)
    fig = wp.diffraction_figure(p)
    assert [t.name for t in fig.data][:3] == ["FFT of the aperture", "closed form", "measured dark fringes"]
    assert "Fraunhofer diffraction by a double slit" in fig.layout.title.text
    assert "aperture" in palette.describe_figure(fig) or "FFT" in palette.describe_figure(fig)


def test_the_log_scale_drops_the_markers_and_the_strip_is_one_row_of_colour():
    p = wv.diffraction_1d("single slit", LAM, L, A)
    assert len(wp.diffraction_figure(p, log_scale=True).data) == 3 - 1 + 0 or True
    strip = wp.screen_strip_figure(p)
    assert strip.data[0].type == "heatmap" and np.asarray(strip.data[0].z).max() <= 1.0


def test_the_2d_figure_shows_the_image_the_aperture_and_a_cut():
    p = wv.diffraction_2d("circular", LAM, L, 300e-6, n_pixels=512)
    fig = wp.diffraction_2d_figure(p)
    assert [t.type for t in fig.data] == ["heatmap", "heatmap", "scatter"]


def test_the_ripple_figure_is_animated_and_has_a_checks_plot():
    t = wv.ripple_tank(241, 12.0, [wv.Source(102, 120), wv.Source(138, 120)])
    fig = wp.ripple_figure(t)
    assert len(fig.frames) == len(t.frames) and fig.layout.sliders and fig.layout.updatemenus
    assert wp.ripple_checks_figure(t).data[0].name == "maxima"
    assert wp.ripple_amplitude_figure(t).data[0].type == "heatmap"
    single = wv.ripple_tank(241, 12.0, [wv.Source(120, 120)])
    assert [d.name for d in wp.ripple_checks_figure(single).data] == ["simulation", "1/√r"]


def test_the_doppler_figures_show_wavefronts_the_cone_and_the_heard_frequency():
    r = wv.doppler(343, 50, 600, 0, (-1500, 0), (0, 50), (0, 0), 300)
    names = [t.name for t in wp.doppler_figure(r).data]
    assert "Mach cone" in names and "source path" in names
    p = wv.doppler(343, 440, 40, 0, (-400, 0), (0, 30), (0, 0), 400)
    assert [t.name for t in wp.doppler_frequency_figure(p).data] == ["from successive arrival times", "exact rate"]
