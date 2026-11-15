"""The circuits figures build and mark what they should."""
import numpy as np

from modules import circuits as ck
from modules import circuits_plots as cp


def test_transient_and_bode_figures():
    c = ck.RLC(10.0, 10e-3, 10e-6)
    fig = cp.transient_figure(ck.simulate_transient(c))
    assert [t.name for t in fig.data][:2] == ["v_C (simulated)", "closed form"] and "ζ = 0.1581" in fig.layout.title.text
    b = cp.bode_figure(ck.circuit_bode(c), c)
    assert b.layout.xaxis2.type == "log" and any(t.name == "peak of v_R" for t in b.data)
    assert len(cp.pole_zero_figure(c.poles).data) == 1


def test_control_figures_mark_the_band_the_margins_and_the_locus():
    plant = ck.make_plant("First order with delay (FOPDT)", K=2.0, tau=5.0, theta=1.0)
    la = ck.analyse_loop(plant, 1.0, 0.2, 0.0, 0.0)
    assert any(t.name == "±2 % band" for t in cp.step_figure(la).data)
    names = [t.name for t in cp.loop_bode_figure(la).data]
    assert any("phase margin" in n for n in names) and any("gain margin" in n for n in names)
    rl = cp.root_locus_figure(plant, 1.5)
    assert rl.data[-1].name == "poles at Kp = 1.5" and np.isfinite(rl.data[0].x).any()
    bad = ck.analyse_loop(plant, 10.0, 0.0, 0.0, 0.0)
    assert "UNSTABLE" in cp.step_figure(bad).layout.title.text
