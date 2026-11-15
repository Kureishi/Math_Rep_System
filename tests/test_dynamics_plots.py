"""The dynamics figures build and are animated where they should be."""
import numpy as np

from modules import dynamics as dy
from modules import dynamics_plots as dp
from modules import palette


def test_orbit_figure_is_animated_for_a_few_bodies_and_static_for_a_cluster():
    r = dy.run_nbody(dy.preset_system("figure-eight"), "leapfrog", 2e-3, 3.0)
    fig = dp.orbit_figure(r)
    assert fig.frames and fig.layout.sliders and len(fig.data) == 6
    c = dy.run_nbody(dy.preset_system("cluster", n=30, seed=1), "leapfrog", 4e-3, 0.4)
    assert not dp.orbit_figure(c).frames


def test_conservation_and_order_figures():
    r = dy.run_nbody(dy.preset_system("kepler"), "leapfrog", 5e-3, 6.0)
    assert len(dp.conservation_figure(r).data) == 3
    studies, _ = dy.order_study(dy.preset_system("kepler"), dts=(0.04, 0.02, 0.01), integrators=("leapfrog", "rk4"))
    fig = dp.order_study_figure(studies)
    assert fig.layout.xaxis.type == "log" and "slope" in fig.data[0].name


def test_pendulum_lorenz_and_chaos_figures_describe_themselves():
    pr = dy.run_pendulum(dy.PendulumParams(), (2.0, 2.5, 0.0, 0.0), 3.0, n_points=120)
    assert dp.pendulum_figure(pr).frames and "energy" in dp.pendulum_energy_figure(pr).layout.title.text
    ts, X = dy.run_lorenz(dy.LorenzParams(), t_end=5.0, n_points=300)
    assert dp.lorenz_figure(ts, X).data[0].type == "scatter3d"
    ly = dy.lyapunov_spectrum("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=16, t_run=4.0, seed=1)
    assert "Lyapunov exponents" in palette.describe_figure(dp.lyapunov_figure(ly)) or dp.lyapunov_figure(ly).data[0].type == "bar"
    curve = dy.divergence_curve("lorenz", dy.LorenzParams(), np.array([1.0, 1.0, 1.0]), members=16, t_end=6.0, seed=1)
    assert dp.divergence_figure(curve).data[1].name == "mean over the ensemble"
    fm = dy.flip_map(dy.PendulumParams(), n=16, t_end=2.0, dt=0.01)
    assert dp.flip_map_figure(fm).data[0].type == "heatmap"
