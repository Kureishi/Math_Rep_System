"""Tests for modules/sm_plots.py: every figure builds from real results and carries what it should."""
import math

import numpy as np
import plotly.graph_objects as go
import pytest

from modules import sm_distributions as sd, sm_lattice as sl, sm_meanfield as mf, sm_partition as sp_, sm_plots, sm_randommatrix as rm, sm_stochastic as ss


def test_thermo_and_population_figures():
    spec = sp_.two_level(1.0)
    th = sp_.thermodynamics(spec, np.geomspace(0.1, 10, 40))
    fig = sm_plots.thermo_figure(th, spec)
    assert len(fig.data) == 5 + 4                                           # 5 panels + 4 closed-form overlays
    assert len(sm_plots.thermo_figure(sp_.thermodynamics(sp_.rigid_rotor(1.0), np.geomspace(0.1, 10, 40)), sp_.rigid_rotor(1.0)).data) == 5
    pops = sm_plots.populations_figure(th)
    assert len(pops.data) == 2


def test_lattice_figures():
    scans = {L: sl.scan_temperatures(sl.LatticeModel("ising", L), [2.0, 2.4], 50, 300, seed=1) for L in (4, 6)}
    fig = sm_plots.lattice_scan_figure(scans, 2.269)
    assert len(fig.data) == 12 and any(s.type == "line" for s in fig.layout.shapes)
    run = sl.run_single(sl.LatticeModel("ising", 8), 2.0, 60, 6, seed=1)
    anim = sm_plots.lattice_animation_figure(run, max_frames=20)
    assert 2 <= len(anim.frames) <= 20 and anim.layout.updatemenus
    for model in ("potts", "xy"):
        r = sl.run_single(sl.LatticeModel(model, 6), 1.0, 20, 5, seed=1)
        assert len(sm_plots.lattice_animation_figure(r).frames) >= 2
    assert len(sm_plots.series_figure(run).data) == 2
    fit = sl.finite_size_scaling(scans, 2.269)
    assert len(sm_plots.fss_figure(fit).data) == 2


def test_meanfield_figures():
    cv = mf.compare_curves(np.linspace(0.5, 5, 20))
    fig = sm_plots.meanfield_figure(cv, 4.0, 2.885, 2.269)
    assert len(fig.data) == 8
    mc = sl.scan_temperatures(sl.LatticeModel("ising", 4), [2.0, 3.0], 50, 200, seed=1)
    assert len(sm_plots.meanfield_figure(cv, 4.0, 2.885, 2.269, mc).data) == 11
    m = np.linspace(-0.9, 0.9, 50)
    assert len(sm_plots.landau_figure(m, {1.0: mf.mf_landau_free_energy(m, 1.0), 5.0: mf.mf_landau_free_energy(m, 5.0)}).data) == 2


def test_distribution_figures():
    eps = np.linspace(0.01, 5, 100)
    fig = sm_plots.occupation_figure(eps, -0.5, 1.0)
    assert [t.name for t in fig.data] == ["MB", "FD", "BE"]
    assert len(sm_plots.occupation_figure(eps, 1.0, 1.0).data[2].x) < len(eps)               # BE is drawn only above mu
    temps = np.linspace(0.5, 3, 10)
    assert len(sm_plots.mu_figure(temps, sd.mu_curves(0.3, temps), 1.0).data) == 3
    x = np.linspace(1e-3, 1, 10)
    assert len(sm_plots.eos_figure(x, sd.equation_of_state(x)).data) == 3
    assert len(sm_plots.mb_figure(np.linspace(0, 5, 50), 1.0, np.abs(np.random.default_rng(0).normal(size=100))).data) == 2


def test_stochastic_figures():
    free = ss.preset_potential("free")
    r = ss.simulate_underdamped(free, 1.0, 1.0, 1.0, 0.0, 0.01, 2.0, 100, 20, seed=1)
    assert len(sm_plots.paths_figure(r, 5).data) == 5
    msd = sm_plots.msd_figure(r)
    assert len(msd.data) == 4 and msd.layout.xaxis.type == "log"
    pot = ss.preset_potential("harmonic", k=2.0)
    ro = ss.simulate_overdamped(pot, 1.0, 1.0, 1.0, 0.01, 1.0, 100, 10, seed=1)
    assert len(sm_plots.msd_figure(ro).data) == 1                                           # no exact curve for a trap
    sol = ss.solve_fokker_planck(pot, 1.0, 1.0, (-4, 4), 200, 1.0, 0.3, 1.0, 0.01, 10)
    ex = {j: ss.ou_exact(sol.x, float(sol.t[j]), 1.0, 0.3, 1.0, 1.0, 2.0) for j in range(len(sol.t))}
    fig = sm_plots.fp_figure(sol, ro, ex)
    assert any(t.type == "bar" for t in fig.data) and any(t.name == "exact Gaussian" for t in fig.data)


def test_random_matrix_figures():
    rep = rm.analyse("GOE", 100, 10, seed=1)
    assert len(sm_plots.density_figure(rep.eigenvalues, "GOE").data) == 2
    assert len(sm_plots.spacing_figure(rep.spacings, "GOE").data) == 4
    ck, eigs, _ = rm.verify_wishart(60, 120, 3, seed=1)
    assert len(sm_plots.wishart_figure(eigs, 0.5).data) == 2
