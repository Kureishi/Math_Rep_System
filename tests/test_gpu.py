"""GPU backend selection, the CPU fallback, and the GPU code paths.

There is no GPU in CI, so the GPU paths are exercised with `gpu.pretend_gpu`: NumPy flagged as the device. That runs
exactly the code a CuPy run would (device copies, the device generator, batched solves, transfers back) and so
checks the plumbing and the statistics; what it cannot check is CuPy itself -- see modules/gpu.py.
"""
import numpy as np
import pytest

from modules import gpu
from modules import sm_lattice as lat
from modules import sm_randommatrix as rm
from modules import sm_stochastic as st
from modules.sm_common import StatMechError

FAKE = gpu.Backend(np, "Pretend GPU", True, "numpy")


class _Broken:
    """An array module whose every use fails, like a GPU that has run out of memory."""
    def __getattr__(self, name):
        raise RuntimeError("out of memory")


BROKEN = gpu.Backend(_Broken(), "Broken GPU", True, "numpy")


# ------------------------------------------------------------------------------------------ selection

def test_mode_is_validated_and_defaults_to_auto():
    with pytest.raises(ValueError):
        gpu.set_mode("sometimes")
    assert gpu.MODES == ("auto", "on", "off")


@pytest.mark.parametrize("mode", gpu.MODES)
def test_without_a_gpu_everything_is_cpu_in_every_mode(mode):
    with gpu.pretend_gpu(gpu.CPU, mode):
        assert gpu.select(10 ** 9, "langevin_paths") is gpu.CPU
        assert "CPU only" in gpu.describe() or mode == "off"


def test_auto_uses_the_gpu_only_for_big_jobs_on_only_with_a_gpu_and_off_never():
    big, small = gpu.THRESHOLDS["langevin_paths"], gpu.THRESHOLDS["langevin_paths"] - 1
    with gpu.pretend_gpu(FAKE, "auto"):
        assert gpu.select(big, "langevin_paths") is FAKE and gpu.select(small, "langevin_paths") is gpu.CPU
        assert "Pretend GPU" in gpu.describe()
    with gpu.pretend_gpu(FAKE, "on"):
        assert gpu.select(1, "langevin_paths") is FAKE
    with gpu.pretend_gpu(FAKE, "off"):
        assert gpu.select(big, "langevin_paths") is gpu.CPU and "off" in gpu.describe()


def test_the_mode_is_restored_after_a_pretend_block():
    before = gpu.get_mode()
    with gpu.pretend_gpu(FAKE, "on"):
        pass
    assert gpu.get_mode() == before


def test_attempt_falls_back_on_failure_but_passes_the_users_errors_through():
    assert gpu.attempt(FAKE, lambda: (_ for _ in ()).throw(RuntimeError("boom")), lambda: "cpu") == "cpu"
    assert gpu.attempt(gpu.CPU, lambda: "gpu", lambda: "cpu") == "cpu"
    assert gpu.attempt(FAKE, lambda: "gpu", lambda: "cpu") == "gpu"
    with pytest.raises(StatMechError):
        gpu.attempt(FAKE, lambda: (_ for _ in ()).throw(StatMechError("bad input")), lambda: "cpu", (StatMechError,))


def test_to_host_and_on_device_copy_rather_than_alias():
    a = np.arange(4.0)
    assert gpu.to_host(a) is a
    b = gpu.on_device(FAKE, a)
    b[0] = 99
    assert a[0] == 0


# ------------------------------------------------------------------------------------------ Langevin

def test_langevin_on_the_gpu_path_gets_the_physics_right_and_says_which_backend_ran():
    pot = st.preset_potential("harmonic", k=2.0)
    with gpu.pretend_gpu(FAKE):
        over = st.simulate_overdamped(pot, 0.5, 1.0, 0.0, 0.01, 12.0, 4000, seed=3)
        under = st.simulate_underdamped(pot, 0.5, 1.0, 1.0, 0.0, 0.01, 12.0, 4000, seed=3)
    assert over.backend == under.backend == "Pretend GPU"
    assert np.var(over.x[-1]) == pytest.approx(0.5 / 2.0, rel=0.1)            # <x^2> = kT/k
    assert np.var(under.x[-1]) == pytest.approx(0.5 / 2.0, rel=0.1)
    assert np.var(under.v[-1]) == pytest.approx(0.5, rel=0.1)                 # <v^2> = kT/m
    assert isinstance(over.x, np.ndarray) and over.x.shape == (len(over.t), 4000)


def test_a_custom_potential_is_compiled_for_the_device_too():
    pot = st.potential_from_text("a*cos(x)", {"a": 1.0})
    with gpu.pretend_gpu(FAKE):
        res = st.simulate_overdamped(pot, 0.3, 1.0, 0.0, 0.01, 2.0, 500, seed=1)
    assert res.backend == "Pretend GPU" and np.all(np.isfinite(res.x))


def test_a_failing_gpu_run_is_repeated_on_the_cpu_with_the_identical_result():
    pot = st.preset_potential("double_well")
    cpu = st.simulate_underdamped(pot, 0.4, 1.0, 1.0, 1.0, 0.01, 1.0, 200, seed=11)
    with gpu.pretend_gpu(BROKEN):
        again = st.simulate_underdamped(pot, 0.4, 1.0, 1.0, 1.0, 0.01, 1.0, 200, seed=11)
    assert again.backend == "CPU (NumPy)" and np.array_equal(cpu.x, again.x) and np.array_equal(cpu.v, again.v)


def test_bad_input_is_reported_not_swallowed_by_the_fallback():
    with gpu.pretend_gpu(FAKE):
        with pytest.raises(StatMechError):
            st.simulate_overdamped(st.preset_potential("free"), -1.0, 1.0, 0.0, 0.01, 1.0, 100)


# ------------------------------------------------------------------------------------------ lattice

def test_ising_on_the_gpu_path_orders_below_tc_and_disorders_above():
    model = lat.LatticeModel("ising", 12)
    with gpu.pretend_gpu(FAKE):
        cold = lat.run_single(model, 1.2, 200, 50, seed=5)
        hot = lat.run_single(model, 6.0, 200, 50, seed=5)
    assert cold.order[-50:].mean() > 0.9 and hot.order[-50:].mean() < 0.3
    assert all(isinstance(frame, np.ndarray) for _, frame in cold.snapshots) and len(cold.snapshots) == 5


@pytest.mark.parametrize("kind", ["ising", "potts", "xy"])
def test_run_point_on_the_gpu_path_returns_a_host_state_and_leaves_the_input_alone(kind):
    model = lat.LatticeModel(kind, 8, q=3) if kind == "potts" else lat.LatticeModel(kind, 8)
    rng = np.random.default_rng(2)
    start = lat.initial_state(model, np.random.default_rng(9), ordered=True)
    keep = start.copy()
    with gpu.pretend_gpu(FAKE):
        pt, final = lat.run_point(model, 1.0, rng, 20, 100, state=start, seed=2)
    assert pt.backend == "Pretend GPU" and isinstance(final, np.ndarray) and final.shape == start.shape
    assert np.array_equal(start, keep)


def test_a_failing_gpu_lattice_run_gives_the_cpu_result_exactly():
    model = lat.LatticeModel("ising", 8)
    cpu_pt, cpu_state = lat.run_point(model, 2.0, np.random.default_rng(4), 20, 100, seed=4)
    with gpu.pretend_gpu(BROKEN):
        pt, state = lat.run_point(model, 2.0, np.random.default_rng(4), 20, 100, seed=4)
    assert pt.backend == "CPU (NumPy)" and pt.e == cpu_pt.e and pt.m == cpu_pt.m and np.array_equal(state, cpu_state)


def test_the_gpu_path_does_not_consume_the_callers_generator():
    model = lat.LatticeModel("ising", 6)
    rng, twin = np.random.default_rng(8), np.random.default_rng(8)
    with gpu.pretend_gpu(FAKE):
        lat.run_point(model, 2.0, rng, 5, 100, seed=8)
    assert rng.integers(10 ** 9) == twin.integers(10 ** 9)


# ------------------------------------------------------------------------------------------ random matrices

@pytest.mark.parametrize("kind", ["GOE", "GUE"])
def test_gpu_spectra_equal_the_cpu_spectra(kind):
    cpu, _ = rm.spectra(kind, 24, 6, seed=7)
    with gpu.pretend_gpu(FAKE):
        dev, _ = rm.spectra(kind, 24, 6, seed=7)
    assert np.allclose(cpu, dev, atol=1e-10) and np.all(np.diff(dev, axis=1) >= 0)


def test_gpu_wishart_spectra_equal_the_cpu_ones():
    cpu, _ = rm.wishart_spectra(12, 30, 5, seed=3)
    with gpu.pretend_gpu(FAKE):
        dev, _ = rm.wishart_spectra(12, 30, 5, seed=3)
    assert np.allclose(cpu, dev, atol=1e-10)


def test_batching_gives_the_same_answer_as_one_batch(monkeypatch):
    with gpu.pretend_gpu(FAKE):
        whole, _ = rm.spectra("GOE", 16, 7, seed=1)
        monkeypatch.setattr(rm, "_BATCH_BYTES", 16 * 16 * 8 * 2)          # two matrices per batch
        parts, _ = rm.spectra("GOE", 16, 7, seed=1)
    assert np.array_equal(whole, parts)
