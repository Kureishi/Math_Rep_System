"""
Optional GPU acceleration.

The app is local-first and portable, so the GPU is strictly an OPTIONAL accelerator: with no NVIDIA GPU, no CUDA
driver or no CuPy installed, nothing changes -- every function runs on NumPy exactly as before. Install CuPy
(`requirements-gpu.txt`) and the heavy, data-parallel kernels pick the GPU up by themselves.

WHAT RUNS ON THE GPU (and why only these)
  * Langevin ensembles (sm_stochastic.simulate_overdamped / simulate_underdamped): one element-wise update per step
    over every path.
  * Lattice Metropolis sweeps (sm_lattice: Ising, Potts, XY): the checkerboard update is pure array arithmetic.
  * Random-matrix spectra (sm_randommatrix.spectra / wishart_spectra): batched eigenvalue solves. The matrices
    are still drawn on the CPU from the same seeded generators, so these results do not depend on the backend.
  * Black-hole images (showcase.trace_rays): every pixel's light ray is integrated at once; deterministic, so the image
    does not depend on the backend.
  Everything else stays on the CPU on purpose: SymPy (symbolic, single-threaded Python), adaptive ODE/geodesic
  integration (sequential), and the 1-D quantum / PDE grids (a few thousand points: the GPU would be slower than
  the launch overhead alone). LM Studio uses the GPU for the language model by itself (its own GPU-offload setting).

WHEN IT IS USED. `auto` (the default) uses the GPU only when the job is big enough to beat the cost of launching
GPU kernels and copying data (THRESHOLDS below); small jobs stay on the CPU, where they are faster. `on` uses it
whenever it is available (to benchmark, or to test); `off` never does. Set it in Advanced settings, with the
environment variable MRS_GPU=auto|on|off, or with `set_mode()`.

REPRODUCIBILITY, stated plainly. A seed still determines a run, but a GPU draws its random numbers from its own
generator, so the SAME seed gives statistically equivalent -- not bit-identical -- results on the CPU and on the
GPU. Results record which backend produced them (`backend`). The random-matrix spectra are the exception (above).

FAILURE IS NOT FATAL. If a GPU run fails for any reason that is not the user's input (out of memory, a driver
fault, a function CuPy cannot run), `attempt()` logs it and repeats the job on the CPU.
"""
import os
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterator, TypeVar

import numpy as np

from modules.app_logging import logger

T = TypeVar("T")

MODES = ("auto", "on", "off")

# Smallest job (in the unit named) for which `auto` mode prefers the GPU. These are conservative estimates of where
# kernel-launch and transfer overhead stop dominating on a typical discrete GPU; they have not been measured on
# every card -- use mode "on" to compare, and adjust here if your hardware differs.
THRESHOLDS: dict[str, int] = {
    "langevin_paths": 100_000,        # paths in an ensemble
    "lattice_sites": 128 * 128,       # sites on the lattice
    "matrix_dim": 128,                # side of a random matrix
    "raytrace_pixels": 250 * 250,     # rays in a black-hole image
}


@dataclass(frozen=True)
class Backend:
    xp: Any                 # the array module: numpy or cupy
    name: str               # for display and for result metadata
    is_gpu: bool
    module: str = "numpy"   # the name sympy.lambdify understands


CPU = Backend(np, "CPU (NumPy)", False, "numpy")

_mode = "auto"
_override: Backend | None = None          # tests only: pretend this backend is the GPU


def _initial_mode() -> str:
    value = os.getenv("MRS_GPU", "auto").strip().lower()
    return value if value in MODES else "auto"


_mode = _initial_mode()


def get_mode() -> str:
    return _mode


def set_mode(mode: str) -> None:
    global _mode
    if mode not in MODES:
        raise ValueError(f"GPU mode must be one of {', '.join(MODES)}.")
    _mode = mode


@lru_cache(maxsize=1)
def _detect() -> Backend:
    """The CuPy backend if CuPy imports, a CUDA device exists and a trivial kernel actually runs; otherwise CPU."""
    try:
        import cupy as cp
    except ImportError:
        logger.debug("CuPy is not installed; running on the CPU")
        return CPU
    except Exception as exc:  # noqa: BLE001 -- a broken CUDA install must not stop the app starting
        logger.warning("CuPy could not be imported (%s: %s); running on the CPU", type(exc).__name__, exc)
        return CPU
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return CPU
        probe = cp.arange(8.0)
        if float((probe * probe).sum()) != 140.0:            # also forces the kernel to compile and run
            raise RuntimeError("the test kernel gave the wrong answer")
        raw = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)["name"]
        label = raw.decode() if isinstance(raw, bytes) else str(raw)
        return Backend(cp, f"GPU: {label} (CuPy)", True, "cupy")
    except Exception as exc:  # noqa: BLE001
        logger.warning("No usable GPU (%s: %s); running on the CPU", type(exc).__name__, exc)
        return CPU


def available() -> Backend:
    """The GPU backend if there is a working one (or the test override), else the CPU backend."""
    return _override if _override is not None else _detect()


def select(size: int, kind: str) -> Backend:
    """The backend for a job of `size` units of `kind` (a key of THRESHOLDS)."""
    if _mode == "off":
        return CPU
    backend = available()
    if not backend.is_gpu:
        return CPU
    if _mode == "auto" and size < THRESHOLDS[kind]:
        return CPU
    return backend


def describe() -> str:
    """One line for the UI: what is in use and why."""
    backend = available()
    if _mode == "off":
        return "GPU acceleration is off" + (f" (a GPU is available: {backend.name})." if backend.is_gpu else ".")
    if not backend.is_gpu:
        return "CPU only: no usable NVIDIA GPU with CuPy was found (see requirements-gpu.txt)."
    if _mode == "on":
        return f"{backend.name}: used for every supported computation."
    return (f"{backend.name}: used for large ensembles (≥ {THRESHOLDS['langevin_paths']:,} paths), "
            f"lattices (L ≥ {int(THRESHOLDS['lattice_sites'] ** 0.5)}) and matrices (N ≥ {THRESHOLDS['matrix_dim']}).")


def to_host(a: Any) -> np.ndarray:
    """A NumPy array from a NumPy or CuPy array (a no-copy pass-through for NumPy)."""
    if isinstance(a, np.ndarray):
        return a
    if hasattr(a, "get"):
        return a.get()
    return np.asarray(a)


def array_module(a: Any) -> Any:
    """The array module (numpy or cupy) that `a` belongs to."""
    return np if isinstance(a, np.ndarray) else available().xp


def on_device(backend: Backend, a: Any) -> Any:
    """A COPY of `a` as an array of the backend's module (so the caller's array is never changed in place)."""
    return backend.xp.array(a)


def attempt(backend: Backend, on_gpu: Callable[[], T], on_cpu: Callable[[], T],
            passthrough: tuple[type[BaseException], ...] = ()) -> T:
    """Run `on_gpu` when `backend` is a GPU; if it fails for a reason other than the user's input (the exception
    types in `passthrough`, which are re-raised), log it and repeat the job with `on_cpu`."""
    if backend.is_gpu:
        try:
            return on_gpu()
        except passthrough:
            raise
        except Exception as exc:  # noqa: BLE001 -- out of memory, driver fault, unsupported function...
            logger.warning("GPU run failed (%s: %s); repeating it on the CPU", type(exc).__name__, exc)
    return on_cpu()


@contextmanager
def pretend_gpu(backend: Backend, mode: str = "on") -> Iterator[None]:
    """Tests: route the GPU code paths through `backend` (e.g. NumPy flagged as a GPU) so they run on any machine."""
    global _override, _mode
    saved = _override, _mode
    _override, _mode = backend, mode
    try:
        yield
    finally:
        _override, _mode = saved
