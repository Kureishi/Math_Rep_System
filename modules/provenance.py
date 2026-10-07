"""
What a solved problem was produced with, recorded so an export can be traced and reproduced -- Streamlit-free.

A report that says "x = 2.0" is only as trustworthy as the ability to ask how it was made: which model
extracted the equations, what tolerances the checks used, which library versions did the algebra, which
seeds drove any sampling. `capture()` snapshots that at SOLVE time and the snapshot is stored with the
problem's history record, so reopening a problem next month still reports what it was actually solved
with, rather than whatever the settings happen to be today.

A problem saved before this existed has no snapshot. `describe()` then says so plainly and labels the
current settings as "now", instead of presenting today's settings as if they produced an old result.

What is NOT claimed: the extraction is sampled from a language model at a temperature, so the same text and
the same model do not promise the same equations. The footer states the temperature rather than implying
determinism; what IS deterministic and reproducible is everything after extraction (the SymPy solve, the
checks) and any Monte Carlo run given its seed.
"""
import platform
from datetime import datetime
from typing import Any

from config import APP_VERSION, Settings, settings as default_settings

PROVENANCE_VERSION = 1


def _library_versions() -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("sympy", "numpy", "scipy"):
        try:
            out[name] = __import__(name).__version__
        except Exception:  # noqa: BLE001 -- a missing or odd package must never block a solve
            out[name] = "unknown"
    return out


def capture(cfg: Settings | None = None, used_secondary_model: bool = False) -> dict[str, Any]:
    """The settings and versions in force right now, as a JSON-safe dict."""
    cfg = cfg or default_settings
    return {
        "version": PROVENANCE_VERSION,
        "app_version": APP_VERSION,
        "captured_at": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "reasoning_model": cfg.reasoning_model,
        "secondary_model": cfg.secondary_reasoning_model if used_secondary_model and cfg.secondary_reasoning_model else "",
        "temperature_extraction": cfg.temperature_extraction,
        "temperature_narration": cfg.temperature_narration,
        "numeric_tolerance": cfg.numeric_tolerance,
        "cross_check_tolerance": cfg.cross_check_tolerance,
        "max_verification_retries": cfg.max_verification_retries,
        "computation_timeout_seconds": cfg.computation_timeout_seconds,
        "python": platform.python_version(),
        "libraries": _library_versions(),
    }


def describe(stored: dict[str, Any] | None, monte_carlo_runs: list[dict[str, Any]] | None = None,
             now: dict[str, Any] | None = None) -> list[tuple[str, str]]:
    """(label, value) rows for an export's reproducibility footer.

    `stored` is the snapshot saved with the problem, or None for one saved before snapshots existed (then
    the rows come from `now`, defaulting to the current settings, and say so)."""
    now = now or capture()
    known = isinstance(stored, dict) and stored.get("reasoning_model") is not None
    src = stored if known and stored is not None else now
    rows: list[tuple[str, str]] = []
    if known:
        rows.append(("Solved", f"{src.get('captured_at', 'unknown time')} with app version {src.get('app_version', '?')}"))
    else:
        rows.append(("Solved", "no record of how this problem was solved (saved before settings were recorded); "
                               "the settings below are the CURRENT ones, not necessarily those that produced it"))
    rows.append(("Exported", f"{now.get('captured_at', '')} with app version {now.get('app_version', '?')}"))
    model_text = str(src.get("reasoning_model", "unknown"))
    if src.get("secondary_model"):
        model_text += f" (cross-checked with {src['secondary_model']})"
    rows.append(("Language model", f"{model_text}; extraction temperature {src.get('temperature_extraction', '?')}, "
                                    f"narration temperature {src.get('temperature_narration', '?')}"))
    rows.append(("Checks", f"numeric tolerance {src.get('numeric_tolerance', '?')}, independent cross-check "
                           f"tolerance {src.get('cross_check_tolerance', '?')}, up to "
                           f"{src.get('max_verification_retries', '?')} verification retries, "
                           f"{src.get('computation_timeout_seconds', '?')} s per symbolic computation"))
    libs = src.get("libraries") or {}
    rows.append(("Software", f"Python {src.get('python', '?')}; " +
                 ", ".join(f"{k} {v}" for k, v in sorted(libs.items()))))
    for run in monte_carlo_runs or []:
        rows.append((f"Monte Carlo ({run.get('target', '?')})",
                     f"{run.get('n', '?')} samples, seed {run.get('seed', 'unrecorded')}, uncertain inputs "
                     f"{', '.join(run.get('inputs', [])) or 'none recorded'}"))
    rows.append(("Reproducibility",
                 "Equation extraction is sampled from a language model and is not guaranteed to repeat; the "
                 "solve, the checks and any Monte Carlo run (given its seed) are deterministic."))
    return rows
