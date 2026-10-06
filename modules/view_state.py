"""
Which widget settings count as "the exploration state" of a solved problem, and how to capture them
for saving with the problem -- Streamlit-free (it takes any mapping, so tests need no Streamlit).

Reopening a past problem used to give back the solution but not the way the person had been looking at
it: the Monte Carlo inputs and seed, the sweep they set up, the plot axes and sliders, the time views.
This module decides what to remember. It is a WHITELIST of key patterns, never "everything in
session_state": the session also holds uploads, API objects, other pages' inputs and the problem text,
none of which belong to one problem's saved view.

Only plain scalars (bool, int, float, str) and short lists of them are kept. A data_editor's value is a
nested dict of edits that only makes sense against the exact table it was edited from, so those widgets
(the tables of uncertainties and sweep ranges) are not saved -- their rows are rebuilt from the
remembered selections, with the default widths, rather than half-restored.

A saved view is plain JSON, stored with the problem's history record (modules/history.py) and carried by
project bundles (modules/project_bundle.py).
"""
import hashlib
import json
import math
import re
from collections.abc import Mapping
from typing import Any

VIEW_STATE_VERSION = 1
MAX_KEYS = 400
MAX_STRING = 500
MAX_LIST = 100

# fullmatch'd against a session_state key. Each line names the panel whose widgets it covers; a test
# scans the pages' source so a new widget key in one of these panels is not silently left out, and so a
# pattern never matches a key from a page that is NOT part of a solved problem's results.
EXPLORATION_KEY_PATTERNS: tuple[str, ...] = (
    r"var_.+",                                              # the editable variables table
    r"sweep_target", r"sweep_symbols",                      # N-dimensional parameter sweep
    r"bulk_mc_vars", r"bulk_mc_n", r"bulk_mc_seed",         # uncertainty across all targets
    r"mc_vars_.+", r"mc_n_.+", r"mc_seed_.+",               # per-target Monte Carlo
    r"ep_vars_.+", r"iv_vars_.+",                           # error propagation, interval bounds
    r"gs_symbol_.+", r"gs_target_.+",                       # goal seek
    r"sens_pct_.+", r"sweep_pick_.+",                       # sensitivity
    r"tutor_on_.+", r"altmethod_.+", r"explain_mode_.+",    # step list
    r"surf_.+", r"contour_.+", r"line_.+", r"slider_.+",    # interactive plots
    r"rotate_.+", r"region_.+",
    r"ode(?:param|range|eval|err_end)_.+", r"phase_(?:trange|extra)_.+",     # ODE views
    r"rec(?:param|range|eval)_.+", r"cobweb_(?:range|steps)_.+",             # recurrence views
    r"cobweb2_.+", r"fan_.+", r"morph_.+", r"bif_.+",                        # time views
    r"motion_.+",                                                            # motion diagram
)
_RX = re.compile("|".join(f"(?:{p})" for p in EXPLORATION_KEY_PATTERNS))


def is_exploration_key(key: Any) -> bool:
    return isinstance(key, str) and _RX.fullmatch(key) is not None


def _scalar(v: Any) -> Any:
    """`v` as a JSON-safe scalar, or _DROP."""
    if hasattr(v, "item") and not isinstance(v, (str, bytes)):          # numpy scalars
        try:
            v = v.item()
        except Exception:  # noqa: BLE001
            return _DROP
    if isinstance(v, bool) or v is None:
        return v if v is not None else _DROP
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return v if math.isfinite(v) else _DROP
    if isinstance(v, str):
        return v if len(v) <= MAX_STRING else _DROP
    return _DROP


_DROP = object()


def normalize_value(v: Any) -> Any:
    """`v` made JSON-safe, or _DROP if it isn't a scalar or a short list of scalars."""
    if isinstance(v, (list, tuple)):
        if len(v) > MAX_LIST:
            return _DROP
        items = [_scalar(i) for i in v]
        return _DROP if any(i is _DROP for i in items) else items
    return _scalar(v)


def capture(session_state: Any) -> dict[str, Any]:
    """The exploration state in `session_state` (anything with keys() and item access -- a dict, or
    Streamlit's session-state proxy, which is not a Mapping), as {key: value}, sorted by key."""
    out: dict[str, Any] = {}
    for key in sorted(k for k in list(session_state.keys()) if is_exploration_key(k)):
        try:
            value = normalize_value(session_state[key])
        except Exception:  # noqa: BLE001 -- a key that cannot even be read is not worth saving
            continue
        if value is _DROP:
            continue
        out[key] = value
        if len(out) >= MAX_KEYS:
            break
    return out


def digest(state: Mapping[str, Any]) -> str:
    """A short fingerprint of a captured state, to tell whether it changed since it was last saved."""
    return hashlib.blake2b(json.dumps(state, sort_keys=True).encode(), digest_size=8).hexdigest()


def sanitize_saved(saved: Any) -> dict[str, Any]:
    """What is safe to restore from `saved` (loaded from the database or a bundle, so untrusted): only
    exploration keys, only scalar/short-list values."""
    if not isinstance(saved, Mapping):
        return {}
    values = saved.get("values", saved) if "values" in saved else saved
    if not isinstance(values, Mapping):
        return {}
    clean: dict[str, Any] = {}
    for key, value in values.items():
        if not is_exploration_key(key):
            continue
        v = normalize_value(value)
        if v is _DROP:
            continue
        clean[key] = v
        if len(clean) >= MAX_KEYS:
            break
    return clean


def wrap(state: Mapping[str, Any]) -> dict[str, Any]:
    """The storable form: the values plus a format version."""
    return {"version": VIEW_STATE_VERSION, "values": dict(state)}


def keys_to_clear(session_state: Any) -> list[str]:
    """Every exploration key currently in `session_state` -- cleared before a different problem's saved
    view is applied, so one problem's settings (keyed by symbol and target names, which problems share)
    never leak into another's."""
    return [k for k in list(session_state.keys()) if is_exploration_key(k)]
