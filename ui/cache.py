"""
Per-session caching for work that a Streamlit rerun would otherwise repeat.

Streamlit re-executes the whole script on every widget interaction, so moving
one slider used to rebuild every figure and re-solve every differential
equation on the page -- about 2 seconds of work for an ODE page, almost all of
it for things the slider had nothing to do with (an animated Plotly figure with
60 frames costs a second or more to construct, because every frame's traces are
validated one property at a time).

`cached(key, parts, compute)` returns the previous result if `parts` -- every
input the result depends on -- is unchanged since the last time that `key` ran,
and computes (and remembers) it otherwise. Staleness is prevented by what goes
into `parts`: it is fingerprinted by CONTENT (array bytes, expression structure,
dataclass fields), never by identity or `repr`, and anything that cannot be
fingerprinted makes the call fall back to computing -- slower, never wrong.

The results are shared between reruns, so the caller must treat them as read-only
(pass a cached figure to st.plotly_chart; do not call update_layout on it).
"""
import dataclasses
import hashlib
from collections.abc import Callable
from typing import Any, TypeVar

import numpy as np
import streamlit as st
import sympy as sp

T = TypeVar("T")

MAX_ENTRIES = 64               # the most recent distinct keys are kept; older ones are evicted


def _feed(h: Any, obj: Any) -> None:
    if obj is None:
        h.update(b"N;")
    elif isinstance(obj, bool):
        h.update(b"B1;" if obj else b"B0;")
    elif isinstance(obj, (int, float, complex, np.integer, np.floating, np.bool_)):
        h.update(b"n" + type(obj).__name__.encode() + b":" + repr(obj.item() if isinstance(obj, np.generic) else obj).encode() + b";")
    elif isinstance(obj, str):
        data = obj.encode()
        h.update(b"s" + str(len(data)).encode() + b":" + data)
    elif isinstance(obj, bytes):
        h.update(b"b" + str(len(obj)).encode() + b":" + obj)
    elif isinstance(obj, np.ndarray):
        if obj.dtype == object:
            raise TypeError("object arrays can't be fingerprinted by content")
        h.update(b"a" + str(obj.dtype).encode() + str(obj.shape).encode() + b":")
        h.update(np.ascontiguousarray(obj).tobytes())
    elif isinstance(obj, sp.Basic):
        _feed(h, sp.srepr(obj))
    elif dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        h.update(b"D" + type(obj).__qualname__.encode() + b"{")
        for f in dataclasses.fields(obj):
            h.update(f.name.encode() + b"=")
            _feed(h, getattr(obj, f.name))
        h.update(b"}")
    elif isinstance(obj, dict):
        h.update(b"d" + str(len(obj)).encode() + b"{")
        for k in sorted(obj, key=repr):
            _feed(h, k)
            _feed(h, obj[k])
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update((b"l" if isinstance(obj, list) else b"t") + str(len(obj)).encode() + b"(")
        for item in obj:
            _feed(h, item)
        h.update(b")")
    elif isinstance(obj, (set, frozenset)):
        h.update(b"S" + str(len(obj)).encode() + b"{")
        for item in sorted(obj, key=repr):
            _feed(h, item)
        h.update(b"}")
    else:
        raise TypeError(f"can't fingerprint a {type(obj).__name__} by content")


def fingerprint(*parts: Any) -> str:
    """A stable hash of the CONTENT of `parts`. Raises TypeError for anything it
    can't hash by content (a function, an arbitrary object), because a hash
    based on identity would be wrong the first time the object was rebuilt."""
    h = hashlib.blake2b(digest_size=16)
    for part in parts:
        _feed(h, part)
    return h.hexdigest()


def cached(key: str, parts: tuple, compute: Callable[[], T]) -> T:
    """`compute()` the first time, and again only when `parts` changes (see the
    module docstring). `key` names the call site (include anything that
    distinguishes several uses of it, such as the function being plotted)."""
    try:
        fp = fingerprint(*parts)
    except TypeError:
        return compute()
    store: dict[str, tuple[str, Any]] = st.session_state.setdefault("_ui_cache", {})
    entry = store.get(key)
    if entry is not None and entry[0] == fp:
        store[key] = store.pop(key)                 # most recently used goes last
        return entry[1]
    value = compute()
    store.pop(key, None)
    store[key] = (fp, value)
    while len(store) > MAX_ENTRIES:
        store.pop(next(iter(store)))
    return value


def model_signature(model: Any) -> str:
    """What identifies a ProblemModel for the purpose of re-using something
    derived from it: its variables and their values, its equations and its
    initial conditions."""
    return fingerprint(
        [(v.symbol, v.known_value, v.unit, getattr(v, "is_function", False)) for v in model.variables],
        [(e.name, e.kind, e.raw_expression) for e in model.equations],
        [(ic.raw_expression, ic.value) for ic in model.initial_conditions],
        getattr(model, "independent_variable", None),
    )


def cached_by_model(key: str, model: Any, compute: Callable[[], T], *extra: Any) -> T:
    """cached(), keyed by the model (plus any `extra` inputs) -- for pure
    functions of the model such as solve_ode(model)."""
    return cached(key, (model_signature(model), *extra), compute)
