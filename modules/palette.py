"""
Colour-blind-safe, theme-aware chart colours and text descriptions of charts -- Streamlit-free.

WHY NOT "JUST USE A COLOUR-BLIND PALETTE". The palette this replaces (Tableau's) had a green and a red that are almost the same
colour to someone with deuteranopia (colour difference ΔE 1.7 where 2 is barely perceptible), several colours below 3:1 contrast on
one of the two backgrounds, and was copied into eight plot modules. This one is chosen against measurable criteria, and the tests
in tests/test_palette.py enforce them:
  * every series colour has at least 3:1 contrast (WCAG 1.4.11, graphical objects) against BOTH a white page and the dark theme's
    background (#0E1117), so a chart reads in either;
  * the first four colours stay at least ΔE 15 apart under simulated protanopia, deuteranopia and tritanopia (Machado et al.
    2009, severity 1; ΔE is CIE76 in Lab); the status pair blue / vermillion is far further apart than that;
  * the badge text colours reach WCAG AA (4.5:1) on their tints.
It starts from the Okabe-Ito palette (designed for colour-blind readers) but adjusts shades where Okabe-Ito itself fails the
contrast criterion (its light orange, sky blue, yellow and black).

WHAT COLOUR CANNOT DO, STATED. With more than about four series, colour alone cannot keep every pair apart for every reader (orange
and vermillion are close under any vision type, which is why orange is last). `line_style(i)` therefore pairs colour with line
dash from the fifth series on, and status badges carry a symbol as well as a colour (see ui/theme.py).

CHART DESCRIPTIONS. `describe_figure` writes a plain-text summary of any Plotly figure (title, axes, series, ranges, panels,
animation) so a chart is not only a picture: it is read out to screen readers and can be shown as text (ui/a11y.py).
"""
import itertools
import math
from collections.abc import Sequence
from typing import Any

import numpy as np

# ------------------------------------------------------------------------------------------ the colours

BLUE = "#0072B2"
VERMILLION = "#D55E00"
PURPLE = "#CC79A7"
GREEN = "#009E73"
SKY = "#4A9BC8"
ORANGE = "#C48700"
GRAY = "#8A8F98"

CATEGORICAL: list[str] = [BLUE, VERMILLION, PURPLE, GREEN, SKY, ORANGE, GRAY]     # ordered: the first colours are the most distinct
INK = GRAY                                  # a neutral mark that is visible on a light AND a dark background (data points, guides)
GOOD, ATTENTION = BLUE, VERMILLION          # the status pair, in place of green / red
SEQUENTIAL = "Viridis"                      # perceptually uniform and colour-blind safe
DIVERGING = "RdBu"                          # blue-red diverging scale: safe for the common deficiencies
DASHES = ("solid", "dash", "dot", "dashdot", "longdash", "longdashdot")

LIGHT_BACKGROUND = "#FFFFFF"
DARK_BACKGROUND = "#0E1117"                 # the app's dark mode (ui/theme.py)

# (background tint, text) per badge kind; the text meets WCAG AA on its tint
BADGE_COLORS: dict[str, tuple[str, str]] = {
    "pass": ("#DDF1F7", "#005A8C"),
    "warn": ("#FFF3D6", "#7A5200"),
    "fail": ("#F8E1EB", "#8A2A5E"),
    "neutral": ("#EEF1F6", "#3B4252"),
}


def line_style(i: int) -> tuple[str, str]:
    """(colour, dash) for the i-th series: the first four are solid; from the fifth on the dash changes too, so a series is
    identifiable by pattern as well as by colour."""
    return CATEGORICAL[i % len(CATEGORICAL)], DASHES[(i // 4) % len(DASHES)]


# ------------------------------------------------------------------------------------------ measuring colour

def _rgb(hex_colour: str) -> np.ndarray:
    h = hex_colour.lstrip("#")
    return np.array([int(h[i:i + 2], 16) for i in (0, 2, 4)], dtype=float)


def _linear(rgb255: np.ndarray) -> np.ndarray:
    c = np.asarray(rgb255, dtype=float) / 255.0
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _gamma(lin: np.ndarray) -> np.ndarray:
    c = np.clip(lin, 0.0, 1.0)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055) * 255.0


def luminance(hex_colour: str) -> float:
    r, g, b = _linear(_rgb(hex_colour))
    return float(0.2126 * r + 0.7152 * g + 0.0722 * b)


def contrast_ratio(a: str, b: str) -> float:
    """WCAG contrast ratio between two colours (1 = identical, 21 = black on white)."""
    la, lb = luminance(a), luminance(b)
    return (max(la, lb) + 0.05) / (min(la, lb) + 0.05)


# Machado, Oliveira & Fernandes (2009), full severity, applied in linear RGB.
_CVD = {
    "protanopia": np.array([[0.152286, 1.052583, -0.204868], [0.114503, 0.786281, 0.099216], [-0.003882, -0.048116, 1.051998]]),
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968], [0.280085, 0.672501, 0.047413], [-0.011820, 0.042940, 0.968881]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779], [-0.078411, 0.930809, 0.147602], [0.004733, 0.691367, 0.303900]]),
}
VISION_TYPES = ("normal",) + tuple(_CVD)


def simulate(hex_colour: str, vision: str) -> np.ndarray:
    """The colour as seen with the given vision type, as an sRGB triple in 0-255."""
    rgb = _rgb(hex_colour)
    return rgb if vision == "normal" else _gamma(_CVD[vision] @ _linear(rgb))


def _lab(rgb255: np.ndarray) -> np.ndarray:
    lin = _linear(rgb255)
    x = lin @ np.array([0.4124, 0.3576, 0.1805]) / 0.95047
    y = lin @ np.array([0.2126, 0.7152, 0.0722])
    z = lin @ np.array([0.0193, 0.1192, 0.9505]) / 1.08883

    def f(t):
        return np.where(t > 0.008856, np.cbrt(t), 7.787 * t + 16 / 116)

    fx, fy, fz = f(x), f(y), f(z)
    return np.array([116 * fy - 16, 500 * (fx - fy), 200 * (fy - fz)])


def delta_e(a: str, b: str, vision: str = "normal") -> float:
    """CIE76 colour difference between two colours as seen with `vision` (about 2 is just noticeable; 10+ is plainly different)."""
    return float(np.linalg.norm(_lab(simulate(a, vision)) - _lab(simulate(b, vision))))


def worst_pair_distance(colours: Sequence[str]) -> float:
    """The smallest ΔE between any two of `colours` under any of the vision types: how badly the closest pair can collapse."""
    return min(delta_e(a, b, v) for a, b in itertools.combinations(colours, 2) for v in VISION_TYPES)


# ------------------------------------------------------------------------------------------ theming a Plotly figure

def plotly_theme(dark: bool) -> dict[str, Any]:
    """Layout settings that make a figure follow the app's light/dark choice and use the colour-blind-safe series colours for
    any trace that does not set its own. Idempotent: applying it twice, or to a cached figure, changes nothing further."""
    return dict(template="plotly_dark" if dark else "plotly_white", colorway=list(CATEGORICAL),
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")


def apply_theme(fig: Any, dark: bool) -> Any:
    fig.update_layout(**plotly_theme(dark))
    return fig


# ------------------------------------------------------------------------------------------ describing a figure

_KIND = {"scatter": "line or point series", "scattergl": "line or point series", "bar": "bars", "histogram": "histogram",
         "histogram2d": "2-D histogram", "histogram2dcontour": "density contours", "heatmap": "heat map", "image": "image",
         "surface": "3-D surface", "scatter3d": "3-D series", "contour": "contours", "box": "box plot", "violin": "violin plot",
         "pie": "pie chart", "table": "table", "mesh3d": "3-D mesh", "cone": "arrows", "scatterpolar": "polar series"}


def _numbers(values: Any) -> np.ndarray | None:
    if values is None:
        return None
    try:
        arr = np.asarray(values, dtype=float).ravel()
    except (TypeError, ValueError):
        return None
    arr = arr[np.isfinite(arr)]
    return arr if arr.size else None


def _count(trace: Any) -> int | None:
    for attr in ("y", "x", "z", "values", "r"):
        v = getattr(trace, attr, None)
        if v is not None:
            try:
                return int(np.shape(v)[0]) if np.ndim(v) else None
            except (TypeError, IndexError):
                return None
    return None


def _trace_phrase(trace: Any) -> str:
    kind = _KIND.get(trace.type, trace.type)
    mode = getattr(trace, "mode", None)
    if trace.type in ("scatter", "scattergl"):
        if mode:
            kind = "line" if mode == "lines" else "points" if mode == "markers" else "line with points"
        if getattr(trace, "fill", None) in ("toself", "tonexty", "tozeroy", "tonextx", "tozerox"):
            kind = "shaded band"
    name = getattr(trace, "name", None) or "unnamed"
    n = _count(trace)
    parts = [kind] + ([f"{n} values"] if n else [])
    for axis in ("y", "z"):
        nums = _numbers(getattr(trace, axis, None))
        if nums is not None and trace.type not in ("image", "table"):
            lo, hi = float(nums.min()), float(nums.max())
            parts.append(f"{axis} from {lo:.3g} to {hi:.3g}")
            break
    return f"{name} ({', '.join(parts)})"


def describe_figure(fig: Any, max_traces: int = 6, max_chars: int = 900) -> str:
    """A plain-text description of a Plotly figure. Never raises: a description is an extra and must not break the chart."""
    try:
        parts = []
        title = getattr(getattr(fig.layout, "title", None), "text", None)
        parts.append(f"Chart: {title}." if title else "Chart.")
        titles = [a.text for a in (fig.layout.annotations or []) if getattr(a, "text", None) and getattr(a, "yref", None) == "paper"
                  and not getattr(a, "showarrow", False)]
        if len(titles) > 1:
            parts.append(f"{len(titles)} panels: " + "; ".join(t for t in titles[:8]) + ("; …" if len(titles) > 8 else "") + ".")
        for axis, label in (("xaxis", "x axis"), ("yaxis", "y axis")):
            text = getattr(getattr(getattr(fig.layout, axis, None), "title", None), "text", None)
            if text:
                parts.append(f"{label}: {text}.")
        traces = [t for t in fig.data if getattr(t, "visible", True) is not False]
        if traces:
            shown = [_trace_phrase(t) for t in traces[:max_traces]]
            noun = "series" if len(traces) != 1 else "series"
            parts.append(f"{len(traces)} {noun}: " + "; ".join(shown) + (f"; and {len(traces) - max_traces} more" if len(traces) > max_traces else "") + ".")
        else:
            parts.append("No data is plotted.")
        if fig.frames:
            parts.append(f"Animated: {len(fig.frames)} frames with a slider and play button.")
        text = " ".join(parts)
        return text if len(text) <= max_chars else text[:max_chars - 1].rstrip() + "…"
    except Exception:  # noqa: BLE001
        return "Chart (no description could be generated)."
