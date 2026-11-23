"""
Animation preferences for Plotly figures with frames (Streamlit-free): a speed multiplier and a "reduce motion" switch.

  speed          0.25x .. 4x. Every Play button's frame duration is divided by it, so 2x plays twice as fast and 0.5x half as fast.
  reduce motion  no Play button at all (and no autoplay): the figure shows one frame and the person steps through the others with its slider. People who
                 get dizzy or distracted by moving graphics, and anyone on a slow machine, can still see every frame.

`apply` never changes the figure it is given: figures are cached and reused across reruns, and a figure with its Play button removed could never get it back when the setting is
turned off again. It returns the same object when there is nothing to do (no frames, speed 1, motion not reduced), so ordinary charts pay nothing.
"""
from typing import Any

import plotly.graph_objects as go

SPEEDS: tuple[float, ...] = (0.25, 0.5, 1.0, 1.5, 2.0, 4.0)
DEFAULT_SPEED = 1.0
MIN_FRAME_MS = 1                      # Plotly treats 0 as "as fast as possible"; a Play button never gets a duration below this


def _is_play(button: Any) -> bool:
    """A Play button runs the animation with a non-zero frame duration; Pause (and a slider step) use a duration of 0."""
    if getattr(button, "method", None) != "animate":
        return False
    args = button.args
    opts = args[1] if args is not None and len(args) > 1 else None
    if not isinstance(opts, dict):
        return False
    frame = opts.get("frame") or {}
    return bool(frame.get("duration", 0)) and args[0] is None


def has_animation(fig: go.Figure) -> bool:
    return bool(fig.frames) and any(_is_play(b) for m in (fig.layout.updatemenus or ()) for b in (m.buttons or ()))


def clamp_speed(speed: float | None) -> float:
    try:
        value = float(speed) if speed is not None else DEFAULT_SPEED
    except (TypeError, ValueError):
        return DEFAULT_SPEED
    return min(max(value, SPEEDS[0]), SPEEDS[-1])


def apply(fig: go.Figure, speed: float = DEFAULT_SPEED, reduce_motion: bool = False) -> go.Figure:
    """The figure to draw: `fig` itself if nothing changes, otherwise a copy with the Play durations scaled (or the Play buttons removed)."""
    speed = clamp_speed(speed)
    if not has_animation(fig) or (speed == 1.0 and not reduce_motion):
        return fig
    out = go.Figure(fig)
    menus = []
    for menu in out.layout.updatemenus or ():
        buttons = []
        for button in menu.buttons or ():
            if _is_play(button):
                if reduce_motion:
                    continue                                          # no Play button: the slider still steps through every frame
                args = list(button.args)
                opts = {**args[1], "frame": {**args[1]["frame"]}}
                opts["frame"]["duration"] = max(MIN_FRAME_MS, round(opts["frame"]["duration"] / speed))
                args[1] = opts
                button = {**button.to_plotly_json(), "args": args}
            buttons.append(button if isinstance(button, dict) else button.to_plotly_json())
        if any(_is_play_json(b) for b in buttons) or not reduce_motion:
            menus.append({**menu.to_plotly_json(), "buttons": buttons})
    out.layout.updatemenus = menus
    return out


def _is_play_json(button: dict) -> bool:
    args = button.get("args") or []
    return button.get("method") == "animate" and len(args) > 1 and isinstance(args[1], dict) and bool((args[1].get("frame") or {}).get("duration", 0)) and args[0] is None
