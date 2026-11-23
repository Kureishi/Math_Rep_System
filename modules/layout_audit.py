"""
A static check of a Plotly figure's panel layout, in pixels, at a given page width (Streamlit-free; used by tests/test_layout_audit.py and scripts/figure_layout_audit.py).

It cannot see rendered text (scripts/chart_text_overlap_scan.py measures that in a real browser), but it can do the arithmetic that decides whether text FITS: the gap between two
stacked panels against what sits in it (the upper panel's tick labels and axis title, the lower panel's subplot title), the gap between two side-by-side panels against the right
panel's y tick labels and y-axis title, and whether each subplot title is narrower than its own panel. Text sizes are estimates (12 px font: 22 px for a row of tick labels, 20 px for a title
line, 6.5 px per character, 45 px for y tick labels), so it reports LIKELY problems; the browser scan is the final word.

Widths: a figure fills its container. In a 1500 px window with the sidebar open the page is about 1100 px wide; in an 800 px window with the sidebar open, about 400 px.
"""
from dataclasses import dataclass
from typing import Any

import plotly.graph_objects as go

WIDTHS = {"1500 px window": 1100, "800 px window, sidebar closed": 700, "800 px window, sidebar open": 400}
TICK_ROW_PX, TITLE_LINE_PX, CHAR_PX, Y_TICKS_PX, TOLERANCE_PX = 22, 20, 6.5, 45, 3
DEFAULT_HEIGHT, DEFAULT_MARGIN = 450, {"l": 80, "r": 80, "t": 100, "b": 80}


@dataclass(frozen=True)
class Issue:
    kind: str            # 'stacked-gap' | 'side-gap' | 'subplot-title'
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.detail}"


def _margin(fig: go.Figure, side: str) -> float:
    v = getattr(fig.layout.margin, side, None)
    return DEFAULT_MARGIN[side] if v is None else v


def _axes(fig: go.Figure, kind: str) -> dict[str, Any]:
    """{'xaxis2': axis, ...} for the axes that exist in the layout and are drawn (have a domain)."""
    out = {}
    for name in fig.layout:
        if name.startswith(f"{kind}axis") and fig.layout[name].domain is not None:
            out[name] = fig.layout[name]
    return out


def _has_ticks(axis) -> bool:
    return axis.showticklabels is not False and axis.visible is not False


def _titled(axis) -> bool:
    return bool(axis.title and axis.title.text)


def audit(fig: go.Figure, content_width: float) -> list[Issue]:
    """The likely layout problems of `fig` when drawn `content_width` px wide."""
    plot_w = (fig.layout.width or content_width) - _margin(fig, "l") - _margin(fig, "r")
    plot_h = (fig.layout.height or DEFAULT_HEIGHT) - _margin(fig, "t") - _margin(fig, "b")
    xs, ys = _axes(fig, "x"), _axes(fig, "y")
    if len(xs) < 2 and len(ys) < 2:
        return []
    # each panel = (x axis, y axis) pair that anchor to one another
    panels = []
    for yname, y in ys.items():
        anchor = y.anchor or "x"
        xname = "xaxis" + anchor[1:] if anchor.startswith("x") else "xaxis"
        if xname in xs:
            panels.append((xname, xs[xname], yname, y))
    issues: list[Issue] = []
    titles = [a for a in (fig.layout.annotations or ()) if a.text and a.xref == "paper" and a.yref == "paper"]

    for i, (xn_a, xa, yn_a, ya) in enumerate(panels):
        for xn_b, xb, yn_b, yb in panels[i + 1:]:
            x_overlap = min(xa.domain[1], xb.domain[1]) - max(xa.domain[0], xb.domain[0])
            y_overlap = min(ya.domain[1], yb.domain[1]) - max(ya.domain[0], yb.domain[0])
            if x_overlap > 0.02 and y_overlap <= 0:                                      # stacked
                (ux, uy), (lx, ly) = ((xa, ya), (xb, yb)) if ya.domain[0] >= yb.domain[1] else ((xb, yb), (xa, ya))
                gap_px = (uy.domain[0] - ly.domain[1]) * plot_h
                needed = (TICK_ROW_PX if _has_ticks(ux) and ux.matches is None else 0) + (TITLE_LINE_PX if _titled(ux) else 0)
                if any(abs(t.y - ly.domain[1]) < 0.06 and lx.domain[0] - 0.02 <= t.x <= lx.domain[1] + 0.02 for t in titles):
                    needed += TITLE_LINE_PX + 4                                          # the lower panel's subplot title sits just above it
                if gap_px < needed - TOLERANCE_PX:
                    issues.append(Issue("stacked-gap", f"{gap_px:.0f} px between {yn_a} and {yn_b} but about {needed} px of text sits there"))
            elif y_overlap > 0.02 and x_overlap <= 0:                                    # side by side
                left, right = ((xa, ya), (xb, yb)) if xa.domain[1] <= xb.domain[0] else ((xb, yb), (xa, ya))
                gap_px = (right[0].domain[0] - left[0].domain[1]) * plot_w
                shared_y = right[1].matches is not None
                needed = 0 if shared_y else (Y_TICKS_PX if _has_ticks(right[1]) else 0) + (TITLE_LINE_PX if _titled(right[1]) else 0)
                if gap_px < needed - TOLERANCE_PX:
                    issues.append(Issue("side-gap", f"{gap_px:.0f} px between two panels but their y tick labels and title need about {needed} px"))
    for t in titles:
        owners = [p for p in panels if p[1].domain[0] - 0.02 <= t.x <= p[1].domain[1] + 0.02 and abs(t.y - p[3].domain[1]) < 0.06]
        if not owners:
            continue
        panel_px = (owners[0][1].domain[1] - owners[0][1].domain[0]) * plot_w
        text = t.text.replace("<b>", "").replace("</b>", "")
        longest = max(text.split("<br>"), key=len)                                        # a wrapped title is as wide as its longest line
        if len(longest) * CHAR_PX > panel_px + 2 * 10:
            issues.append(Issue("subplot-title", f"'{longest[:40]}' needs about {len(longest) * CHAR_PX:.0f} px but its panel is {panel_px:.0f} px wide"))
    return issues


def fit_subplot_titles(fig: go.Figure, assumed_content_px: float = 700) -> int:
    """Wrap subplot titles that would not fit their panel at `assumed_content_px` (a laptop window with the sidebar closed, or a wide one with it open) onto two or
    three lines with <br>, and give the top margin the extra room. Idempotent (a title already containing <br> is left alone), so a cached figure can be passed again.
    The width a figure will get is not known in Python, so this is a fixed, conservative assumption: it fixes the 800 px-window cases the audit found without making
    the titles of a wide window look odd (a title that fits is not touched). Returns how many titles were wrapped."""
    import textwrap
    plot_w = (fig.layout.width or assumed_content_px) - _margin(fig, "l") - _margin(fig, "r")
    xs, ys = _axes(fig, "x"), _axes(fig, "y")
    panels = []
    for yname, y in ys.items():
        anchor = y.anchor or "x"
        xname = "xaxis" + anchor[1:] if anchor.startswith("x") else "xaxis"
        if xname in xs:
            panels.append((xs[xname], y))
    wrapped, extra_lines = 0, 0
    for t in fig.layout.annotations or ():
        if not t.text or "<br>" in t.text or t.xref != "paper" or t.yref != "paper":
            continue
        owners = [p for p in panels if p[0].domain[0] - 0.02 <= t.x <= p[0].domain[1] + 0.02 and abs(t.y - p[1].domain[1]) < 0.06]
        if not owners:
            continue
        panel_px = (owners[0][0].domain[1] - owners[0][0].domain[0]) * plot_w
        per_line = max(12, int((panel_px + 20) / CHAR_PX))
        if len(t.text) > per_line:
            lines = textwrap.wrap(t.text, width=per_line, break_long_words=False)
            t.text = "<br>".join(lines)
            wrapped += 1
            extra_lines = max(extra_lines, len(lines) - 1)
    if extra_lines:
        top = _margin(fig, "t")
        fig.update_layout(margin=dict(t=top + 14 * extra_lines))
    return wrapped


def audit_all_widths(fig: go.Figure) -> dict[str, list[Issue]]:
    return {name: audit(fig, w) for name, w in WIDTHS.items()}
