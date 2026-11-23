"""A fingerprint of the files that decide how the app LOOKS (stylesheet, theme config, chart theming), taken when this module is first imported.

Why: a Streamlit server that was left running keeps the code it imported earlier, so a scan of an old server measures the OLD stylesheet and reports a result that is
silently wrong. The gallery page prints this fingerprint; the browser scans (scripts/dark_mode_contrast_scan.py, scripts/chart_text_overlap_scan.py) compute the fingerprint of
the files on disk and refuse to scan when the two differ ("restart the gallery"). Streamlit-free.
"""
import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILES = ("ui/theme.py", "ui/a11y.py", "ui/mode_nav.py", "ui/history_panel.py", "modules/palette.py", "modules/layout_audit.py", "modules/motion_prefs.py",
         "modules/plotter.py", ".streamlit/config.toml")


def digest(root: Path = ROOT) -> str:
    """12 hex characters over the contents of FILES as they are on disk now."""
    h = hashlib.sha256()
    for name in FILES:
        h.update(name.encode())
        h.update((root / name).read_bytes())
    return h.hexdigest()[:12]


LOADED = digest()          # what this process imported (computed once, at import time)
