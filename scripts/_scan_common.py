"""Shared by the browser scans: refuse to scan a server that is serving an old stylesheet."""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from ui.build_stamp import digest   # noqa: E402


def check_server_is_current(page) -> None:
    """The gallery prints 'build <fingerprint>' (the files as the server imported them). If that is not the fingerprint of the files on disk now, the server is stale:
    exit 2 with the instruction, instead of reporting measurements of the old stylesheet."""
    text = page.inner_text("body")
    m = re.search(r"build ([0-9a-f]{12})", text)
    if m is None:
        print("The page does not show a 'build <fingerprint>' line: is this scripts/dark_mode_gallery.py? Start it with\n"
              "  streamlit run scripts/dark_mode_gallery.py --server.port 8599")
        sys.exit(2)
    if m.group(1) != digest():
        print(f"STALE SERVER: the page was built from files fingerprinted {m.group(1)}, the files on disk are {digest()}.\n"
              "Stop the gallery (Ctrl+C) and start it again before scanning; a server left running serves the old stylesheet.")
        sys.exit(2)
