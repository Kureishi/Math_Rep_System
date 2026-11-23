"""The stale-server guard of the browser scans (ui/build_stamp.py): the fingerprint follows the files that decide the look, and is stable when they do not change."""
import shutil
from pathlib import Path

from ui import build_stamp

ROOT = Path(__file__).resolve().parent.parent


def test_the_fingerprint_is_stable_and_follows_the_stylesheet(tmp_path):
    for name in build_stamp.FILES:
        dst = tmp_path / name
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(ROOT / name, dst)
    before = build_stamp.digest(tmp_path)
    assert before == build_stamp.digest(tmp_path) == build_stamp.digest(ROOT) and len(before) == 12
    (tmp_path / "ui/theme.py").write_text((tmp_path / "ui/theme.py").read_text(encoding="utf-8") + "\n# edited\n", encoding="utf-8")
    assert build_stamp.digest(tmp_path) != before


def test_every_file_in_the_fingerprint_exists():
    assert all((ROOT / n).is_file() for n in build_stamp.FILES)
