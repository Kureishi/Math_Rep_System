"""The delivery zip never contains the person's own databases, caches or logs."""
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
spec = importlib.util.spec_from_file_location("package_zip", ROOT / "scripts" / "package_zip.py")
package_zip = importlib.util.module_from_spec(spec)          # type: ignore[arg-type]
spec.loader.exec_module(package_zip)                         # type: ignore[union-attr]


def test_user_data_and_caches_are_left_out(tmp_path):
    for rel in ("data/history.db", "data/app.log", "modules/__pycache__/x.pyc", ".pytest_cache/v", "venv/lib/x.py", "a.db", "figure_layout_audit.txt", ".coverage"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    for rel in ("app.py", "modules/m.py", ".streamlit/config.toml", "tests/test_x.py"):
        f = tmp_path / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x", encoding="utf-8")
    assert package_zip.collect(tmp_path) == [".streamlit/config.toml", "app.py", "modules/m.py", "tests/test_x.py"]


def test_the_real_tree_has_no_database_in_it():
    files = package_zip.collect(ROOT)
    assert "app.py" in files and ".streamlit/config.toml" in files
    assert not [f for f in files if f.startswith("data/") or f.endswith((".db", ".pyc", ".log"))]
