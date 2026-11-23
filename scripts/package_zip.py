"""Build the delivery zip of the project, the same way every time: python scripts/package_zip.py [output.zip]   (default: ../Math_Rep_System.zip)

Left out: data/ (the user's own history.db, chains.db, ... must never travel with the code), caches (__pycache__, .pytest_cache, .mypy_cache, .hypothesis), virtual
environments, coverage output, the figure audit report, and any *.db / *.log file. `collect()` is what tests/test_package_zip.py checks.
"""
import os
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {"data", "venv", ".venv", "__pycache__", ".pytest_cache", ".mypy_cache", ".hypothesis", "htmlcov", ".git", "node_modules"}
SKIP_SUFFIXES = (".pyc", ".db", ".db-journal", ".db-wal", ".db-shm", ".log")
SKIP_FILES = {".coverage", "figure_layout_audit.txt"}


def collect(root: Path = ROOT) -> list[str]:
    """Relative POSIX paths of every file that belongs in the zip."""
    out = []
    for folder, dirs, files in os.walk(root):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)
        for name in sorted(files):
            if name in SKIP_FILES or name.endswith(SKIP_SUFFIXES):
                continue
            out.append((Path(folder) / name).relative_to(root).as_posix())
    return sorted(out)


def main() -> int:
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT.parent / "Math_Rep_System.zip"
    files = collect()
    with zipfile.ZipFile(target, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in files:
            z.write(ROOT / rel, rel)
    print(f"{len(files)} files -> {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
