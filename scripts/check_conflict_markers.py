#!/usr/bin/env python
"""Fail if any file still contains unresolved git merge-conflict markers.

Used three ways, so a half-finished merge cannot slip through anywhere:
  * as a pre-commit hook (.pre-commit-config.yaml) -- before a commit exists;
  * as the first step of CI (.github/workflows/tests.yml) -- with a message that says what to do;
  * by tests/test_repo_hygiene.py -- so a plain `pytest` catches it too.

A file counts as conflicted when it has BOTH an opening marker line and a closing marker line. (A lone line of "=======",
a Markdown heading underline or a table rule, is not a conflict, and a marker in the middle of a line is just text.)

    python scripts/check_conflict_markers.py            # every file git knows about (tracked + untracked, not ignored)
    python scripts/check_conflict_markers.py a.py b.md  # just these files (what pre-commit passes)
"""
import subprocess
import sys
from pathlib import Path

OPEN, CLOSE = "<" * 7, ">" * 7
MAX_BYTES = 5_000_000
SKIP_DIRS = {".git", "__pycache__", ".mypy_cache", ".pytest_cache", ".hypothesis", "htmlcov", "node_modules", "data"}


def _is_marker(line: str, marker: str) -> bool:
    return line == marker or line.startswith(marker + " ")


def conflict_lines(path: Path) -> list[int]:
    """1-based line numbers of the conflict markers in `path` ([] if it is clean, unreadable or binary)."""
    try:
        if path.stat().st_size > MAX_BYTES:
            return []
        raw = path.read_bytes()
    except OSError:
        return []
    if b"\0" in raw[:8192]:
        return []
    lines = raw.decode("utf-8", errors="replace").splitlines()
    opens = [i + 1 for i, ln in enumerate(lines) if _is_marker(ln, OPEN)]
    closes = [i + 1 for i, ln in enumerate(lines) if _is_marker(ln, CLOSE)]
    if not opens or not closes:
        return []
    return sorted(opens + closes + [i + 1 for i, ln in enumerate(lines) if ln == "=" * 7])


def candidate_files(root: Path) -> list[Path]:
    """Files git would commit (tracked, plus untracked and not ignored); every file under `root` outside a git repo."""
    try:
        out = subprocess.run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"], cwd=root,
                             capture_output=True, check=True).stdout
        return [root / p for p in out.decode("utf-8", errors="replace").split("\0") if p]
    except (OSError, subprocess.CalledProcessError):
        return [p for p in root.rglob("*") if p.is_file() and not SKIP_DIRS & set(p.relative_to(root).parts)]


def find_conflicts(files: list[Path]) -> dict[Path, list[int]]:
    return {f: marks for f in files if f.is_file() and (marks := conflict_lines(f))}


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    root = Path(__file__).resolve().parent.parent
    files = [Path(a) for a in args] if args else candidate_files(root)
    found = find_conflicts(files)
    if not found:
        return 0
    print("Unresolved merge-conflict markers found -- finish the merge before committing:\n", file=sys.stderr)
    for f, marks in found.items():
        shown = ", ".join(map(str, marks[:8])) + (" ..." if len(marks) > 8 else "")
        print(f"  {f}: lines {shown}", file=sys.stderr)
    print("\nEdit each file so only the version you want remains (delete the marker lines), then `git add` it.\n"
          "To give up on the merge instead: `git merge --abort`.", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
