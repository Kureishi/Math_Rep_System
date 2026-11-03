#!/usr/bin/env python
"""Apply a delivered snapshot (a .zip or an extracted folder) to this repository without forking its history.

WHY THIS EXISTS. Extracting a delivery over the project folder looks harmless, but it (a) leaves behind every file the
delivery deleted, (b) overwrites whatever .git came in the zip, and (c) is easy to do while origin has moved on -- and
the commit you then make sits on a different line of history from origin's. That is how the same work ends up
committed twice and every `git pull` becomes a merge with conflicts. This script refuses to proceed in that state.

    python scripts/apply_delivery.py Math_Rep_System.zip            # dry run: shows exactly what would change
    python scripts/apply_delivery.py Math_Rep_System.zip --apply    # do it, then review with `git status` / `git diff`

What it checks, in order, and stops at the first problem:
  1. your working tree is clean (nothing uncommitted that the delivery could overwrite);
  2. you are not behind or diverged from origin (it runs `git fetch` first) -- if you are, `git pull --rebase` first;
  3. after applying, no conflict markers remain anywhere.
It never commits, pushes or touches .git: the commit is yours to make, on top of an up-to-date branch.
"""
import argparse
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import check_conflict_markers as markers  # noqa: E402

# Never part of a delivery, whatever the zip contains.
SKIP_TOP = {".git", "data", "__pycache__", ".mypy_cache", ".pytest_cache", ".hypothesis", "htmlcov"}
SKIP_NAMES = {".coverage"}
MAX_DELETE_FRACTION = 0.25      # deleting more than this share of the tracked files means the wrong snapshot was given
SHOW = 40


class DeliveryError(Exception):
    pass


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run git in `repo`; with check=True a failure raises DeliveryError carrying git's own message."""
    proc = subprocess.run(["git", *args], cwd=repo, capture_output=True, text=True)
    if check and proc.returncode:
        raise DeliveryError(f"`git {' '.join(args)}` failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def _snapshot_root(base: Path) -> Path:
    """The folder holding the project: `base` itself, or its only subfolder (zips wrap the project in one)."""
    if (base / "app.py").exists():
        return base
    entries = [p for p in base.iterdir() if not p.name.startswith("__MACOSX")]
    if len(entries) == 1 and entries[0].is_dir():
        return _snapshot_root(entries[0])
    raise DeliveryError("This does not look like the project: no app.py found at the top of the delivery.")


def _wanted(rel: Path) -> bool:
    return not (SKIP_TOP & set(rel.parts[:1])) and "__pycache__" not in rel.parts and rel.name not in SKIP_NAMES \
        and not rel.name.endswith(".pyc") and not rel.name.startswith(".coverage")


def snapshot_files(root: Path) -> dict[str, Path]:
    return {p.relative_to(root).as_posix(): p for p in sorted(root.rglob("*"))
            if p.is_file() and _wanted(p.relative_to(root))}


def _same(a: Path, b: Path) -> bool:
    """Equal contents, ignoring CRLF/LF (a Windows checkout and a delivery must not look different for that alone)."""
    return a.read_bytes().replace(b"\r\n", b"\n") == b.read_bytes().replace(b"\r\n", b"\n")


def check_repository_state(repo: Path) -> None:
    dirty = git(repo, "status", "--porcelain").stdout.strip()
    if dirty:
        lines = dirty.splitlines()
        raise DeliveryError("You have uncommitted changes; commit or stash them first, so the delivery cannot overwrite "
                            "them:\n  " + "\n  ".join(lines[:10]) + ("\n  ..." if len(lines) > 10 else ""))
    upstream = git(repo, "rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}", check=False)
    if upstream.returncode:
        print("Note: this branch has no upstream, so I cannot check whether you are up to date with origin.")
        return
    fetched = git(repo, "fetch", "--quiet", check=False)
    if fetched.returncode:
        print("Note: `git fetch` failed (offline?), so the up-to-date check below uses what you last fetched.")
    ahead, behind = map(int, git(repo, "rev-list", "--left-right", "--count", "HEAD...@{u}").stdout.split())
    if behind:
        how = "have diverged from" if ahead else "are behind"
        raise DeliveryError(f"You {how} {upstream.stdout.strip()} ({behind} commit(s) you do not have). Committing the delivery "
                            "now would put it on a different line of history from origin -- the source of most merge "
                            "conflicts. Run `git pull --rebase`, then run this again.")


def plan(repo: Path, root: Path) -> tuple[list[str], list[str], list[str]]:
    """(added, modified, deleted) relative paths."""
    files = snapshot_files(root)
    ignored = set()
    if files:
        out = subprocess.run(["git", "check-ignore", "--stdin", "-z"], cwd=repo, input="\0".join(files) + "\0",
                             capture_output=True, text=True).stdout
        ignored = {p for p in out.split("\0") if p}
    files = {rel: p for rel, p in files.items() if rel not in ignored}
    tracked = [p for p in git(repo, "ls-files", "-z").stdout.split("\0") if p]
    added = [rel for rel in files if not (repo / rel).exists()]
    modified = [rel for rel, p in files.items() if (repo / rel).exists() and not _same(p, repo / rel)]
    deleted = [rel for rel in tracked if rel not in files and _wanted(Path(rel)) and (repo / rel).exists()]
    return sorted(added), sorted(modified), sorted(deleted)


def _show(title: str, items: list[str]) -> None:
    if items:
        print(f"\n{title} ({len(items)}):")
        for rel in items[:SHOW]:
            print(f"  {rel}")
        if len(items) > SHOW:
            print(f"  ... and {len(items) - SHOW} more")


def run(source: Path, repo: Path, apply: bool, force_delete: bool = False) -> int:
    if not (repo / ".git").exists():
        raise DeliveryError(f"{repo} is not a git repository.")
    check_repository_state(repo)
    with tempfile.TemporaryDirectory() as tmp:
        if source.is_file():
            with zipfile.ZipFile(source) as z:
                z.extractall(tmp)
            root = _snapshot_root(Path(tmp))
        elif source.is_dir():
            root = _snapshot_root(source)
        else:
            raise DeliveryError(f"{source} does not exist.")
        added, modified, deleted = plan(repo, root)
        _show("New files", added)
        _show("Changed files", modified)
        _show("Files the delivery removes", deleted)
        if not (added or modified or deleted):
            print("Nothing to do: the project already matches this delivery.")
            return 0
        tracked_count = len([p for p in git(repo, "ls-files", "-z").stdout.split("\0") if p])
        if deleted and len(deleted) > MAX_DELETE_FRACTION * max(1, tracked_count) and not force_delete:
            raise DeliveryError(f"This would delete {len(deleted)} of {tracked_count} files, which looks like the wrong "
                                "snapshot. Check it, or pass --force-delete if it is intended.")
        if not apply:
            print(f"\nDry run: {len(added)} new, {len(modified)} changed, {len(deleted)} removed. Nothing was written. "
                  "Re-run with --apply to make these changes.")
            return 0
        for rel in added + modified:
            target = repo / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes((root / rel).read_bytes())
        for rel in deleted:
            (repo / rel).unlink()
    bad = markers.find_conflicts(markers.candidate_files(repo))
    if bad:
        raise DeliveryError("The delivery left conflict markers in: " + ", ".join(str(p.relative_to(repo)) for p in bad))
    print(f"\nApplied: {len(added)} new, {len(modified)} changed, {len(deleted)} removed. Nothing is committed or staged.\n"
          "Next:  git status   (review)   then   git add -A && git commit -m \"...\" && git push")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Apply a delivered snapshot to this repository safely.")
    ap.add_argument("source", type=Path, help="the delivery: a .zip, or a folder")
    ap.add_argument("--apply", action="store_true", help="make the changes (default is a dry run)")
    ap.add_argument("--force-delete", action="store_true", help="allow deleting more than a quarter of the files")
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parent.parent, help=argparse.SUPPRESS)
    args = ap.parse_args(argv)
    try:
        return run(args.source, args.repo.resolve(), args.apply, args.force_delete)
    except DeliveryError as exc:
        print(f"\nStopped: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
