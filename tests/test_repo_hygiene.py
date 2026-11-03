"""Guards against the things that made merges painful: committed conflict markers, line-ending churn, and
delivery snapshots applied on top of an out-of-date branch (see scripts/ and README "Working with git")."""
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
import apply_delivery as ad  # noqa: E402
import check_conflict_markers as cm  # noqa: E402

OPEN, MID, CLOSE = "<" * 7, "=" * 7, ">" * 7
needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _conflicted(name_a="ours", name_b="theirs") -> str:
    return f"keep\n{OPEN} HEAD\n{name_a}\n{MID}\n{name_b}\n{CLOSE} abc123\nkeep\n"


# ------------------------------------------------------------------------------------------ the marker check

def test_the_repository_itself_has_no_conflict_markers():
    """The one that would have caught the unresolved merge before CI and every later chat inherited it."""
    found = cm.find_conflicts(cm.candidate_files(ROOT))
    assert not found, f"unresolved conflict markers: {', '.join(f'{p.relative_to(ROOT)}:{m[0]}' for p, m in found.items())}"


def test_a_conflicted_file_is_found_with_every_marker_line(tmp_path):
    f = tmp_path / "a.py"
    f.write_text(_conflicted())
    assert cm.conflict_lines(f) == [2, 4, 6]


def test_crlf_files_are_found_too(tmp_path):
    f = tmp_path / "a.md"
    f.write_bytes(_conflicted().replace("\n", "\r\n").encode())
    assert cm.conflict_lines(f) == [2, 4, 6]


@pytest.mark.parametrize("text", [
    f"Title\n{MID}\nbody\n",                                   # a Markdown heading underline
    f"| a | b |\n{MID}\n",                                      # a table rule
    f"x = '{OPEN} not at the start'\n",                        # a marker in the middle of a line
    f"{OPEN}no space after the marker\n{CLOSE}also none\n",    # not the exact marker shape
    f"{OPEN} only an opening line\nnothing closes it\n",       # one side alone is not a conflict
])
def test_things_that_merely_look_like_markers_are_left_alone(tmp_path, text):
    f = tmp_path / "doc.md"
    f.write_text(text)
    assert cm.conflict_lines(f) == []


def test_binary_and_missing_files_are_ignored(tmp_path):
    b = tmp_path / "x.bin"
    b.write_bytes(b"\0" + _conflicted().encode())
    assert cm.conflict_lines(b) == [] and cm.conflict_lines(tmp_path / "missing") == []


def test_main_reports_the_file_and_exits_nonzero(tmp_path, capsys):
    bad, good = tmp_path / "bad.py", tmp_path / "good.py"
    bad.write_text(_conflicted())
    good.write_text("x = 1\n")
    assert cm.main([str(good)]) == 0
    assert cm.main([str(good), str(bad)]) == 1
    assert "bad.py" in capsys.readouterr().err


def test_line_endings_are_pinned_for_everyone():
    attrs = (ROOT / ".gitattributes").read_text()
    assert "* text=auto eol=lf" in attrs and "*.png  binary" in attrs


def test_ci_and_precommit_run_the_check():
    assert "check_conflict_markers.py" in (ROOT / ".github" / "workflows" / "tests.yml").read_text()
    assert "check_conflict_markers.py" in (ROOT / ".pre-commit-config.yaml").read_text()


# ------------------------------------------------------------------------------------------ applying a delivery

def _git(cwd, *args):
    return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false", *args],
                          cwd=cwd, capture_output=True, text=True, check=True).stdout


def _write(base: Path, files: dict[str, str]) -> None:
    for rel, text in files.items():
        p = base / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text, newline="\n")


@pytest.fixture
def repo(tmp_path):
    """A clone of a 'remote' with a few project files, committed and pushed."""
    if shutil.which("git") is None:
        pytest.skip("git is not installed")
    remote = tmp_path / "remote.git"
    _git(tmp_path, "init", "--bare", "-b", "main", str(remote))
    work = tmp_path / "work"
    _git(tmp_path, "clone", str(remote), str(work))
    _git(work, "checkout", "-b", "main")
    _write(work, {"app.py": "print('v1')\n", "lib/util.py": "A = 1\n", "old_test.py": "pass\n", "keep.md": "keep\n",
                  ".gitignore": "data/\n"})
    _git(work, "add", "-A")
    _git(work, "commit", "-m", "base")
    _git(work, "push", "-u", "origin", "main")
    return work


def _delivery(tmp_path: Path, files: dict[str, str], as_zip=True) -> Path:
    src = tmp_path / "delivery" / "Math_Rep_System"
    _write(src, files)
    _write(src, {".git/HEAD": "ref: stale\n", "data/history.db": "x", "lib/__pycache__/u.pyc": "x"})   # must all be ignored
    if not as_zip:
        return src.parent
    z = tmp_path / "delivery.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for p in sorted(src.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(src.parent).as_posix())
    return z


NEW = {"app.py": "print('v2')\n", "lib/util.py": "A = 1\n", "keep.md": "keep\n", "lib/new.py": "B = 2\n", ".gitignore": "data/\n"}


@needs_git
def test_dry_run_changes_nothing_and_lists_the_plan(repo, tmp_path, capsys):
    assert ad.run(_delivery(tmp_path, NEW), repo, apply=False) == 0
    out = capsys.readouterr().out
    assert "lib/new.py" in out and "app.py" in out and "old_test.py" in out and "Dry run" in out
    assert (repo / "app.py").read_text() == "print('v1')\n" and not (repo / "lib" / "new.py").exists()


@needs_git
def test_apply_adds_changes_and_removes_deleted_files_but_never_touches_git_or_data(repo, tmp_path):
    (repo / "data").mkdir()
    (repo / "data" / "mine.db").write_text("my history")
    head = _git(repo, "rev-parse", "HEAD")
    assert ad.run(_delivery(tmp_path, NEW), repo, apply=True) == 0
    assert (repo / "app.py").read_text() == "print('v2')\n" and (repo / "lib" / "new.py").read_text() == "B = 2\n"
    assert not (repo / "old_test.py").exists()                         # the delivery dropped it, so it is gone, not left behind
    assert (repo / "data" / "mine.db").read_text() == "my history"     # ignored data is never overwritten or removed
    assert _git(repo, "rev-parse", "HEAD") == head and not _git(repo, "diff", "--cached")   # nothing committed or staged
    status = _git(repo, "status", "--porcelain")
    assert "M app.py" in status and "?? lib/new.py" in status and "D old_test.py" in status


@needs_git
def test_a_folder_works_as_well_as_a_zip(repo, tmp_path):
    assert ad.run(_delivery(tmp_path, NEW, as_zip=False), repo, apply=True) == 0
    assert (repo / "lib" / "new.py").exists()


@needs_git
def test_a_second_apply_has_nothing_to_do(repo, tmp_path, capsys):
    z = _delivery(tmp_path, NEW)
    ad.run(z, repo, apply=True)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", "delivery")
    capsys.readouterr()
    assert ad.run(z, repo, apply=True) == 0 and "Nothing to do" in capsys.readouterr().out


@needs_git
def test_line_ending_differences_alone_are_not_changes(repo, tmp_path, capsys):
    (repo / "keep.md").write_bytes(b"keep\r\n")
    _git(repo, "-c", "core.autocrlf=false", "add", "-A")
    _git(repo, "-c", "core.autocrlf=false", "commit", "-m", "crlf")
    _git(repo, "push")
    assert ad.run(_delivery(tmp_path, {"app.py": "print('v1')\n", "lib/util.py": "A = 1\n", "old_test.py": "pass\n",
                                       "keep.md": "keep\n", ".gitignore": "data/\n"}), repo, apply=False) == 0
    assert "Nothing to do" in capsys.readouterr().out


@needs_git
def test_uncommitted_work_stops_it(repo, tmp_path):
    (repo / "app.py").write_text("print('my edit')\n")
    with pytest.raises(ad.DeliveryError, match="uncommitted"):
        ad.run(_delivery(tmp_path, NEW), repo, apply=True)
    assert (repo / "app.py").read_text() == "print('my edit')\n"


@needs_git
def test_being_behind_origin_stops_it_which_is_what_prevents_forked_history(repo, tmp_path):
    other = tmp_path / "other"
    _git(tmp_path, "clone", str(tmp_path / "remote.git"), str(other))
    _write(other, {"from_elsewhere.py": "x = 1\n"})
    _git(other, "add", "-A")
    _git(other, "commit", "-m", "work done on another machine")
    _git(other, "push")
    with pytest.raises(ad.DeliveryError, match=r"pull --rebase"):
        ad.run(_delivery(tmp_path, NEW), repo, apply=True)
    assert (repo / "app.py").read_text() == "print('v1')\n"


@needs_git
def test_a_delivery_that_would_delete_most_of_the_project_is_refused(repo, tmp_path):
    with pytest.raises(ad.DeliveryError, match="wrong snapshot"):
        ad.run(_delivery(tmp_path, {"app.py": "print('v1')\n"}), repo, apply=True)
    assert (repo / "lib" / "util.py").exists()


@needs_git
def test_a_zip_that_is_not_the_project_is_refused(repo, tmp_path):
    z = tmp_path / "other.zip"
    with zipfile.ZipFile(z, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises(ad.DeliveryError, match="app.py"):
        ad.run(z, repo, apply=True)
