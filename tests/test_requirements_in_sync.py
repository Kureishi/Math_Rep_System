"""The requirements files, the code and the docs must say the same thing.

The crash that prompted this: `modules/report_office.py` imported python-docx at the top, a machine that had not installed it lost the whole results view, and nothing
had ever checked that the packages the code imports are the ones the requirements files declare. Now something does."""
import ast
import sys
from pathlib import Path

import pytest

from modules import health, optional_deps

ROOT = Path(__file__).resolve().parent.parent
SKIP_DIRS = {".mypy_cache", "__pycache__", ".venv", "venv", ".git", ".pytest_cache", ".hypothesis", "node_modules"}

# an import name that is not its pip distribution's name (or is part of another distribution)
IMPORT_TO_DIST = {d.module: d.dist for d in optional_deps.DEPS} | {"mpl_toolkits": "matplotlib", "yaml": "PyYAML"}
# imported only if present, and installed from its own optional file (requirements-gpu.txt): deliberately not in requirements.txt
OPTIONAL_ELSEWHERE = {"cupy"}
# listed in requirements.txt although the code never `import`s it by that name -- and why, so a stale pin is not mistaken for one of these
USED_INDIRECTLY = {
    "httpx": "the HTTP client the OpenAI SDK uses (pinned: the SDK and httpx have had incompatible releases)",
    "openpyxl": "pandas writes .xlsx with it (the batch Excel export)",
    "pytesseract": None,       # imported lazily by modules/ocr.py: must be found by the scan, not allow-listed
}
USED_INDIRECTLY.pop("pytesseract")


def _py_files():
    for f in ROOT.rglob("*.py"):
        if not (set(f.relative_to(ROOT).parts) & SKIP_DIRS):
            yield f


def _local_names() -> set[str]:
    names = {p.stem for p in ROOT.glob("*.py")} | {p.name for p in ROOT.iterdir() if p.is_dir() and (p / "__init__.py").exists()}
    names |= {p.stem for p in (ROOT / "scripts").glob("*.py")}         # scripts/ is put on sys.path by the tests that use it
    return names | {"tests", "plugins", "scripts", "modules", "ui"}


def imported_third_party() -> dict[str, set[str]]:
    local = _local_names()
    found: dict[str, set[str]] = {}
    for f in _py_files():
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module.split(".")[0]]
            else:
                continue
            for n in names:
                if n not in sys.stdlib_module_names and n not in local and n != "__future__":
                    found.setdefault(n, set()).add(str(f.relative_to(ROOT)))
    return found


def declared() -> dict[str, set[str]]:
    """{distribution (lower-case): the requirements files that name it}"""
    out: dict[str, set[str]] = {}
    for req in ROOT.glob("requirements*.txt"):
        for name, _, _ in health.parse_requirements(req):
            out.setdefault(name.lower().replace("_", "-"), set()).add(req.name)
    return out


def test_every_package_the_code_imports_is_declared_in_a_requirements_file():
    decl = declared()
    missing = {}
    for module, where in sorted(imported_third_party().items()):
        if module in OPTIONAL_ELSEWHERE:
            continue
        dist = IMPORT_TO_DIST.get(module, module).lower().replace("_", "-")
        if dist not in decl:
            missing[module] = sorted(where)[:3]
    assert not missing, f"imported but not in any requirements file (add it, with a pin, to requirements.txt): {missing}"


def test_anything_the_runtime_needs_is_in_requirements_txt_not_only_the_dev_file():
    runtime = {n.lower().replace("_", "-") for n, _, _ in health.parse_requirements(ROOT / "requirements.txt")}
    modules_dir = {m for m, where in imported_third_party().items() if any(w.startswith(("modules/", "ui/", "app.py", "config.py", "cli.py")) for w in where)}
    gone = {m for m in modules_dir if m not in OPTIONAL_ELSEWHERE and IMPORT_TO_DIST.get(m, m).lower().replace("_", "-") not in runtime}
    # api_server.py's own dependencies live in requirements-api.txt; everything else the app imports must be in requirements.txt
    assert not gone, f"used by the app but only declared outside requirements.txt: {sorted(gone)}"


def test_no_requirement_is_stale():
    imported = {IMPORT_TO_DIST.get(m, m).lower().replace("_", "-") for m in imported_third_party()}
    stale = [n for n, _, _ in health.parse_requirements(ROOT / "requirements.txt")
             if n.lower().replace("_", "-") not in imported and n.lower() not in USED_INDIRECTLY]
    assert not stale, f"in requirements.txt but never imported (remove it, or add it to USED_INDIRECTLY with the reason): {stale}"


def test_the_registry_of_optional_packages_matches_requirements_txt_exactly():
    reqs = {n.lower() for n, _, _ in health.parse_requirements(ROOT / "requirements.txt")}
    registry = {d.dist.lower() for d in optional_deps.DEPS}
    assert registry <= reqs, f"registered but not required: {sorted(registry - reqs)}"
    extra = reqs - registry
    assert not extra, f"required but not in modules/optional_deps.py (so the UI cannot say what it is for): {sorted(extra)}"


def test_the_requirements_are_installed_at_the_pinned_versions():
    c = health.check_packages(ROOT / "requirements.txt")
    assert c.status in ("ok", "warn"), c.detail


def test_every_requirements_file_is_mentioned_in_the_docs():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    for req in ROOT.glob("requirements*.txt"):
        assert req.name in readme, f"README.md does not mention {req.name}"


def test_the_docs_describe_the_failure_handling_modules():
    text = (ROOT / "ARCHITECTURE.md").read_text(encoding="utf-8")
    for module in ("optional_deps.py", "llm_diagnosis.py", "safe.py", "diagnosis.py"):
        assert module in text, f"ARCHITECTURE.md does not mention {module}"
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    assert "Max response tokens" in readme and "LM_MAX_TOKENS" in readme


@pytest.mark.parametrize("name", ["optional_deps", "llm_diagnosis"])
def test_the_new_modules_do_not_import_streamlit(name):
    tree = ast.parse((ROOT / "modules" / f"{name}.py").read_text(encoding="utf-8"))
    imports = {n.split(".")[0] for node in ast.walk(tree) if isinstance(node, (ast.Import, ast.ImportFrom)) for n in ([a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""])}
    assert "streamlit" not in imports
