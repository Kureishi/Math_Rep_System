"""
Which third-party packages the app uses, what each is FOR, and whether it is installed -- Streamlit-free.

Why this exists: a missing package used to surface as a raw `ModuleNotFoundError` traceback that took the whole page down (python-docx, imported at the
top of the report exporter, hid the entire results view). Now:

  * packages that only one feature needs (python-docx for Word, python-pptx for PowerPoint, fpdf2 for PDF, ...) are imported INSIDE the functions that use
    them, and `available()` / `missing_for()` say whether that feature can run, so the UI can disable one export format with the exact fix instead of crashing;
  * `explain_import_error()` turns any ModuleNotFoundError into "install <pip name> (needed for <feature>)";
  * the Health page and a start-up banner use the same registry, and tests/test_requirements_in_sync.py fails if the code imports a package that
    requirements.txt does not list, so this table, the code and the requirements cannot drift apart again.

`REQUIRED` packages are needed for the app to start at all; the rest degrade one feature.
"""
import importlib.util
from dataclasses import dataclass
from functools import lru_cache


@dataclass(frozen=True)
class Dep:
    module: str        # the name you import
    dist: str          # the name you pip install
    feature: str       # what stops working without it
    required: bool = False


DEPS: tuple[Dep, ...] = (
    Dep("streamlit", "streamlit", "the whole app", True),
    Dep("openai", "openai", "talking to LM Studio", True),
    Dep("httpx", "httpx", "talking to LM Studio and the REST API", True),
    Dep("sympy", "sympy", "all symbolic mathematics", True),
    Dep("mpmath", "mpmath", "arbitrary-precision numerics (also a dependency of sympy)", True),
    Dep("numpy", "numpy", "all numerical work", True),
    Dep("scipy", "scipy", "numerical solvers, fitting and the simulation labs", True),
    Dep("pandas", "pandas", "tables and data import", True),
    Dep("plotly", "plotly", "every interactive chart", True),
    Dep("pydantic", "pydantic", "checking the model's JSON output", True),
    Dep("matplotlib", "matplotlib", "typeset equations in reports and the shareable answer card"),
    Dep("PIL", "Pillow", "images in reports, OCR and the answer card"),
    Dep("fpdf", "fpdf2", "PDF export"),
    Dep("docx", "python-docx", "Word (.docx) export"),
    Dep("pptx", "python-pptx", "PowerPoint (.pptx) export"),
    Dep("pypdf", "pypdf", "PDF import and re-verifying a PDF report"),
    Dep("openpyxl", "openpyxl", "Excel (.xlsx) export"),
    Dep("pytesseract", "pytesseract", "reading a photographed problem without a vision model (also needs the Tesseract program)"),
)
_BY_MODULE = {d.module: d for d in DEPS}


@lru_cache(maxsize=None)
def available(module: str) -> bool:
    """Is `module` importable? (Looked up without importing it, so this is cheap and has no side effects.)"""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ValueError):
        return False


def missing(required_only: bool = False) -> list[Dep]:
    return [d for d in DEPS if (d.required or not required_only) and not available(d.module)]


def missing_for(*modules: str) -> list[Dep]:
    """The packages among `modules` that are not installed (unknown names are looked up as plain modules)."""
    out = []
    for m in modules:
        if not available(m):
            out.append(_BY_MODULE.get(m) or Dep(m, m, "this feature"))
    return out


def install_hint(deps: list[Dep]) -> str:
    """The exact command, and what it is needed for."""
    if not deps:
        return ""
    names = " ".join(d.dist for d in deps)
    why = "; ".join(f"{d.dist}: {d.feature}" for d in deps)
    return f"pip install {names}   ({why}).   To install everything the app uses: pip install -r requirements.txt"


def explain_import_error(exc: BaseException) -> str | None:
    """A plain-language fix for a missing package, or None if `exc` is not an import error."""
    if not isinstance(exc, ImportError):
        return None
    name = (getattr(exc, "name", None) or "").split(".")[0]
    dep = _BY_MODULE.get(name)
    if dep is None:
        return f"The package '{name or 'unknown'}' could not be imported. Install it with pip, or run: pip install -r requirements.txt"
    return (f"The package **{dep.dist}** is not installed in this Python environment ({dep.feature} needs it). "
            f"Run `pip install {dep.dist}` (or `pip install -r requirements.txt` for everything), then restart the app.")


def reset_cache() -> None:
    """Forget cached lookups (tests; or after the person installs a package while the app is running)."""
    available.cache_clear()
