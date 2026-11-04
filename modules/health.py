"""
First-run health check -- Streamlit-free.

When the app does not work the cause is almost never the maths: LM Studio is not running, the model is not loaded, a
package is the wrong version, the data folder is read-only. This module checks each of those in turn and, for every
problem, says what is wrong AND what to do about it, so a first run (or a broken one) ends with a diagnosis rather than a
stack trace.

EVERY check is isolated: an exception inside one becomes a `fail` with the error, never a crash of the page, and the
rest still run. Nothing here changes any setting or data; the only side effect is a short-lived probe file in the data
folder to prove it is writable. The optional round trip to the language model is opt-in because it costs a real (if
tiny) generation.

`HealthReport.text()` is a plain-text report meant to be pasted into a bug report.
"""
import importlib.metadata as md
import platform
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from config import settings
from modules import gpu

STATUSES = ("ok", "warn", "fail", "info")
ICONS = {"ok": "✅", "warn": "🟡", "fail": "❌", "info": "ℹ️"}
ROOT = Path(__file__).resolve().parent.parent
MIN_PYTHON = (3, 12)
LOW_DISK_BYTES = 200 * 2 ** 20
SLOW_LLM_SECONDS = 25.0

# What the self-test problem must produce: a car going 8 -> 20 m/s in 6 s.
_SELF_TEST_JSON = {
    "problem_domain": "kinematics", "problem_type": "algebraic",
    "variables": [{"symbol": "v", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
                  {"symbol": "u", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
                  {"symbol": "t", "meaning": "time", "known_value": "6", "unit": "s"},
                  {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
                  {"symbol": "d", "meaning": "distance", "known_value": None, "unit": "m"}],
    "equations": [{"name": "final velocity equation", "kind": "equation", "expression": "Eq(v, a*t + u)", "derivation": "x"},
                  {"name": "displacement equation", "kind": "equation", "expression": "Eq(d, 0.5*a*t**2 + t*u)", "derivation": "x"}],
    "solve_for": ["a", "d"], "assumptions": ["uniform acceleration"],
}


@dataclass(frozen=True)
class Check:
    id: str
    title: str
    status: str                  # ok | warn | fail | info
    detail: str
    fix: str = ""
    seconds: float | None = None

    @property
    def icon(self) -> str:
        return ICONS[self.status]


@dataclass
class HealthReport:
    checks: list[Check] = field(default_factory=list)

    @property
    def overall(self) -> str:
        """fail if anything essential failed, else warn if anything is off, else ok."""
        statuses = {c.status for c in self.checks}
        return "fail" if "fail" in statuses else "warn" if "warn" in statuses else "ok"

    def by_id(self, check_id: str) -> Check:
        return next(c for c in self.checks if c.id == check_id)

    def headline(self) -> str:
        n = {s: sum(c.status == s for c in self.checks) for s in STATUSES}
        if self.overall == "ok":
            return f"Everything essential works ({n['ok']} checks passed)."
        return f"{n['fail']} problem(s) and {n['warn']} warning(s) found; {n['ok']} checks passed."

    def text(self) -> str:
        lines = [f"Math Representation System health report -- {time.strftime('%Y-%m-%d %H:%M:%S')}",
                 f"Python {platform.python_version()} on {platform.platform()}", self.headline(), ""]
        for c in self.checks:
            lines.append(f"[{c.status.upper():4}] {c.title}: {c.detail}" + (f"  ({c.seconds:.2f}s)" if c.seconds is not None else ""))
            if c.fix and c.status in ("fail", "warn"):
                lines.append(f"       fix: {c.fix}")
        return "\n".join(lines)


# ------------------------------------------------------------------------------------------ the checks

def parse_requirements(path: Path) -> list[tuple[str, str, str]]:
    """(name, operator, version) for each pinned or minimum requirement in a requirements file (others are skipped)."""
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.split("#")[0].strip()
        m = re.match(r"^([A-Za-z0-9_.\-]+)\s*(==|>=|~=)\s*([A-Za-z0-9_.\-]+)", line)
        if m:
            out.append((m.group(1), m.group(2), m.group(3)))
    return out


def _version_tuple(v: str) -> tuple[int, ...]:
    return tuple(int(p) for p in re.findall(r"\d+", v)[:4])


def check_python() -> Check:
    v = sys.version_info
    shown = f"{v[0]}.{v[1]}" + (f".{v[2]}" if len(v) > 2 else "")
    if (v[0], v[1]) >= MIN_PYTHON:
        return Check("python", "Python", "ok", shown)
    return Check("python", "Python", "warn", f"{shown} is older than the supported {MIN_PYTHON[0]}.{MIN_PYTHON[1]}",
                 f"Install Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]} or newer and recreate the virtual environment.")


def check_packages(requirements: Path | None = None) -> Check:
    req = requirements or ROOT / "requirements.txt"
    if not req.exists():
        return Check("packages", "Required packages", "info", "requirements.txt was not found, so versions were not compared.")
    missing, off = [], []
    for name, op, wanted in parse_requirements(req):
        try:
            have = md.version(name)
        except md.PackageNotFoundError:
            missing.append(name)
            continue
        if op == "==" and have != wanted:
            off.append(f"{name} {have} (pinned {wanted})")
        elif op in (">=", "~=") and _version_tuple(have) < _version_tuple(wanted):
            off.append(f"{name} {have} (needs ≥ {wanted})")
    if missing:
        return Check("packages", "Required packages", "fail", "missing: " + ", ".join(missing), "pip install -r requirements.txt")
    if off:
        return Check("packages", "Required packages", "warn", "different from requirements.txt: " + "; ".join(off),
                     "pip install -r requirements.txt  (the app usually still runs, but it is only tested with the pinned versions)")
    return Check("packages", "Required packages", "ok", "all installed at the versions in requirements.txt")


def check_symbolic_engine() -> Check:
    import sympy as sp
    from modules.timeout_utils import run_with_timeout
    t0 = time.perf_counter()
    x = sp.Symbol("x")
    roots = sorted(run_with_timeout(lambda: sp.solve(x ** 2 - 4, x), timeout=10, label="self-test"))
    integral = run_with_timeout(lambda: sp.integrate(sp.sin(x), x), timeout=10, label="self-test")
    dt = time.perf_counter() - t0
    if roots == [-2, 2] and integral == -sp.cos(x):
        return Check("sympy", "Symbolic engine", "ok", f"SymPy {sp.__version__} solves and integrates correctly", seconds=dt)
    return Check("sympy", "Symbolic engine", "fail", f"SymPy {sp.__version__} gave unexpected results ({roots}, {integral})",
                 "pip install --force-reinstall sympy", seconds=dt)


def check_pipeline() -> Check:
    """End to end without a language model: build a model from JSON, solve it, run every computed check, compare to the
    known answers (a = 2, d = 84)."""
    from modules.equation_engine import build_model
    from modules.solve_compare import deterministic_report
    t0 = time.perf_counter()
    report = deterministic_report(build_model(dict(_SELF_TEST_JSON)))
    dt = time.perf_counter() - t0
    a, d = report.sympy_numeric_answers.get("a"), report.sympy_numeric_answers.get("d")
    if report.passed and a is not None and d is not None and abs(a - 2.0) < 1e-9 and abs(d - 84.0) < 1e-9:
        return Check("pipeline", "Solve-and-verify pipeline", "ok", "a self-test problem solves to the right answer and passes every check", seconds=dt)
    return Check("pipeline", "Solve-and-verify pipeline", "fail", f"the self-test problem came out as a={a}, d={d} (expected 2 and 84)",
                 "Reinstall the dependencies; if it persists, report this health report.", seconds=dt)


def check_data_folder(data_dir: Path | None = None) -> Check:
    folder = data_dir or ROOT / "data"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=folder, prefix=".health-", delete=True) as probe:
            probe.write(b"ok")
        free = shutil.disk_usage(folder).free
    except OSError as exc:
        return Check("data", "Data folder", "fail", f"{folder} is not writable ({exc})",
                     "History, templates and settings are saved there: fix its permissions or move the app to a writable folder.")
    if free < LOW_DISK_BYTES:
        return Check("data", "Data folder", "warn", f"{folder} is writable but only {free // 2 ** 20} MB are free",
                     "Free some disk space; saving history may start to fail.")
    return Check("data", "Data folder", "ok", f"{folder} is writable ({free // 2 ** 30} GB free)")


def check_databases() -> Check:
    from modules import history, templates
    try:
        for mod in (history, templates):
            mod._connect().close()
    except Exception as exc:  # noqa: BLE001
        return Check("databases", "Databases", "fail", f"could not open {type(exc).__name__}: {exc}",
                     "Close other copies of the app; if the file is damaged, move data/*.db aside to start fresh.")
    return Check("databases", "Databases", "ok", "history and templates open and are up to date")


def check_lm_studio(client: Any) -> list[Check]:
    if client is None:
        return [Check("lm_studio", "LM Studio", "info", "not checked")]
    ok, msg = client.is_available()
    if not ok:
        return [Check("lm_studio", "LM Studio", "fail", msg,
                      "Start LM Studio, open the Developer tab and start the local server, then check LM_STUDIO_BASE_URL "
                      f"(currently {settings.lm_studio_base_url}).")]
    checks = [Check("lm_studio", "LM Studio", "ok", msg)]
    loaded = client.list_models()
    ok_r, msg_r = client.validate_model(settings.reasoning_model)
    checks.append(Check("reasoning_model", "Reasoning model", "ok" if ok_r else "fail", msg_r,
                        "" if ok_r else "Load that model in LM Studio, or set LM_REASONING_MODEL to one that is loaded."))
    ok_v, msg_v = client.validate_model(settings.vision_model)
    checks.append(Check("vision_model", "Vision model (photo input)", "ok" if ok_v else "warn", msg_v if ok_v else
                        f"{msg_v} Photo input is unavailable until it is.", "" if ok_v else
                        "Load a vision-capable model and set LM_VISION_MODEL; everything else works without it."))
    from modules.llm_client import pick_embedding_model
    emb = pick_embedding_model(loaded)
    checks.append(Check("embedding_model", "Embedding model (semantic search)", "ok" if emb else "info",
                        f"'{emb}' will be used" if emb else "none loaded: 'Have I solved this?' uses word and equation-shape matching only",
                        "" if emb else "Optional: load an embedding model (e.g. nomic-embed-text) for meaning-based matching."))
    return checks


def check_llm_round_trip(client: Any) -> Check:
    t0 = time.perf_counter()
    try:
        reply = client.chat(system="Reply with exactly one word.", user="Reply with the single word: ok", temperature=0.0)
    except Exception as exc:  # noqa: BLE001
        return Check("llm_round_trip", "Model round trip", "fail", f"{type(exc).__name__}: {exc}",
                     "Check that the model finished loading in LM Studio and that its context length is not zero.")
    dt = time.perf_counter() - t0
    if "ok" not in str(reply).lower():
        return Check("llm_round_trip", "Model round trip", "warn", f"the model answered {str(reply)[:60]!r} to a one-word request", seconds=dt)
    if dt > SLOW_LLM_SECONDS:
        return Check("llm_round_trip", "Model round trip", "warn", f"works, but a one-word reply took {dt:.0f}s",
                     "Use a smaller model or enable GPU offload in LM Studio; extraction makes several calls per problem.", seconds=dt)
    return Check("llm_round_trip", "Model round trip", "ok", f"the model answered in {dt:.1f}s", seconds=dt)


def check_gpu() -> Check:
    backend = gpu.available()
    return Check("gpu", "GPU acceleration", "ok" if backend.is_gpu else "info", gpu.describe(),
                 "" if backend.is_gpu else "Optional: see requirements-gpu.txt")


def check_latex() -> Check:
    found = shutil.which("pdflatex") or shutil.which("xelatex") or shutil.which("lualatex")
    return Check("latex", "LaTeX (PDF reports)", "ok" if found else "info", f"found {Path(found).name}" if found else
                 "no LaTeX installation found: reports export as LaTeX source, HTML and Word but not compiled PDF",
                 "" if found else "Optional: install TeX Live or MiKTeX.")


def _guard(check_id: str, title: str, fn) -> list[Check]:
    """Run one check; an exception becomes a failed check instead of an error page."""
    t0 = time.perf_counter()
    try:
        out = fn()
        return out if isinstance(out, list) else [out]
    except Exception as exc:  # noqa: BLE001 -- the health check must survive anything it is checking
        return [Check(check_id, title, "fail", f"the check itself failed: {type(exc).__name__}: {exc}",
                      "Report this health report.", seconds=time.perf_counter() - t0)]


def run_checks(client: Any = None, include_llm_round_trip: bool = False, data_dir: Path | None = None,
               requirements: Path | None = None) -> HealthReport:
    """Run every check. `client` is the LM Studio client (None skips the language-model checks)."""
    checks: list[Check] = []
    for check_id, title, fn in (
        ("python", "Python", check_python),
        ("packages", "Required packages", lambda: check_packages(requirements)),
        ("sympy", "Symbolic engine", check_symbolic_engine),
        ("pipeline", "Solve-and-verify pipeline", check_pipeline),
        ("data", "Data folder", lambda: check_data_folder(data_dir)),
        ("databases", "Databases", check_databases),
        ("lm_studio", "LM Studio", lambda: check_lm_studio(client)),
    ):
        checks += _guard(check_id, title, fn)
    if include_llm_round_trip and client is not None and any(c.id == "reasoning_model" and c.status == "ok" for c in checks):
        checks += _guard("llm_round_trip", "Model round trip", lambda: check_llm_round_trip(client))
    checks += _guard("gpu", "GPU acceleration", check_gpu)
    checks += _guard("latex", "LaTeX (PDF reports)", check_latex)
    return HealthReport(checks)
