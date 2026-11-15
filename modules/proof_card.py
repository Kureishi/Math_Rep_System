"""
Proof-carrying answers -- Streamlit-free.

An answer here is shown WITH what backs it: a trust level, a badge on each step that can be checked, and, for anything
exported, a way to replay the checks and see that they still come out the same.

WHAT IS CHECKED, AND WHAT IS NOT. The checks confirm that the equations are consistent, dimensionally sound and
actually satisfied by the answer. The "equation audit" below is the sharpest of them: it substitutes the solved values
into EVERY original equation and measures the residual, which has nothing to do with how the answer was found. What
no check can confirm is that the equations capture what the problem MEANT: they come from a language model's reading of
the text. The independent re-solve (a second attempt from scratch, compared with the first) is evidence about that, not
proof, so the trust summary says so in its `limits` rather than leaving the badge to overclaim.

TRUST LEVELS
  verified   every check passed, every equation is satisfied by the answer, and the independent re-solve agrees
  caution    nothing failed, but something was not confirmed (no independent re-solve ran, or an equation could not be
             evaluated) -- the answer may well be right; it has less behind it
  failed     a check failed, or the answer does not satisfy one of the equations

REPLAY. An exported bundle (modules/project_bundle.py) holds each problem's equations and what its checks said. Replaying
rebuilds the model and runs every check that is COMPUTED rather than asked of a language model
(solve_compare.deterministic_report), then compares: same checks, same outcomes, same answers. The independent re-solve is
not replayed, because it asks a model; the replay says so. A mismatch means the environment (library versions) or the
bundle changed, which is exactly what a reproducibility check is for.
"""
import json
from dataclasses import dataclass, field
from typing import Any

import sympy as sp

from modules.equation_engine import ProblemModel, build_model
from modules.solve_compare import deterministic_report
from modules.verifier import CheckResult, VerificationReport

STATUSES = ("verified", "caution", "failed", "unchecked")
ICONS = {"verified": "✅", "caution": "🟡", "failed": "❌", "unchecked": "⚪"}
RESIDUAL_TOLERANCE = 1e-6               # relative; an exact solution substituted back leaves ~1e-12
REPLAY_TOLERANCE = 1e-6                 # relative difference allowed between a stored and a replayed answer
CROSS_CHECK = "Independent cross-check"


@dataclass(frozen=True)
class Badge:
    status: str
    label: str
    detail: str = ""

    @property
    def icon(self) -> str:
        return ICONS[self.status]

    def text(self) -> str:
        return f"{self.icon} {self.label}"


@dataclass(frozen=True)
class EquationAudit:
    name: str
    status: str                       # verified | failed | unchecked
    residual: float | None            # |lhs - rhs| / scale, when it could be evaluated
    detail: str


@dataclass
class TrustSummary:
    level: str                        # verified | caution | failed
    headline: str
    score: float
    reasons: list[str] = field(default_factory=list)       # what backs the answer
    limits: list[str] = field(default_factory=list)        # what it does NOT establish
    audits: list[EquationAudit] = field(default_factory=list)
    categories: dict[str, tuple[int, int]] = field(default_factory=dict)   # category -> (passed, total)

    @property
    def icon(self) -> str:
        return ICONS[self.level]


# ------------------------------------------------------------------------------------------ equation audit

def _known_values(model: ProblemModel) -> dict[sp.Symbol, float]:
    return {sp.Symbol(v.symbol): float(v.known_value) for v in model.variables
            if v.known_value is not None and not v.is_function and not v.is_vector}


def audit_equations(model: ProblemModel, answers: dict[str, float]) -> list[EquationAudit]:
    """Substitute the known values and the solved answers into every algebraic equation of the model and measure how far
    each is from holding. Differential equations, recurrences and inequalities are not audited here (their own checks
    cover them), and an equation with an unknown left over is reported `unchecked`, never silently passed."""
    subs: dict[sp.Symbol, float] = {**_known_values(model), **{sp.Symbol(k): v for k, v in answers.items()}}
    audits = []
    for eq in model.equations:
        if eq.kind != "equation" or not isinstance(eq.sympy_eq, sp.Equality):
            continue
        left, right = eq.sympy_eq.lhs.subs(subs), eq.sympy_eq.rhs.subs(subs)
        free = sorted(str(s) for s in (left - right).free_symbols)
        if free:
            audits.append(EquationAudit(eq.name, "unchecked", None, f"not evaluated: {', '.join(free)} has no value"))
            continue
        try:
            lv, rv = complex(sp.N(left)), complex(sp.N(right))
        except (TypeError, ValueError):
            audits.append(EquationAudit(eq.name, "unchecked", None, "could not be evaluated numerically"))
            continue
        residual = abs(lv - rv) / max(1.0, abs(lv), abs(rv))
        if residual <= RESIDUAL_TOLERANCE:
            audits.append(EquationAudit(eq.name, "verified", residual, f"both sides agree (relative residual {residual:.1e})"))
        else:
            audits.append(EquationAudit(eq.name, "failed", residual,
                                        f"the answer leaves a residual of {residual:.3g}: {lv.real:.6g} vs {rv.real:.6g}"))
    return audits


# ------------------------------------------------------------------------------------------ badges

def _cross_check_for(report: VerificationReport, target: str) -> CheckResult | None:
    return next((c for c in report.checks if c.label == f"{CROSS_CHECK}: {target}"), None)


def step_badge(description: str, target: str, report: VerificationReport, audits: list[EquationAudit]) -> Badge | None:
    """The badge for one solution step, or None when nothing independent can be said about that kind of step.

    Steps carry their text only, so the match is by what the step IS: restating an equation (audited by name), the
    numeric result (audited, plus the independent re-solve if one ran), or substituting units-bearing knowns (the
    dimensional checks). Other steps (intermediate algebra) get no badge rather than an invented one."""
    by_name = {a.name: a for a in audits}
    for prefix in ("Start from: ", "Start from constraint: "):
        if description.startswith(prefix):
            audit = by_name.get(description[len(prefix):].strip())
            if audit is None:
                return None
            return Badge(audit.status, "satisfied by the answer" if audit.status == "verified" else
                         "not satisfied by the answer" if audit.status == "failed" else "not evaluated", audit.detail)
    if description.startswith("Numeric result"):
        if any(a.status == "failed" for a in audits):
            return Badge("failed", "answer breaks an equation", "the answer leaves a residual in at least one equation")
        cross = _cross_check_for(report, target)
        audited = [a for a in audits if a.status == "verified"]
        if cross is not None and not cross.passed:
            return Badge("failed", "independent re-solve disagrees", cross.detail)
        if audited and cross is not None:
            return Badge("verified", "satisfies every equation; independent re-solve agrees", cross.detail)
        if audited:
            return Badge("caution", "satisfies every equation; no independent re-solve", "the equations hold, but a second "
                         "solve from scratch was not run to confirm they were read from the problem correctly")
        return Badge("unchecked", "not audited", "")
    if description.startswith("Substitute known values"):
        cat = report.confidence_report().categories.get("Dimensional consistency")
        if cat is None:
            return None
        return Badge("verified" if cat.all_passed else "failed", "units consistent" if cat.all_passed else "unit mismatch",
                     f"{cat.passed} of {cat.total} dimensional checks passed")
    return None


# ------------------------------------------------------------------------------------------ trust summary

def trust_summary(model: ProblemModel, report: VerificationReport) -> TrustSummary:
    conf = report.confidence_report()
    audits = audit_equations(model, report.sympy_numeric_answers)
    cats = {name: (c.passed, c.total) for name, c in conf.categories.items()}
    cross = [c for c in report.checks if c.label.startswith(CROSS_CHECK)]
    failed_audits = [a for a in audits if a.status == "failed"]
    verified_audits = [a for a in audits if a.status == "verified"]
    unchecked_audits = [a for a in audits if a.status == "unchecked"]

    reasons, limits = [], []
    if conf.total_count:
        reasons.append(f"{conf.passed_count} of {conf.total_count} verification checks passed ({conf.label}).")
    if verified_audits:
        reasons.append(f"The answer satisfies {len(verified_audits)} of {len(audits)} equation(s) when substituted back.")
    if cross and all(c.passed for c in cross):
        reasons.append(f"An independent re-solve from scratch agrees on {len(cross)} quantity/quantities.")

    if not report.passed or failed_audits:
        level, headline = "failed", "Not verified: a check failed"
        for c in conf.critical_failures[:3]:
            limits.append(f"{c.label}: {c.detail}")
        for a in failed_audits[:3]:
            limits.append(f"{a.name}: {a.detail}")
    elif cross and all(c.passed for c in cross) and verified_audits and not unchecked_audits:
        level, headline = "verified", "Verified: checks pass and an independent re-solve agrees"
    else:
        level, headline = "caution", "Passed, with less behind it"
        if not cross:
            limits.append("No independent re-solve was run, so nothing yet confirms the equations were read from the problem correctly.")
        if unchecked_audits:
            limits.append("Not evaluated: " + ", ".join(a.name for a in unchecked_audits[:4]) + ".")
        if not verified_audits and not unchecked_audits:
            limits.append("There was no algebraic equation to substitute back (differential, recurrence or inequality "
                          "problems are checked by their own tests).")
    limits.append("The equations are a language model's reading of the problem; the checks show the maths is consistent and "
                  "solved correctly, not that the equations mean what you meant." if level != "failed" else
                  "Fix the failing check before relying on this answer.")
    return TrustSummary(level, headline, conf.score, reasons, limits, audits, cats)


# ------------------------------------------------------------------------------------------ replay

@dataclass
class ReplayItem:
    title: str
    status: str                       # reproduced | differs | not replayable
    details: list[str] = field(default_factory=list)


@dataclass
class ReplayReport:
    items: list[ReplayItem] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors and all(i.status == "reproduced" for i in self.items)

    def counts(self) -> dict[str, int]:
        out = {"reproduced": 0, "differs": 0, "not replayable": 0}
        for i in self.items:
            out[i.status] += 1
        return out


def _close(a: float, b: float) -> bool:
    return abs(a - b) <= REPLAY_TOLERANCE * max(1.0, abs(a), abs(b))


def replay_record(rec: dict[str, Any]) -> ReplayItem:
    """Rebuild one stored problem and rerun every check that does not ask a language model."""
    title = str(rec.get("problem_text", "?"))[:60]
    try:
        model = build_model(rec["raw_json"])
        stored = rec["verification"]
        stored_checks = [c for c in stored["checks"] if not str(c["label"]).startswith(CROSS_CHECK)]
        stored_answers = stored.get("sympy_numeric_answers", {})
    except Exception as exc:  # noqa: BLE001 -- one unreadable record must not stop the rest
        return ReplayItem(title, "not replayable", [f"{type(exc).__name__}: {exc}"])
    try:
        fresh = deterministic_report(model)
    except Exception as exc:  # noqa: BLE001
        return ReplayItem(title, "not replayable", [f"re-running the checks failed: {type(exc).__name__}: {exc}"])

    diffs: list[str] = []
    new_by_label = {c.label: c for c in fresh.checks}
    for c in stored_checks:
        now = new_by_label.get(c["label"])
        if now is None:
            diffs.append(f"check no longer runs: {c['label']}")
        elif bool(c["passed"]) != now.passed:
            diffs.append(f"{c['label']}: was {'pass' if c['passed'] else 'fail'}, now {'pass' if now.passed else 'fail'}")
    for key, old in stored_answers.items():
        new = fresh.sympy_numeric_answers.get(key)
        if new is None:
            diffs.append(f"{key}: was {old:.6g}, now not solved")
        elif not _close(float(old), float(new)):
            diffs.append(f"{key}: was {old:.6g}, now {new:.6g}")
    skipped = len(stored["checks"]) - len(stored_checks)
    notes = [f"{len(stored_checks)} computed check(s) and {len(stored_answers)} answer(s) compared"]
    if skipped:
        notes.append(f"{skipped} independent re-solve check(s) not replayed (they ask the language model)")
    return ReplayItem(title, "differs" if diffs else "reproduced", diffs + notes if diffs else notes)


def replay_bundle(bundle_json: str) -> ReplayReport:
    from modules.project_bundle import BUNDLE_FORMAT_VERSION
    report = ReplayReport()
    try:
        bundle = json.loads(bundle_json)
    except json.JSONDecodeError as exc:
        report.errors.append(f"Not valid JSON: {exc}")
        return report
    if not isinstance(bundle, dict) or bundle.get("format_version") != BUNDLE_FORMAT_VERSION:
        report.errors.append(f"Unrecognized bundle format version "
                             f"{bundle.get('format_version') if isinstance(bundle, dict) else None!r} "
                             f"(expected {BUNDLE_FORMAT_VERSION}).")
        return report
    report.items = [replay_record(rec) for rec in bundle.get("history", [])]
    if not report.items:
        report.errors.append("The bundle contains no solved problems to replay.")
    return report


# ------------------------------------------------------------------------------------------ the replay manifest (re-verifying a REPORT)

MANIFEST_FORMAT = "mrs-replay-manifest"
MANIFEST_VERSION = 1
MANIFEST_BEGIN = "-----BEGIN MRS REPLAY MANIFEST-----"
MANIFEST_END = "-----END MRS REPLAY MANIFEST-----"
_LINE = 76


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def content_hash(content: dict[str, Any]) -> str:
    """SHA-256 of the manifest's content (the problem, its equations and what the checks said): the proof's identity."""
    import hashlib
    return hashlib.sha256(_canonical(content).encode("ascii")).hexdigest()


def build_manifest(problem_text: str, model: ProblemModel, report: VerificationReport, environment: dict[str, Any] | None = None) -> dict[str, Any]:
    """Everything needed to re-verify a report: the problem, the equations the model extracted, and what every check and answer was, plus a hash
    of that content (so an edited manifest is detected) and the environment it was produced in (which a replay may legitimately differ from)."""
    from dataclasses import asdict
    from modules import provenance
    content = {
        "problem_text": problem_text,
        "raw_json": model.raw_json,
        "verification": {
            "passed": bool(report.passed),
            "checks": [{"label": c.label, "passed": bool(c.passed), "detail": c.detail} if not hasattr(c, "__dataclass_fields__") else asdict(c) for c in report.checks],
            "sympy_numeric_answers": {k: float(v) for k, v in report.sympy_numeric_answers.items()},
            "llm_independent_answers": {k: float(v) for k, v in (report.llm_independent_answers or {}).items()},
        },
    }
    return {"format": MANIFEST_FORMAT, "version": MANIFEST_VERSION, "content": content, "content_sha256": content_hash(content),
            "environment": environment if environment is not None else provenance.capture()}


def encode_manifest(manifest: dict[str, Any]) -> str:
    """The manifest as a block of base64 lines between marker lines: it survives being pasted into Markdown, HTML, LaTeX, Word or a PDF."""
    import base64
    import zlib
    raw = base64.b64encode(zlib.compress(_canonical(manifest).encode("ascii"), 9)).decode("ascii")
    lines = [raw[i:i + _LINE] for i in range(0, len(raw), _LINE)]
    return "\n".join([MANIFEST_BEGIN, *lines, MANIFEST_END])


def extract_manifest(text: str) -> dict[str, Any] | None:
    """Find and decode a manifest block in any text (a report's Markdown, HTML, LaTeX or text extracted from a PDF or Word file); None if there is none."""
    import base64
    import re
    import zlib
    m = re.search(re.escape(MANIFEST_BEGIN) + r"(.*?)" + re.escape(MANIFEST_END), text, flags=re.S)
    if not m:
        return None
    payload = re.sub(r"[^A-Za-z0-9+/=]", "", m.group(1))
    try:
        data = json.loads(zlib.decompress(base64.b64decode(payload)).decode("ascii"))
    except Exception:  # noqa: BLE001 -- a damaged block is simply not a manifest
        return None
    return data if isinstance(data, dict) and data.get("format") == MANIFEST_FORMAT else None


def text_of_report_file(name: str, data: bytes) -> str:
    """The text of an uploaded report (Markdown, HTML, LaTeX, plain text, a PDF or a Word file), for finding a manifest in it."""
    lower = name.lower()
    if lower.endswith(".pdf"):
        import io
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(io.BytesIO(data)).pages)
    if lower.endswith(".docx"):
        import io
        from docx import Document
        return "\n".join(p.text for p in Document(io.BytesIO(data)).paragraphs)
    if lower.endswith((".zip",)):
        import io
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as z:
            return "\n".join(z.read(n).decode("utf-8", "replace") for n in z.namelist() if n.endswith(".tex"))
    return data.decode("utf-8", "replace")


@dataclass
class ManifestVerification:
    integrity_ok: bool
    integrity_detail: str
    replay: ReplayItem | None
    trust_level: str | None
    headline: str

    @property
    def ok(self) -> bool:
        return self.integrity_ok and self.replay is not None and self.replay.status == "reproduced"


def verify_manifest(manifest: dict[str, Any]) -> ManifestVerification:
    """Is the manifest intact (its hash matches its content), and do the computed checks still come out the same when replayed?"""
    content = manifest.get("content")
    if not isinstance(content, dict):
        return ManifestVerification(False, "The manifest has no content.", None, None, "Not a valid manifest")
    expected = manifest.get("content_sha256")
    actual = content_hash(content)
    if expected != actual:
        return ManifestVerification(False, f"The content hash does not match (recorded {str(expected)[:12]}, computed {actual[:12]}): the manifest was edited after it was made.",
                                    None, None, "Tampered or damaged manifest")
    item = replay_record(content)
    level = None
    if item.status != "not replayable":
        try:
            model = build_model(content["raw_json"])
            fresh = deterministic_report(model)
            level = trust_summary(model, fresh).level
        except Exception:  # noqa: BLE001
            level = None
    headline = {"reproduced": "Re-verified: every computed check and answer came out the same", "differs": "Replayed, but something differs",
                "not replayable": "Could not be replayed"}[item.status]
    return ManifestVerification(True, f"The content hash matches ({actual[:12]}…): the manifest is as it was made.", item, level, headline)
