"""
Worksheet/batch mode: solve a whole problem set -- pasted as multiple
problems, or a folder of photographed problems -- in one pass, producing
one combined report instead of requiring someone to run each problem
through the app individually and reassemble the results by hand. This
is a natural fit for a local tool with file access; it's the kind of
thing that's awkward for a per-query web calculator to offer at all.

Mirrors (rather than imports) the same extract -> verify -> retry ->
compute_steps pipeline app.py's single-problem flow uses, since that
flow lives as inline Streamlit script code, not an importable function
-- mirroring it here keeps batch mode from risking any change to the
already-working single-problem flow. Narration and scenario generation
are skipped by default (each is its own LLM round trip per problem;
across a whole batch that adds up, and neither changes whether a
problem's math is right), but can be turned on for a smaller batch
where the extra polish is worth the wait.
"""
from dataclasses import dataclass, field
import io
import re

from config import settings
from modules.equation_engine import extract_model, ProblemModel
from modules.llm_client import LMStudioClient, LLMOutputError
from modules.verifier import VerificationReport, verify
from modules.solver import SolutionStep, compute_steps, narrate_steps

_NUMBERED_ITEM = re.compile(r"^\s*(?:\d+[.)]|Problem\s+\d+[:.]?)\s+", re.IGNORECASE | re.MULTILINE)


@dataclass
class BatchItemResult:
    index: int
    problem_text: str
    model: ProblemModel | None = None
    report: VerificationReport | None = None
    steps: dict[str, list[SolutionStep]] = field(default_factory=dict)
    retries: int = 0
    error: str | None = None  # set instead of the above on total failure


def split_batch_text(raw_text: str) -> list[str]:
    """Splits pasted/extracted batch text into individual problems.
    Tried in order (first match wins, since each is a stronger signal
    than the next): a numbered-list pattern ("1. ...", "2) ...",
    "Problem 3: ...") -- the most common structure for a worksheet
    (typed OR extracted from a PDF, where blank-line spacing between
    problems is often lost in text extraction); an explicit "---"
    delimiter line; falling back to splitting on blank lines."""
    raw_text = raw_text.strip()
    if not raw_text:
        return []

    numbered_matches = list(_NUMBERED_ITEM.finditer(raw_text))
    if numbered_matches:
        parts = []
        for i, m in enumerate(numbered_matches):
            start = m.end()
            end = numbered_matches[i + 1].start() if i + 1 < len(numbered_matches) else len(raw_text)
            parts.append(raw_text[start:end].strip())
        return [p for p in parts if p]

    if any(line.strip() == "---" for line in raw_text.splitlines()):
        parts = []
        current: list[str] = []
        for line in raw_text.splitlines():
            if line.strip() == "---":
                parts.append("\n".join(current).strip())
                current = []
            else:
                current.append(line)
        parts.append("\n".join(current).strip())
    else:
        parts = [p.strip() for p in raw_text.split("\n\n")]
    return [p for p in parts if p]


def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Pulls all text out of an uploaded PDF worksheet, one page after
    another. Uses pypdf (already a dependency -- added earlier for
    merging batch PDF reports), so this doesn't introduce anything new.
    Returns whatever text pypdf can extract; a scanned/image-only PDF
    with no embedded text layer will come back empty rather than
    raising -- there's no OCR step here, only text extraction."""
    from pypdf import PdfReader
    reader = PdfReader(io.BytesIO(pdf_bytes))
    pages_text = [page.extract_text() or "" for page in reader.pages]
    return "\n".join(pages_text).strip()


def solve_one(client: LMStudioClient, index: int, problem_text: str,
              narrate: bool = False) -> BatchItemResult:
    """Runs one problem through the same extract/verify/retry pipeline
    the main app uses, catching everything so one bad problem in a batch
    of 20 doesn't take the other 19 down with it."""
    try:
        model = extract_model(client, problem_text)
        report = verify(model, client, problem_text)
        retries = 0
        while not report.passed and retries < settings.max_verification_retries:
            retries += 1
            model = extract_model(client, problem_text, retry_reason=report.failure_reason)
            report = verify(model, client, problem_text)

        steps = compute_steps(model)
        if narrate:
            steps = narrate_steps(client, model, steps)

        return BatchItemResult(index=index, problem_text=problem_text, model=model,
                                 report=report, steps=steps, retries=retries)
    except LLMOutputError as e:
        return BatchItemResult(index=index, problem_text=problem_text, error=str(e))
    except Exception as e:  # noqa: BLE001
        return BatchItemResult(index=index, problem_text=problem_text,
                                 error=f"Unexpected error: {e}")


def solve_batch(client: LMStudioClient, problem_texts: list[str],
                 narrate: bool = False, progress_callback=None) -> list[BatchItemResult]:
    """Solves every problem in problem_texts in order. progress_callback,
    if given, is called as progress_callback(done_count, total_count)
    after each problem -- e.g. to drive a Streamlit progress bar,
    without this module needing to import Streamlit itself."""
    results = []
    total = len(problem_texts)
    for i, text in enumerate(problem_texts):
        results.append(solve_one(client, i, text, narrate=narrate))
        if progress_callback:
            progress_callback(i + 1, total)
    return results


def batch_summary(results: list[BatchItemResult]) -> dict:
    """A quick at-a-glance rollup: how many solved cleanly, how many
    needed a retry, how many failed outright -- for the top of a batch
    report, before diving into each problem's own detail. A result with
    neither `error` nor `report` set (shouldn't happen via solve_one(),
    which always sets one or the other, but is reachable if a
    BatchItemResult is ever constructed some other way) counts as
    failed rather than silently vanishing from every bucket."""
    total = len(results)
    solved = sum(1 for r in results if r.report is not None and r.report.passed)
    needed_retry = sum(1 for r in results if r.retries > 0 and r.report is not None and r.report.passed)
    failed = sum(1 for r in results if r.error is not None or r.report is None or not r.report.passed)
    return {"total": total, "solved": solved, "needed_retry": needed_retry, "failed": failed}
