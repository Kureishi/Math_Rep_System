"""The proof card (what backs an answer) and the replay panel (do an exported bundle's checks still come out the same).
The logic is in modules/proof_card.py; this file only draws it."""
import streamlit as st

from modules import proof_card as pc
from modules.app_logging import logger
from modules.equation_engine import ProblemModel
from modules.verifier import VerificationReport
from ui.theme import badge_row

_KIND = {"verified": "pass", "caution": "warn", "failed": "fail"}


def render_proof_card(model: ProblemModel, report: VerificationReport) -> None:
    """A one-line trust verdict, with the reasoning behind it in an expander. A failure to build the card must never
    stop the results from showing, so it is logged and skipped."""
    try:
        summary = pc.trust_summary(model, report)
    except Exception:  # noqa: BLE001
        logger.exception("proof card could not be built")
        return
    badge_row([(f"{summary.icon} {summary.headline}", _KIND[summary.level])])
    with st.expander("🧾 Proof card: what backs this answer, and what does not"):
        if summary.reasons:
            st.markdown("**What backs it**")
            for line in summary.reasons:
                st.markdown(f"- {line}")
        if summary.audits:
            st.markdown("**Each equation, with the answer substituted back in**")
            for a in summary.audits:
                st.markdown(f"- {pc.ICONS[a.status]} `{a.name}`: {a.detail}")
        if summary.limits:
            st.markdown("**What it does not establish**" if summary.level != "failed" else "**What to fix**")
            for line in summary.limits:
                st.markdown(f"- {line}")
        if summary.categories:
            st.caption("Checks by category: " + " · ".join(f"{name} {p}/{t}" for name, (p, t) in summary.categories.items()))
        st.caption("Export the problem in a project bundle (sidebar → Project) and use **Replay** there to confirm that "
                   "every computed check still comes out the same.")


def render_replay(bundle_text: str) -> None:
    """Replay a bundle's checks and show, per problem, whether they reproduced."""
    result = pc.replay_bundle(bundle_text)
    for err in result.errors:
        st.error(err)
    if not result.items:
        return
    counts = result.counts()
    summary = f"{counts['reproduced']} reproduced · {counts['differs']} differ · {counts['not replayable']} not replayable"
    (st.success if result.ok else st.warning)(summary)
    for item in result.items:
        icon = {"reproduced": "✅", "differs": "❌", "not replayable": "⚪"}[item.status]
        with st.expander(f"{icon} {item.title}"):
            for line in item.details:
                st.caption(line)


def render_manifest_replay(name: str, data: bytes) -> None:
    """Find the replay manifest in an uploaded report, check it is intact, replay its computed checks and say whether they came out the same."""
    try:
        manifest = pc.extract_manifest(pc.text_of_report_file(name, data))
    except Exception as exc:  # noqa: BLE001 -- an unreadable file is reported, never a traceback
        st.error(f"Could not read that file: {type(exc).__name__}: {exc}")
        return
    if manifest is None:
        st.warning("No replay manifest was found in that file. Export the report with the appendix ticked (the lab-report and homework themes include it).")
        return
    result = pc.verify_manifest(manifest)
    (st.success if result.ok else st.error)(f"{'✅' if result.ok else '⚠️'} {result.headline}")
    st.caption(result.integrity_detail)
    if result.replay is not None:
        for line in result.replay.details:
            st.caption(line)
    if result.trust_level:
        st.caption(f"Trust level of the replayed checks: {result.trust_level}. (The independent re-solve asks the language model and is not replayed.)")
    st.caption("Problem: " + str(manifest["content"]["problem_text"])[:200])
