"""Showing and downloading the shareable answer card (modules/answer_card.py)."""
import streamlit as st

from modules import answer_card as ac
from modules.app_logging import logger
from ui.cache import cached


def render_card(card: ac.AnswerCard, key: str) -> None:
    """The card as an image with a download button, the alt text to post with it, and a light/dark choice."""
    dark = st.toggle("Dark card", key=f"{key}_dark")
    try:
        png = cached(f"card_{key}", (card.alt_text, card.proof_id, dark), lambda: ac.render_png(card, dark=dark))
    except Exception:  # noqa: BLE001 -- a card is an extra: failing to draw it must not break the page
        logger.exception("could not draw the answer card")
        st.warning("The card could not be drawn.")
        return
    st.image(png, width="stretch")
    st.download_button("⬇️ Download the card (PNG)", data=png, file_name="answer_card.png", mime="image/png", key=f"{key}_dl")
    st.caption("Text to post with the image (also its description for screen readers)")
    st.code(card.caption, language=None)


def render_lab_card(title: str, answers: list[str], checks, key: str, problem: str = "") -> None:
    """A collapsed 'share as an image' panel for a lab result."""
    with st.expander("🖼️ Share this result as an image"):
        render_card(ac.from_checks(title, answers, checks, problem), key)
