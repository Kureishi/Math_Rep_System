"""The answer card panel and the replay-from-a-report panel work inside Streamlit."""
import io

from streamlit.testing.v1 import AppTest

from modules import answer_card as ac
from modules.quantum_common import CheckList


def _card_page():
    import streamlit as st
    from modules import answer_card as ac_
    from modules.quantum_common import CheckList as CL
    from ui.answer_card import render_card, render_lab_card
    ck = CL()
    ck.add("a", True, "fine")
    render_card(ac_.from_checks("Demo", ["x = 1 m"], ck, "a problem"), "demo")
    render_lab_card("Lab", ["y = 2"], ck, "demo_lab")
    st.write("done")


def test_the_card_panel_draws_the_image_a_download_button_and_the_caption():
    at = AppTest.from_function(_card_page, default_timeout=60).run()
    assert not at.exception and at.markdown[-1].value == "done"
    assert at.toggle(key="demo_dark") is not None and "✓ Verified" in at.code[0].value and "x = 1 m" in at.code[0].value


def _replay_page():
    import streamlit as st
    from ui.results.proof_card import render_manifest_replay
    from modules import proof_card as pc
    from tests.report_helpers import kinematics_ctx
    import dataclasses
    from modules import report_export as rx
    ctx = dataclasses.replace(kinematics_ctx(), theme="Lab report")
    f = rx.render_export("md", ctx)
    render_manifest_replay("report.md", f.data)
    render_manifest_replay("empty.md", b"no manifest here")
    tampered = f.data.decode().replace("A car accelerates", "A cat accelerates", 1)
    m = pc.extract_manifest(f.data.decode())
    m["content"]["problem_text"] = "edited"
    render_manifest_replay("tampered.md", (pc.MANIFEST_BEGIN + "\n" + pc.encode_manifest(m).split("\n", 1)[1]).encode())
    st.write(len(tampered))


def test_replaying_a_report_says_reproduced_missing_or_tampered():
    at = AppTest.from_function(_replay_page, default_timeout=60).run()
    assert not at.exception
    assert any("Re-verified" in s.value for s in at.success)
    assert any("No replay manifest" in w.value for w in at.warning)
    assert any("Tampered" in e.value or "Not a valid" in e.value or "edited" in c.value for e in at.error for c in at.caption)
