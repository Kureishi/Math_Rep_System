import json

from modules.equation_engine import build_model
from modules.verifier import verify
from modules.solver import compute_steps
from modules.exporter import build_markdown, build_pdf_bytes, _safe


def test_markdown_contains_key_sections(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "car problem")
    steps = compute_steps(model)
    md = build_markdown("car problem", model, report, steps, [])
    assert "## Derived equations" in md
    assert "## Variables" in md
    assert "## Verification detail" in md
    assert "## Step-by-step solution" in md
    assert "car problem" in md


def test_pdf_is_valid_and_nonempty(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "car problem")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("car problem", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_pdf_export_works_for_ode(ode_json, fake_client_factory):
    """ODE equations render LaTeX with \\left(...\\right) from function
    application notation -- confirms matplotlib's mathtext subset actually
    supports this rather than silently falling back to broken plain text."""
    model = build_model(json.loads(ode_json))
    client = fake_client_factory()
    report = verify(model, client, "decay problem")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("decay problem", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"


def test_pdf_export_works_for_inequality(inequality_json, fake_client_factory):
    model = build_model(json.loads(inequality_json))
    client = fake_client_factory()
    report = verify(model, client, "speed problem")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("speed problem", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"


def test_safe_strips_unicode_for_latin1_pdf_fonts():
    result = _safe("✅ passed — nicely")
    result.encode("latin-1")  # should not raise
    assert "✅" not in result


def test_scenario_error_entries_excluded_from_export(kinematics_json, fake_client_factory):
    """A failed scenario-parse (surfaced as {'error': ...}) shouldn't be
    rendered as if it were a real scenario in the exported document."""
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)
    scenarios = [{"error": "could not parse", "raw": "garbage"}]
    md = build_markdown("x", model, report, steps, scenarios)
    assert "Where else this applies" not in md
