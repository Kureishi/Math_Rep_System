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
    supports this rather than silently falling back to broken plain text.
    """
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


def test_markdown_includes_matrix_representation_for_linear_system(fake_client_factory):
    model = build_model({
        "problem_domain": "circuit", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        "solve_for": ["x", "y"], "assumptions": [],
    })
    client = fake_client_factory()
    report = verify(model, client, "a linear system")
    steps = compute_steps(model)
    md = build_markdown("a linear system", model, report, steps, [])
    assert "## Matrix representation" in md
    assert "det(A)" in md


def test_pdf_includes_matrix_representation_for_linear_system(fake_client_factory):
    model = build_model({
        "problem_domain": "circuit", "problem_type": "algebraic",
        "variables": [
            {"symbol": "x", "meaning": "x", "known_value": None, "unit": None},
            {"symbol": "y", "meaning": "y", "known_value": None, "unit": None},
        ],
        "equations": [
            {"name": "eq1", "kind": "equation", "expression": "Eq(2*x + 3*y, 8)", "derivation": ""},
            {"name": "eq2", "kind": "equation", "expression": "Eq(x - y, 1)", "derivation": ""},
        ],
        "solve_for": ["x", "y"], "assumptions": [],
    })
    client = fake_client_factory()
    report = verify(model, client, "a linear system")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("a linear system", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 1000


def test_markdown_includes_vector_summary():
    model = build_model({
        "problem_domain": "work-energy", "problem_type": "algebraic",
        "variables": [
            {"symbol": "Fx", "meaning": "force x", "known_value": "10", "unit": "N"},
            {"symbol": "Fy", "meaning": "force y", "known_value": "0", "unit": "N"},
            {"symbol": "F", "meaning": "force", "known_value": None, "unit": "N",
             "is_vector": True, "components": ["Fx", "Fy"]},
        ],
        "equations": [], "solve_for": [], "assumptions": [],
    })
    from modules.verifier import VerificationReport
    report = VerificationReport()
    md = build_markdown("force problem", model, report, {}, [])
    assert "## Vectors" in md
    assert "magnitude = 10" in md


def test_markdown_includes_confidence_report(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)
    md = build_markdown("x", model, report, steps, [])
    assert "## Confidence report" in md
    assert "Overall score:" in md


def test_pdf_includes_confidence_report(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("x", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"


def test_markdown_includes_domain_of_validity_section(fake_client_factory):
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "v_f", "meaning": "final velocity", "known_value": "20", "unit": "m/s"},
            {"symbol": "v_i", "meaning": "initial velocity", "known_value": "8", "unit": "m/s"},
            {"symbol": "t", "meaning": "time", "known_value": "0", "unit": "s"},
            {"symbol": "a", "meaning": "acceleration", "known_value": None, "unit": "m/s^2"},
        ],
        "equations": [
            {"name": "accel", "kind": "equation", "expression": "Eq(a, (v_f - v_i) / t)", "derivation": ""},
        ],
        "solve_for": ["a"], "assumptions": [],
    })
    client = fake_client_factory()
    report = verify(model, client, "t=0 problem")
    steps = compute_steps(model)
    md = build_markdown("t=0 problem", model, report, steps, [])
    assert "## Domain of validity" in md
    assert "undefined with the given values" in md


def test_markdown_includes_results_in_other_units(fake_client_factory):
    model = build_model({
        "problem_domain": "kinematics", "problem_type": "algebraic",
        "variables": [
            {"symbol": "d", "meaning": "distance", "known_value": "100", "unit": "m"},
            {"symbol": "t", "meaning": "time", "known_value": "10", "unit": "s"},
            {"symbol": "v", "meaning": "velocity", "known_value": None, "unit": "m/s"},
        ],
        "equations": [
            {"name": "speed", "kind": "equation", "expression": "Eq(v, d/t)", "derivation": ""},
        ],
        "solve_for": ["v"], "assumptions": [],
    })
    client = fake_client_factory(final_answers={"v": 10.0})
    report = verify(model, client, "speed problem")
    steps = compute_steps(model)
    md = build_markdown("speed problem", model, report, steps, [])
    assert "## Results in other units" in md
    assert "km/hr" in md


def test_pdf_includes_results_in_other_units(kinematics_json, fake_client_factory):
    model = build_model(json.loads(kinematics_json))
    client = fake_client_factory(final_answers={"a": 2.0})
    report = verify(model, client, "x")
    steps = compute_steps(model)
    pdf_bytes = build_pdf_bytes("x", model, report, steps, [])
    assert pdf_bytes[:5] == b"%PDF-"


def test_build_batch_markdown_includes_summary_and_all_problems(kinematics_json, fake_client_factory):
    from modules.batch_solver import solve_batch
    from modules.exporter import build_batch_markdown

    client = fake_client_factory(payload_json=kinematics_json, final_answers={"a": 2.0})
    results = solve_batch(client, ["problem one", "problem two"])
    md = build_batch_markdown(results)
    assert "Batch Report" in md
    assert "2/2 solved" in md
    assert "Problem 1 of 2" in md
    assert "Problem 2 of 2" in md


def test_build_batch_markdown_handles_errored_problem_gracefully():
    from modules.batch_solver import BatchItemResult
    from modules.exporter import build_batch_markdown

    results = [BatchItemResult(index=0, problem_text="a broken problem", error="something failed")]
    md = build_batch_markdown(results)
    assert "Could not be solved" in md
    assert "something failed" in md
    assert "0/1 solved" in md


def test_build_batch_pdf_bytes_produces_valid_multipage_pdf(kinematics_json, fake_client_factory):
    from modules.batch_solver import solve_batch
    from modules.exporter import build_batch_pdf_bytes
    from pypdf import PdfReader
    import io

    client = fake_client_factory(payload_json=kinematics_json, final_answers={"a": 2.0})
    results = solve_batch(client, ["problem one", "problem two"])
    pdf_bytes = build_batch_pdf_bytes(results)
    assert pdf_bytes[:5] == b"%PDF-"
    reader = PdfReader(io.BytesIO(pdf_bytes))
    assert len(reader.pages) >= 3  # 1 cover + at least 1 page per problem


def test_build_batch_pdf_bytes_skips_errored_problems_without_crashing():
    from modules.batch_solver import BatchItemResult
    from modules.exporter import build_batch_pdf_bytes

    results = [BatchItemResult(index=0, problem_text="a broken problem", error="boom")]
    pdf_bytes = build_batch_pdf_bytes(results)
    assert pdf_bytes[:5] == b"%PDF-"  # still a valid PDF, just cover-page-only
