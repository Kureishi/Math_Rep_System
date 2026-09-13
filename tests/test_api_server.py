"""Tests for api_server.py -- the REST API surface over the core engine.
Uses FastAPI's TestClient (in-process, no running server needed).

/solve's SUCCESS path isn't tested here: verify() performs its own
independent LLM re-solve (see verifier.py), so a full success run needs
LM Studio actually reachable -- not available in a CI/test environment,
and not something to fake with a mock here (that would test the mock,
not this API layer). What IS tested: malformed input is rejected with
a clear 422 before ever reaching verify(), and an unreachable LM Studio
produces the documented 503 rather than a raw connection traceback --
both fully deterministic, no external service needed."""
from fastapi.testclient import TestClient

from api_server import app

client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


# --------------------------------------------------------------------- /equivalence
def test_equivalence_true_case():
    response = client.post("/equivalence", json={"expr1": "sin(x)**2+cos(x)**2", "expr2": "1"})
    assert response.status_code == 200
    body = response.json()
    assert body["equivalent"] is True
    assert body["method"] == "symbolic"


def test_equivalence_false_case():
    response = client.post("/equivalence", json={"expr1": "x", "expr2": "x+1"})
    assert response.status_code == 200
    assert response.json()["equivalent"] is False


def test_equivalence_missing_field_returns_422():
    response = client.post("/equivalence", json={"expr1": "x"})
    assert response.status_code == 422


# --------------------------------------------------------------------- /curve-fit
def test_curve_fit_linear():
    response = client.post("/curve-fit", json={"xs": [1, 2, 3, 4, 5], "ys": [2, 4, 6, 8, 10],
                                                  "family": "linear"})
    assert response.status_code == 200
    body = response.json()
    assert body["family"] == "linear"
    assert abs(body["parameters"]["c1"] - 2.0) < 1e-6
    assert body["r_squared"] > 0.999


def test_curve_fit_best_fit_default():
    response = client.post("/curve-fit", json={"xs": [1, 2, 3, 4, 5], "ys": [2, 4, 6, 8, 10]})
    assert response.status_code == 200
    assert response.json()["r_squared"] > 0.999


def test_curve_fit_unknown_family_returns_422():
    response = client.post("/curve-fit", json={"xs": [1, 2, 3], "ys": [1, 2, 3], "family": "quadratic"})
    assert response.status_code == 422


def test_curve_fit_mismatched_lengths_returns_422():
    response = client.post("/curve-fit", json={"xs": [1, 2, 3], "ys": [1, 2], "family": "linear"})
    assert response.status_code == 422


# --------------------------------------------------------------------- /dimensional-analysis
def test_dimensional_analysis_feasible():
    response = client.post("/dimensional-analysis",
                             json={"inputs": {"m": "kg", "a": "m/s^2"}, "target_unit": "N"})
    assert response.status_code == 200
    body = response.json()
    assert body["feasible"] is True
    assert body["particular_solution"] is not None
    # every value must be a plain JSON number (the Fraction-serialization
    # bug this test guards against would have made this response fail
    # to serialize at all, not just return a wrong value)
    assert all(isinstance(v, (int, float)) for v in body["particular_solution"].values())


def test_dimensional_analysis_infeasible():
    response = client.post("/dimensional-analysis",
                             json={"inputs": {"t": "s"}, "target_unit": "kg"})
    assert response.status_code == 200
    assert response.json()["feasible"] is False


# --------------------------------------------------------------------- /solve
def test_solve_malformed_payload_returns_422_before_any_llm_call():
    response = client.post("/solve", json={"problem": {"equations": [{"name": "eq1"}]}})
    assert response.status_code == 422
    assert "expression" in response.json()["detail"]


def test_solve_well_formed_payload_without_lm_studio_returns_503():
    """The documented, deterministic behavior when LM Studio isn't
    reachable -- a clear 503, not a raw connection-error traceback."""
    payload = {
        "problem": {
            "variables": [{"symbol": "F", "meaning": "force"},
                           {"symbol": "m", "meaning": "mass", "known_value": 2.0},
                           {"symbol": "a", "meaning": "acceleration", "known_value": 3.0}],
            "equations": [{"name": "eq1", "expression": "Eq(F, m*a)"}],
            "solve_for": ["F"],
        }
    }
    response = client.post("/solve", json=payload)
    assert response.status_code == 503
    assert "LM Studio" in response.json()["detail"]
