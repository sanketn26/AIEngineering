"""API contract tests — these must pass in CI."""

from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_healthz():
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_triage_schema():
    r = client.post("/v1/triage", json={"text": "I forgot my password"})
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in {"billing", "shipping", "account", "product", "other"}
    assert body["priority"] in {"low", "medium", "high", "urgent"}
    assert isinstance(body["rationale"], str) and body["rationale"]
    assert isinstance(body["citations"], list)
    assert isinstance(body["proposed_actions"], list)
    assert isinstance(body["model_id"], str) and body["model_id"]
    assert isinstance(body["request_id"], str) and body["request_id"]


def test_triage_rejects_empty():
    r = client.post("/v1/triage", json={"text": ""})
    assert r.status_code == 422
