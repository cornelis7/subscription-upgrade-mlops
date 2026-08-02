"""
These tests are what the CI/CD pipeline (.github/workflows/ci-cd.yml) runs
on every push. If any of these fail, the pipeline blocks the Docker image
from being built -- this is what "continuous integration" means in
practice: broken code never reaches the deployment stage.
"""

import pytest
from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

VALID_PAYLOAD = {
    "monthly_usage_hours": 45.0,
    "days_since_signup": 180,
    "num_support_tickets": 2,
    "avg_session_minutes": 22.5,
    "num_referrals": 1,
    "discount_pct_used": 15.0,
}


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_model_info():
    r = client.get("/model-info")
    assert r.status_code == 200
    body = r.json()
    assert body["model_name"] == "subscription-upgrade-predictor"
    assert "production_version" in body


def test_predict_valid_payload():
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["will_upgrade"] in (True, False)
    assert 0.0 <= body["upgrade_probability"] <= 1.0
    assert isinstance(body["drift_warning"], bool)


def test_predict_rejects_invalid_input():
    bad_payload = {**VALID_PAYLOAD, "discount_pct_used": 150}  # over max=100
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422  # Pydantic validation error


def test_predict_rejects_missing_field():
    bad_payload = {k: v for k, v in VALID_PAYLOAD.items() if k != "num_referrals"}
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_drift_endpoint_shape():
    r = client.get("/monitoring/drift")
    assert r.status_code == 200
    body = r.json()
    assert "drifted" in body
    assert isinstance(body["drifted"], bool)
