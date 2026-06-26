"""
test_api.py
Tests for the churn serving API: health, prediction correctness,
input validation, and metrics exposure.

Run:
    pytest tests/ -v
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from fastapi.testclient import TestClient
from app.main import app

VALID_PAYLOAD = {
    "gender": "Female", "SeniorCitizen": 0, "Partner": "No", "Dependents": "No",
    "tenure": 2, "PhoneService": "Yes", "MultipleLines": "No",
    "InternetService": "Fiber optic", "OnlineSecurity": "No", "OnlineBackup": "No",
    "DeviceProtection": "No", "TechSupport": "No", "StreamingTV": "Yes",
    "StreamingMovies": "Yes", "Contract": "Month-to-month", "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check", "MonthlyCharges": 95.50, "TotalCharges": 191.00,
}


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_predict_valid_payload(client):
    r = client.post("/predict", json=VALID_PAYLOAD)
    assert r.status_code == 200
    body = r.json()
    assert body["churn_prediction"] in ("Yes", "No")
    assert 0.0 <= body["churn_probability"] <= 1.0


def test_predict_high_risk_profile_flagged(client):
    # month-to-month, no support, new customer -> should skew toward churn
    r = client.post("/predict", json=VALID_PAYLOAD)
    body = r.json()
    assert body["churn_probability"] > 0.5


def test_predict_rejects_invalid_category(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["Contract"] = "Lifetime"  # not a valid enum value
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422  # Pydantic validation error


def test_predict_rejects_missing_field(client):
    bad_payload = dict(VALID_PAYLOAD)
    del bad_payload["tenure"]
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_predict_rejects_negative_tenure(client):
    bad_payload = dict(VALID_PAYLOAD)
    bad_payload["tenure"] = -5
    r = client.post("/predict", json=bad_payload)
    assert r.status_code == 422


def test_metrics_endpoint_exposes_prometheus_format(client):
    client.post("/predict", json=VALID_PAYLOAD)  # generate at least one metric
    r = client.get("/metrics")
    assert r.status_code == 200
    assert "churn_predictions_total" in r.text
    assert "churn_prediction_latency_seconds" in r.text


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "service" in r.json()
