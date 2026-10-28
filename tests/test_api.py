"""Automated tests for FastAPI endpoints and API contract."""

import pytest
from fastapi.testclient import TestClient
from app.main import app

VALID_CUSTOMER_PAYLOAD = {
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 1,
    "PhoneService": "No",
    "MultipleLines": "No phone service",
    "InternetService": "DSL",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 29.85,
    "TotalCharges": 29.85,
    "customerID": "7590-VHVEG"
}


@pytest.fixture(scope="module")
def client():
    """Create a FastAPI TestClient with lifespan context."""
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    """Test GET / returns 200 and valid API metadata."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "running"
    assert "/docs" in data["docs_url"]
    assert "/health" in data["health_url"]
    assert "/predict" in data["predict_url"]


def test_health_endpoint(client):
    """Test GET /health returns 200 and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"
    assert "scikit_learn_version" in data


def test_predict_endpoint_valid(client):
    """Test POST /predict with a valid customer payload."""
    response = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert data["prediction"] in ["Yes", "No"]
    assert "churn_probability" in data
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["model_version"] == "1.0.0"


def test_predict_endpoint_string_total_charges(client):
    """Test POST /predict gracefully handles string-encoded TotalCharges."""
    payload = dict(VALID_CUSTOMER_PAYLOAD)
    payload["TotalCharges"] = " 123.45 "
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["prediction"] in ["Yes", "No"]


def test_predict_endpoint_missing_field(client):
    """Test POST /predict returns 422 when required fields are missing."""
    invalid_payload = {
        "gender": "Female",
        "tenure": 5
    }
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422


def test_predict_endpoint_invalid_categorical(client):
    """Test POST /predict returns 422 when an invalid categorical option is passed."""
    payload = dict(VALID_CUSTOMER_PAYLOAD)
    payload["Contract"] = "Lifetime"  # Invalid contract
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_negative_tenure(client):
    """Test POST /predict returns 422 when numeric constraints are violated."""
    payload = dict(VALID_CUSTOMER_PAYLOAD)
    payload["tenure"] = -10  # Negative tenure invalid
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_endpoint_unexpected_field(client):
    """Test POST /predict returns 422 when unexpected extra fields are supplied."""
    payload = dict(VALID_CUSTOMER_PAYLOAD)
    payload["unauthorized_field"] = "malicious_payload"
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_model_unavailable_returns_503(client):
    """Test POST /predict and /health behave gracefully when model is unavailable."""
    from unittest.mock import patch
    with patch("app.main.get_model", return_value=None):
        health_resp = client.get("/health")
        assert health_resp.status_code == 200
        assert health_resp.json()["status"] == "degraded"
        assert health_resp.json()["model_loaded"] is False

        predict_resp = client.post("/predict", json=VALID_CUSTOMER_PAYLOAD)
        assert predict_resp.status_code == 503
        assert "not loaded" in predict_resp.json()["detail"]
