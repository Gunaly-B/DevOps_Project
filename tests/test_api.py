"""
Integration tests for FastAPI endpoints.
"""

import pytest
from fastapi.testclient import TestClient
from src.app import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "version" in data
    assert "model_loaded" in data


def test_predict_endpoint_valid_payload():
    payload = {
        "age": 57.0,
        "sex": 1.0,
        "cp": 4.0,
        "trestbps": 140.0,
        "chol": 241.0,
        "fbs": 0.0,
        "restecg": 1.0,
        "thalach": 123.0,
        "exang": 1.0,
        "oldpeak": 1.5,
        "slope": 2.0,
        "ca": 0.0,
        "thal": 7.0,
    }
    response = client.post("/predict", json=payload)

    # Status code can be 200 if trained or 503 if model not yet loaded
    assert response.status_code in [200, 503]

    if response.status_code == 200:
        data = response.json()
        assert "prediction" in data
        assert "probability" in data
        assert "heart_disease_risk" in data
        assert "model_version" in data


def test_predict_endpoint_invalid_payload():
    invalid_payload = {"age": "invalid_age"}
    response = client.post("/predict", json=invalid_payload)
    assert response.status_code == 422  # Unprocessable Entity
