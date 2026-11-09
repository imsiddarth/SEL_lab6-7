"""
Automated REST API tests using FastAPI TestClient.
Validates HTTP status codes, request schemas, business logic constraints, and error handling.
"""
import pytest
from fastapi.testclient import TestClient
from src.api import app

client = TestClient(app)


def test_health_endpoint():
    """Verify GET /health returns 200 OK with model status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["model_loaded"] is True
    assert data["model_version"] == "1.0.0"


def test_predict_valid_payload():
    """Verify POST /predict returns 200 OK and accurate prediction array for valid payload."""
    payload = {
        "square_feet": 2200.0,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location_score": 8.0,
        "year_built": 2012,
        "garage_spaces": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "prediction" in data
    assert isinstance(data["prediction"], list)
    assert len(data["prediction"]) == 1
    assert data["prediction"][0] > 0
    assert data["status"] == "success"
    assert data["units"] == "USD"


def test_predict_missing_field_returns_422():
    """Verify POST /predict returns 422 Unprocessable Entity when required field is missing."""
    payload = {
        # 'square_feet' is omitted
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location_score": 7.5,
        "year_built": 2000,
        "garage_spaces": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422
    data = response.json()
    assert "detail" in data
    assert any("square_feet" in err.get("loc", []) for err in data["detail"])


def test_predict_wrong_data_type_returns_422():
    """Verify POST /predict returns 422 when field contains wrong data type."""
    payload = {
        "square_feet": "two_thousand_sqft",  # Invalid type: string instead of float
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location_score": 7.5,
        "year_built": 2000,
        "garage_spaces": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_predict_invalid_business_value_constraint():
    """Verify POST /predict returns 422 when input violates domain business constraints."""
    # Test 1: Non-positive square feet
    payload_negative_area = {
        "square_feet": -500.0,  # Constraint: gt=0
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location_score": 7.5,
        "year_built": 2000,
        "garage_spaces": 1
    }
    resp1 = client.post("/predict", json=payload_negative_area)
    assert resp1.status_code == 422

    # Test 2: Location score out of range (1-10)
    payload_bad_score = {
        "square_feet": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2.0,
        "location_score": 15.0,  # Constraint: le=10.0
        "year_built": 2000,
        "garage_spaces": 1
    }
    resp2 = client.post("/predict", json=payload_bad_score)
    assert resp2.status_code == 422


def test_predict_batch_endpoint():
    """Verify POST /predict_batch handles multiple records simultaneously."""
    batch_payload = [
        {"square_feet": 1200.0, "bedrooms": 2, "bathrooms": 1.0, "location_score": 6.0, "year_built": 1990, "garage_spaces": 1},
        {"square_feet": 2800.0, "bedrooms": 4, "bathrooms": 3.0, "location_score": 8.5, "year_built": 2018, "garage_spaces": 2}
    ]
    response = client.post("/predict_batch", json=batch_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 2
    assert len(data["predictions"]) == 2
    assert data["predictions"][1] > data["predictions"][0]  # Larger house should be priced higher

