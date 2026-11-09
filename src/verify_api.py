"""
Automated Verification Script for Session 8 and Session 9.
Simulates curl/HTTP requests for:
1. GET /health
2. POST /predict (Valid payload)
3. POST /predict (Missing required field -> 422)
4. POST /predict (Wrong data type -> 422)
5. POST /predict (Invalid business value -> 422)
6. POST /predict_batch (Batch scoring)
Saves evidence table to reports/api_test_evidence.json.
"""
import json
import os
from starlette.testclient import TestClient
from src.api import app

def run_api_verification():
    client = TestClient(app)
    evidence = []

    print("=" * 60)
    print("      FASTAPI ENDPOINT & ERROR VALIDATION TEST RUN      ")
    print("=" * 60)

    # 1. Health check
    print("\n[Test 1] GET /health")
    resp_health = client.get("/health")
    print(f"Status Code: {resp_health.status_code}")
    print(f"Response: {resp_health.json()}")
    evidence.append({
        "test": "Health check",
        "method": "GET",
        "endpoint": "/health",
        "payload": None,
        "expected_status": 200,
        "observed_status": resp_health.status_code,
        "expected_behaviour": "Service reports healthy with model loaded",
        "observed_result": resp_health.json(),
        "status": "PASS" if resp_health.status_code == 200 else "FAIL"
    })

    # 2. Valid Prediction
    print("\n[Test 2] POST /predict (Valid Request)")
    valid_payload = {
        "square_feet": 2200.0,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location_score": 8.0,
        "year_built": 2012,
        "garage_spaces": 2
    }
    resp_valid = client.post("/predict", json=valid_payload)
    print(f"Status Code: {resp_valid.status_code}")
    print(f"Response: {resp_valid.json()}")
    evidence.append({
        "test": "Valid input",
        "method": "POST",
        "endpoint": "/predict",
        "payload": valid_payload,
        "expected_status": 200,
        "observed_status": resp_valid.status_code,
        "expected_behaviour": "Prediction returned in expected schema",
        "observed_result": resp_valid.json(),
        "status": "PASS" if resp_valid.status_code == 200 else "FAIL"
    })

    # 3. Missing Required Field
    print("\n[Test 3] POST /predict (Missing Field 'square_feet')")
    missing_payload = {
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location_score": 8.0,
        "year_built": 2012,
        "garage_spaces": 2
    }
    resp_missing = client.post("/predict", json=missing_payload)
    print(f"Status Code: {resp_missing.status_code}")
    print(f"Response: {resp_missing.json()}")
    evidence.append({
        "test": "Missing field",
        "method": "POST",
        "endpoint": "/predict",
        "payload": missing_payload,
        "expected_status": 422,
        "observed_status": resp_missing.status_code,
        "expected_behaviour": "Validation error (HTTP 422)",
        "observed_result": resp_missing.json(),
        "status": "PASS" if resp_missing.status_code == 422 else "FAIL"
    })

    # 4. Wrong Data Type
    print("\n[Test 4] POST /predict (Wrong Data Type: string for bedrooms)")
    wrong_type_payload = {
        "square_feet": 2200.0,
        "bedrooms": "three_bedrooms",
        "bathrooms": 2.5,
        "location_score": 8.0,
        "year_built": 2012,
        "garage_spaces": 2
    }
    resp_wrong = client.post("/predict", json=wrong_type_payload)
    print(f"Status Code: {resp_wrong.status_code}")
    print(f"Response: {resp_wrong.json()}")
    evidence.append({
        "test": "Wrong type",
        "method": "POST",
        "endpoint": "/predict",
        "payload": wrong_type_payload,
        "expected_status": 422,
        "observed_status": resp_wrong.status_code,
        "expected_behaviour": "Validation error (HTTP 422)",
        "observed_result": resp_wrong.json(),
        "status": "PASS" if resp_wrong.status_code == 422 else "FAIL"
    })

    # 5. Invalid Business Value
    print("\n[Test 5] POST /predict (Invalid Business Value: negative square_feet)")
    invalid_val_payload = {
        "square_feet": -150.0,
        "bedrooms": 3,
        "bathrooms": 2.5,
        "location_score": 8.0,
        "year_built": 2012,
        "garage_spaces": 2
    }
    resp_invalid = client.post("/predict", json=invalid_val_payload)
    print(f"Status Code: {resp_invalid.status_code}")
    print(f"Response: {resp_invalid.json()}")
    evidence.append({
        "test": "Invalid business value",
        "method": "POST",
        "endpoint": "/predict",
        "payload": invalid_val_payload,
        "expected_status": 422,
        "observed_status": resp_invalid.status_code,
        "expected_behaviour": "Controlled validation error if constraint exists",
        "observed_result": resp_invalid.json(),
        "status": "PASS" if resp_invalid.status_code == 422 else "FAIL"
    })

    # 6. Batch Inference
    print("\n[Test 6] POST /predict_batch (Batch scoring)")
    batch_payload = [
        {"square_feet": 1200.0, "bedrooms": 2, "bathrooms": 1.0, "location_score": 6.0, "year_built": 1990, "garage_spaces": 1},
        {"square_feet": 3000.0, "bedrooms": 4, "bathrooms": 3.0, "location_score": 9.0, "year_built": 2021, "garage_spaces": 3}
    ]
    resp_batch = client.post("/predict_batch", json=batch_payload)
    print(f"Status Code: {resp_batch.status_code}")
    print(f"Response: {resp_batch.json()}")
    evidence.append({
        "test": "Batch prediction",
        "method": "POST",
        "endpoint": "/predict_batch",
        "payload": batch_payload,
        "expected_status": 200,
        "observed_status": resp_batch.status_code,
        "expected_behaviour": "Scored predictions returned for all records",
        "observed_result": resp_batch.json(),
        "status": "PASS" if resp_batch.status_code == 200 else "FAIL"
    })

    os.makedirs("reports", exist_ok=True)
    with open("reports/api_test_evidence.json", "w") as f:
        json.dump(evidence, f, indent=4)
    print("\n" + "=" * 60)
    print("Verification evidence written to reports/api_test_evidence.json")
    print("=" * 60)

if __name__ == "__main__":
    run_api_verification()

