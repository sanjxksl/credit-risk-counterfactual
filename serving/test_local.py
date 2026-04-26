"""
Test the API locally before deploying.

Run the server first:
  MODEL_DIR=../models uvicorn app.main:app --reload --port 8080

Then run this script in a separate terminal:
  python test_local.py
"""

import json
import requests

BASE = "http://localhost:8080"

# A sample high-risk application
high_risk = {
    "year": 2019,
    "loan_amount": 450000,
    "term": 360,
    "property_value": 460000,
    "income": 4200,
    "credit_score": 540,
    "ltv": 97.8,
    "dtir1": 52.0,
    "loan_limit": "cf",
    "gender": "Male",
    "approv_in_adv": "nopre",
    "loan_type": "type1",
    "loan_purpose": "p1",
    "credit_worthiness": "l1",
    "open_credit": "nopc",
    "business_or_commercial": "nob/c",
    "neg_ammortization": "not_neg",
    "interest_only": "not_int",
    "lump_sum_payment": "not_lpsm",
    "occupancy_type": "pr",
    "total_units": "1U",
    "age": "35-44",
    "submission_of_application": "to_inst",
    "region": "south",
}

low_risk = {**high_risk, "loan_amount": 200000, "ltv": 60.0, "dtir1": 28.0, "credit_score": 780}


def test_health():
    r = requests.get(f"{BASE}/ping")
    assert r.status_code == 200, r.text
    print("✓ Health check OK")


def test_predict(payload, label):
    r = requests.post(f"{BASE}/predict", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    print(f"✓ /predict ({label})")
    print(f"  probability={data['default_probability']}  risk={data['risk_level']}")
    print(f"  message: {data['message']}")
    return data


def test_explain(payload):
    r = requests.post(f"{BASE}/predict/explain", json=payload)
    assert r.status_code == 200, r.text
    data = r.json()
    print(f"✓ /predict/explain")
    print(f"  probability={data['default_probability']}  counterfactuals_available={data['counterfactuals_available']}")
    if data["counterfactuals"]:
        for i, cf in enumerate(data["counterfactuals"], 1):
            print(f"  Scenario {i}: p={cf['predicted_probability_after']}  — {cf['description']}")
    return data


def test_invalid():
    bad = {**high_risk, "credit_score": 200}   # out of range
    r = requests.post(f"{BASE}/predict", json=bad)
    assert r.status_code == 422, f"Expected 422, got {r.status_code}"
    print("✓ Invalid input correctly rejected (422)")


if __name__ == "__main__":
    print("=== Local API Tests ===\n")
    test_health()
    test_predict(high_risk, "high risk")
    test_predict(low_risk, "low risk")
    test_explain(high_risk)
    test_invalid()
    print("\nAll tests passed.")
