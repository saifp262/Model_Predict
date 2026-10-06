# ==============================================================================
# MLOPS UNIT END PROJECT 1: AUTOMATED API TEST SUITE
# ==============================================================================

import sys
import json
from fastapi.testclient import TestClient
from app import app

# Ensure UTF-8 output encoding on Windows consoles
if sys.platform.startswith('win'):
    sys.stdout.reconfigure(encoding='utf-8')

client = TestClient(app)

def test_endpoints():
    print("=" * 70)
    print("🧪 RUNNING AUTOMATED UNIT TESTS FOR PATIENT READMISSION API")
    print("=" * 70)
    
    # 1. Healthcheck Test
    health_resp = client.get("/health")
    assert health_resp.status_code == 200, f"Expected 200, got {health_resp.status_code}"
    print("1. Health Endpoint (/health):")
    print(f"   Status Code: {health_resp.status_code}")
    print(f"   Response: {health_resp.json()}\\n")
    
    # 2. Patient A: High-Risk Profile (Female, Class 1, Age 35, Title Mrs, Fare 80.0)
    patient_high_risk = {
        "Pclass": 1,
        "Sex": "female",
        "Age": 35.0,
        "Fare": 80.0,
        "Embarked": "C",
        "Title": "Mrs",
        "FamilySize": 2,
        "IsAlone": 0
    }
    
    resp_a = client.post("/predict", json=patient_high_risk)
    assert resp_a.status_code == 200, f"Expected 200, got {resp_a.status_code}"
    data_a = resp_a.json()
    print("2. Scoring Patient A (High Risk / Vulnerable Demographics):")
    print(f"   Status Code: {resp_a.status_code}")
    print(f"   Response:\\n{json.dumps(data_a, indent=2)}\\n")
    
    # 3. Patient B: Low-Risk Profile (Male, Class 3, Age 22, Title Mr, Fare 7.25)
    patient_low_risk = {
        "Pclass": 3,
        "Sex": "male",
        "Age": 22.0,
        "Fare": 7.25,
        "Embarked": "S",
        "Title": "Mr",
        "FamilySize": 1,
        "IsAlone": 1
    }
    
    resp_b = client.post("/predict", json=patient_low_risk)
    assert resp_b.status_code == 200, f"Expected 200, got {resp_b.status_code}"
    data_b = resp_b.json()
    print("3. Scoring Patient B (Low Risk / Standard Demographics):")
    print(f"   Status Code: {resp_b.status_code}")
    print(f"   Response:\\n{json.dumps(data_b, indent=2)}")
    
    print("=" * 70)
    print("✅ ALL API IN-MEMORY UNIT TESTS PASSED SUCCESSFULLY!")
    print("=" * 70)

if __name__ == "__main__":
    test_endpoints()
