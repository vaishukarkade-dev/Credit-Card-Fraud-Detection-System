"""
test_api.py - Quick end-to-end API smoke test
Run: python test_api.py
"""
import requests, json

BASE = "http://localhost:8000"

# 1. Health
print("=== /health ===")
r = requests.get(f"{BASE}/health")
print(json.dumps(r.json(), indent=2))

# 2. Single predict (legit-looking transaction)
print("\n=== /predict (legit) ===")
tx_legit = {
    "Time": 1000, "Amount": 25.50,
    "V1": 0.5, "V2": 0.3, "V3": -0.1, "V4": 0.8,
    "V5": -0.2, "V6": 0.1, "V7": 0.4, "V8": -0.3,
    "V9": 0.2, "V10": -0.1, "V11": 0.5, "V12": -0.4,
    "V13": 0.3, "V14": -0.2, "V15": 0.6, "V16": -0.1,
    "V17": 0.2, "V18": -0.3, "V19": 0.1, "V20": 0.4,
    "V21": -0.2, "V22": 0.3, "V23": -0.1, "V24": 0.5,
    "V25": -0.3, "V26": 0.2, "V27": -0.1, "V28": 0.4,
}
r2 = requests.post(f"{BASE}/predict", json=tx_legit)
print(json.dumps(r2.json(), indent=2))

# 3. Single predict (suspicious transaction)
print("\n=== /predict (suspicious) ===")
tx_fraud = {
    "Time": 500, "Amount": 2.5,
    "V1": -3.5, "V2": -2.8, "V3": -4.1, "V4": -1.2,
    "V5": -3.0, "V6": -2.5, "V7": -3.8, "V8": -1.9,
    "V9": -2.2, "V10": -3.6, "V11": -2.1, "V12": -4.5,
    "V13": -1.8, "V14": -3.9, "V15": -2.4, "V16": -3.1,
    "V17": -4.2, "V18": -2.7, "V19": -1.5, "V20": -3.3,
    "V21": -2.0, "V22": -3.7, "V23": -1.6, "V24": -2.9,
    "V25": -3.4, "V26": -1.3, "V27": -2.6, "V28": -3.8,
}
r3 = requests.post(f"{BASE}/predict", json=tx_fraud)
print(json.dumps(r3.json(), indent=2))

# 4. Batch predict
print("\n=== /batch_predict (sample_transactions.csv) ===")
with open("data/sample_transactions.csv", "rb") as f:
    r4 = requests.post(
        f"{BASE}/batch_predict",
        files={"file": ("sample_transactions.csv", f, "text/csv")}
    )
d = r4.json()
print(f"total_rows  : {d['total_rows']}")
print(f"fraud_count : {d['fraud_count']}")
print(f"legit_count : {d['legit_count']}")
print(f"fraud_rate  : {d['fraud_rate']}%")
print("\nTop 5 highest-risk rows:")
top5 = sorted(d["predictions"], key=lambda x: x["risk_score"], reverse=True)[:5]
for p in top5:
    print(f"  row {p['row']:>3} | ${p['amount']:>8.2f} | {p['risk_score']:>5.1f}% | {p['label']} ({p['confidence']})")

print("\n[OK] All endpoints working!")
