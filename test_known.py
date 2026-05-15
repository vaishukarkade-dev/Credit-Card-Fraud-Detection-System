import requests
r = requests.post(
    "http://localhost:8000/batch_predict",
    files={"file": ("test.csv", open("data/samples/test_known_fraud.csv","rb"), "text/csv")}
)
d = r.json()
print(f"Fraud detected: {d['fraud_count']}/{d['total_rows']}")
for p in d["predictions"]:
    print(f"  row {p['row']}: {p['risk_score']}% {p['label']} ({p['confidence']})")
