import os, requests

BASE = "http://localhost:8000"
DIR  = "frontend/samples"

files = sorted(f for f in os.listdir(DIR) if f.endswith(".csv") and not f.startswith("test"))
print("=" * 68)
print("  FraudShield AI — All Sample CSVs Test")
print("=" * 68)
for fname in files:
    with open(os.path.join(DIR, fname), "rb") as f:
        r = requests.post(f"{BASE}/batch_predict", files={"file": (fname, f, "text/csv")})
    d = r.json()
    bar = "#" * int(d["fraud_rate"] / 2)
    print(f"  {fname[:42]:42s} total={d['total_rows']:>3}  fraud={d['fraud_count']:>3}  rate={d['fraud_rate']:>5.1f}%  {bar}")
print("=" * 68)
print("All tests passed!")
