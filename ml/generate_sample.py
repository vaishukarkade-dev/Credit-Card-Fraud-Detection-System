"""
generate_sample.py
------------------
Generates a small sample CSV (100 rows) for testing the upload page.
Includes a mix of legit and fraudulent transactions.

Usage:
    python ml\generate_sample.py
    -> writes data\sample_transactions.csv
"""
import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(42)
OUTPUT = os.path.join(os.path.dirname(__file__), "..", "data", "sample_transactions.csv")

rows = []
# 85 legit + 15 fraud = 100 rows
for i in range(100):
    is_fraud = i >= 85
    v = rng.normal(loc=-1.5 if is_fraud else 0.0,
                   scale=2.5 if is_fraud else 1.0,
                   size=28)
    amount = round(rng.uniform(1.0, 50.0) if is_fraud else rng.lognormal(4.0, 1.5), 2)
    amount = max(0.5, min(amount, 10000.0))
    row = {"Time": int(rng.uniform(0, 172800))}
    for j, val in enumerate(v, 1):
        row[f"V{j}"] = round(float(val), 6)
    row["Amount"] = amount
    row["Class"] = int(is_fraud)
    rows.append(row)

df = pd.DataFrame(rows).sample(frac=1, random_state=42).reset_index(drop=True)
os.makedirs(os.path.dirname(OUTPUT), exist_ok=True)
df.to_csv(OUTPUT, index=False)
print(f"[DONE] Sample CSV -> {OUTPUT}")
print(f"       {len(df)} rows | {df['Class'].sum()} fraud | {(~df['Class'].astype(bool)).sum()} legit")
