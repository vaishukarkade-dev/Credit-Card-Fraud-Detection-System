"""
create_dummy_csvs.py  (v2)
--------------------------
Creates 5 realistic dummy CSV files by sampling from the actual
training dataset, ensuring the model classifies them correctly.

Run:
    python ml/create_dummy_csvs.py
"""

import os
import numpy as np
import pandas as pd

rng = np.random.default_rng(99)
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
os.makedirs(OUT, exist_ok=True)

# Load real training data to sample from
df_all = pd.read_csv(DATA)
df_legit = df_all[df_all["Class"] == 0].drop(columns=["Class"])
df_fraud = df_all[df_all["Class"] == 1].drop(columns=["Class"])

print(f"[INFO] Loaded {len(df_legit)} legit and {len(df_fraud)} fraud rows to sample from")

COLS = list(df_legit.columns)  # Time, V1-V28, Amount


def sample_legit(n):
    return df_legit.sample(n=n, random_state=rng.integers(0, 10000), replace=True).reset_index(drop=True)


def sample_fraud(n):
    return df_fraud.sample(n=n, random_state=rng.integers(0, 10000), replace=True).reset_index(drop=True)


def add_noise(df, scale=0.02):
    """Add tiny noise to avoid exact duplicates of training rows."""
    numeric_cols = [c for c in df.columns if c not in ["Time"]]
    noise = pd.DataFrame(
        rng.normal(0, scale, size=(len(df), len(numeric_cols))),
        columns=numeric_cols
    )
    df[numeric_cols] = df[numeric_cols] + noise
    df["Amount"] = df["Amount"].clip(lower=0.5).round(2)
    df["Time"] = df["Time"].astype(int)
    return df


def save(df, name, desc):
    path = os.path.join(OUT, name)
    df.to_csv(path, index=False)
    print(f"[OK] {name:45s} {desc}")


# ============================================================================
# CSV 1: ALL LEGIT - Weekend Shopping (50 rows)
# ============================================================================
df1 = sample_legit(50)
df1 = add_noise(df1)
save(df1, "01_all_legit_weekend_shopping.csv",
     "50 rows | 0 fraud  | Regular weekend shopping")


# ============================================================================
# CSV 2: ALL FRAUD - Stolen Card at 3AM (30 rows)
# ============================================================================
df2 = sample_fraud(30)
df2 = add_noise(df2, scale=0.01)
# Set time to 3 AM range (10800-14400 seconds)
df2["Time"] = rng.integers(10800, 14400, size=30)
save(df2, "02_all_fraud_stolen_card.csv",
     "30 rows | 30 fraud | Stolen card used at 3 AM")


# ============================================================================
# CSV 3: MIXED - Corporate Card (80 rows, ~15% fraud)
# ============================================================================
n_fraud_3 = 12
legit_part = sample_legit(80 - n_fraud_3)
fraud_part = sample_fraud(n_fraud_3)
df3 = pd.concat([legit_part, fraud_part], ignore_index=True)
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)
df3 = add_noise(df3)
save(df3, "03_mixed_corporate_card.csv",
     f"80 rows | {n_fraud_3} fraud | Corporate card with hidden fraud")


# ============================================================================
# CSV 4: EDGE CASES - Online Merchant (40 rows, ~20% fraud)
# ============================================================================
n_fraud_4 = 8
legit_part = sample_legit(40 - n_fraud_4)
fraud_part = sample_fraud(n_fraud_4)
# Make amounts small (micro-transactions)
legit_part["Amount"] = rng.uniform(0.5, 15.0, size=len(legit_part)).round(2)
fraud_part["Amount"] = rng.uniform(0.5, 5.0, size=len(fraud_part)).round(2)
df4 = pd.concat([legit_part, fraud_part], ignore_index=True)
df4 = df4.sample(frac=1, random_state=77).reset_index(drop=True)
df4 = add_noise(df4, scale=0.03)
save(df4, "04_edge_cases_online_merchant.csv",
     f"40 rows | {n_fraud_4} fraud  | Micro-transactions & edge cases")


# ============================================================================
# CSV 5: LARGE BATCH - Full Retailer Day (500 rows, ~1.7% fraud)
# ============================================================================
n_fraud_5 = 8
legit_part = sample_legit(500 - n_fraud_5)
fraud_part = sample_fraud(n_fraud_5)
df5 = pd.concat([legit_part, fraud_part], ignore_index=True)
df5 = df5.sample(frac=1, random_state=123).reset_index(drop=True)
# Sort by time to simulate a real day
df5 = df5.sort_values("Time").reset_index(drop=True)
df5 = add_noise(df5)
save(df5, "05_large_batch_retailer_day.csv",
     f"500 rows | {n_fraud_5} fraud  | Full retailer day (1.6% fraud rate)")


print(f"\nAll 5 sample CSVs saved to: {OUT}")
print("Upload any of these at http://localhost:5500/upload.html")
