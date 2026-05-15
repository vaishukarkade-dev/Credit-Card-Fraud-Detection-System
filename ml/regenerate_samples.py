"""
regenerate_samples.py
---------------------
Regenerates ALL 10 sample CSV files with realistic noise injection.

Noise strategy:
  - Gaussian noise on V1-V28 features (scale=0.15-0.30)
  - Feature-specific jitter to break memorization
  - Amount perturbation (+/-5-15%)
  - Time drift (+/-300-600 seconds)
  - Random sign flips on low-weight features (V22-V28)

This ensures the model sees genuinely novel inputs, not near-copies
of training rows, giving realistic fraud/legit predictions.

Run:
    python ml/regenerate_samples.py
"""

import os
import sys
import numpy as np
import pandas as pd

rng = np.random.default_rng(2026)

# ── Paths ──────────────────────────────────────────────────────────────
BASE     = os.path.dirname(os.path.dirname(__file__))
DATA     = os.path.join(BASE, "data", "creditcard.csv")
OUT_DATA = os.path.join(BASE, "data", "samples")
OUT_FE   = os.path.join(BASE, "frontend", "samples")
os.makedirs(OUT_DATA, exist_ok=True)
os.makedirs(OUT_FE,   exist_ok=True)

# ── Load real dataset ──────────────────────────────────────────────────
if not os.path.exists(DATA):
    print(f"[ERROR] Dataset not found: {DATA}")
    print("        Download from: https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud")
    sys.exit(1)

df_all   = pd.read_csv(DATA)
df_legit = df_all[df_all["Class"] == 0].drop(columns=["Class"])
df_fraud = df_all[df_all["Class"] == 1].drop(columns=["Class"])
print(f"[INFO] Pool: {len(df_legit):,} legit | {len(df_fraud):,} fraud rows")


# ── Noise injection ───────────────────────────────────────────────────
def pick(src, n):
    """Sample n rows with replacement from source DataFrame."""
    return src.sample(n=n, replace=True,
                      random_state=rng.integers(0, 99999)).reset_index(drop=True)


def inject_noise(df, v_scale=0.15, amount_pct=0.10, time_drift=300,
                 flip_prob=0.08):
    """
    Inject realistic noise into transaction data.

    Args:
        df:          DataFrame with Time, V1-V28, Amount
        v_scale:     Gaussian noise std for V features (higher = more different)
        amount_pct:  Max fractional change in Amount (e.g. 0.10 = +/-10%)
        time_drift:  Max seconds to shift Time by
        flip_prob:   Probability of sign-flipping low-weight V features
    """
    df = df.copy()
    v_cols = [f"V{i}" for i in range(1, 29)]

    # 1. Gaussian noise on V features -- scaled per-feature by their std
    for col in v_cols:
        col_std = df[col].std()
        if col_std == 0:
            col_std = 1.0
        noise = rng.normal(0, v_scale * col_std, size=len(df))
        df[col] = df[col] + noise

    # 2. Random sign flips on low-importance features (V22–V28)
    for col in [f"V{i}" for i in range(22, 29)]:
        mask = rng.random(len(df)) < flip_prob
        df.loc[mask, col] = -df.loc[mask, col]

    # 3. Amount perturbation -- multiplicative jitter
    amount_noise = 1.0 + rng.uniform(-amount_pct, amount_pct, size=len(df))
    df["Amount"] = (df["Amount"] * amount_noise).clip(lower=0.01).round(2)

    # 4. Time drift -- additive jitter
    time_jitter = rng.integers(-time_drift, time_drift + 1, size=len(df))
    df["Time"] = (df["Time"] + time_jitter).clip(lower=0).astype(int)

    # 5. Feature interaction noise -- small correlated perturbation on V1-V4
    #    (these are the highest-variance PCA components)
    cross_noise = rng.normal(0, 0.05, size=len(df))
    for col in ["V1", "V2", "V3", "V4"]:
        df[col] = df[col] + cross_noise * rng.normal(0.5, 0.3)

    return df


def save(df, name, desc):
    """Save CSV to both data/samples/ and frontend/samples/."""
    for d in [OUT_DATA, OUT_FE]:
        df.to_csv(os.path.join(d, name), index=False)
    print(f"  [OK] {name:48s} {desc}")


print("\n[GENERATING] 10 sample CSVs with noise injection...\n")

# ═══════════════════════════════════════════════════════════════════════
# CSV 1: ALL LEGIT — Weekend Shopping (50 rows)
# ═══════════════════════════════════════════════════════════════════════
df1 = pick(df_legit, 50)
df1["Time"] = rng.integers(43200, 86400, size=50)  # afternoon-evening
df1 = inject_noise(df1, v_scale=0.20, amount_pct=0.12)
save(df1, "01_all_legit_weekend_shopping.csv",
     "50 rows  |  0 fraud | Weekend shopping")

# ═══════════════════════════════════════════════════════════════════════
# CSV 2: ALL FRAUD — Stolen Card at 3 AM (30 rows)
# ═══════════════════════════════════════════════════════════════════════
df2 = pick(df_fraud, 30)
df2["Time"] = rng.integers(10800, 14400, size=30)   # 3–4 AM
df2 = inject_noise(df2, v_scale=0.15, amount_pct=0.08)
save(df2, "02_all_fraud_stolen_card.csv",
     "30 rows  | 30 fraud | Stolen card at 3 AM")

# ═══════════════════════════════════════════════════════════════════════
# CSV 3: MIXED — Corporate Card (80 rows, ~15% fraud)
# ═══════════════════════════════════════════════════════════════════════
n3f = 12
df3 = pd.concat([pick(df_legit, 80 - n3f), pick(df_fraud, n3f)])
df3 = df3.sample(frac=1, random_state=42).reset_index(drop=True)
df3 = inject_noise(df3, v_scale=0.18, amount_pct=0.15)
save(df3, "03_mixed_corporate_card.csv",
     f"80 rows  | {n3f} fraud | Corporate card (hidden fraud)")

# ═══════════════════════════════════════════════════════════════════════
# CSV 4: EDGE CASES — Online Merchant, micro-transactions (40 rows)
# ═══════════════════════════════════════════════════════════════════════
n4f = 8
legit4 = pick(df_legit, 40 - n4f)
legit4["Amount"] = rng.uniform(0.5, 15.0, size=len(legit4)).round(2)
fraud4 = pick(df_fraud, n4f)
fraud4["Amount"] = rng.uniform(0.5, 5.0, size=n4f).round(2)
df4 = pd.concat([legit4, fraud4]).sample(frac=1, random_state=77).reset_index(drop=True)
df4 = inject_noise(df4, v_scale=0.25, amount_pct=0.20, flip_prob=0.12)
save(df4, "04_edge_cases_online_merchant.csv",
     f"40 rows  |  {n4f} fraud | Micro-transactions & edge cases")

# ═══════════════════════════════════════════════════════════════════════
# CSV 5: LARGE BATCH — Full Retailer Day (500 rows, ~1.6% fraud)
# ═══════════════════════════════════════════════════════════════════════
n5f = 8
df5 = pd.concat([pick(df_legit, 500 - n5f), pick(df_fraud, n5f)])
df5 = df5.sample(frac=1, random_state=123).reset_index(drop=True)
df5 = df5.sort_values("Time").reset_index(drop=True)
df5 = inject_noise(df5, v_scale=0.20, amount_pct=0.10, time_drift=600)
save(df5, "05_large_batch_retailer_day.csv",
     f"500 rows |  {n5f} fraud | Full retailer day (time-sorted)")

# ═══════════════════════════════════════════════════════════════════════
# CSV 6: HIGH VALUE — Luxury & Travel (100 rows, 20% fraud)
# ═══════════════════════════════════════════════════════════════════════
n6f = 20
legit6 = pick(df_legit, 80)
legit6["Amount"] = rng.uniform(500, 8000, size=80).round(2)
fraud6 = pick(df_fraud, n6f)
fraud6["Amount"] = rng.uniform(2000, 15000, size=n6f).round(2)
df6 = pd.concat([legit6, fraud6]).sample(frac=1, random_state=42).reset_index(drop=True)
df6 = inject_noise(df6, v_scale=0.22, amount_pct=0.12)
save(df6, "06_high_value_transactions.csv",
     f"100 rows | {n6f} fraud | High-value luxury & travel")

# ═══════════════════════════════════════════════════════════════════════
# CSV 7: RAPID-FIRE — Card Testing Attack (200 rows, 30% fraud)
# ═══════════════════════════════════════════════════════════════════════
n7f = 60
legit7 = pick(df_legit, 140)
legit7["Time"] = np.sort(rng.integers(0, 600, size=140))  # 10-min window
fraud7 = pick(df_fraud, n7f)
fraud7["Time"] = rng.integers(0, 600, size=n7f)
fraud7["Amount"] = rng.uniform(0.5, 3.0, size=n7f).round(2)  # micro-amounts
df7 = pd.concat([legit7, fraud7]).sort_values("Time").reset_index(drop=True)
df7 = inject_noise(df7, v_scale=0.15, amount_pct=0.08, time_drift=30)
save(df7, "07_card_testing_attack.csv",
     f"200 rows | {n7f} fraud | Rapid card-testing attack (10 min)")

# ═══════════════════════════════════════════════════════════════════════
# CSV 8: E-COMMERCE — Returns Abuse (150 rows, 10% fraud)
# ═══════════════════════════════════════════════════════════════════════
n8f = 15
legit8 = pick(df_legit, 135)
legit8["Amount"] = rng.uniform(20, 300, size=135).round(2)
fraud8 = pick(df_fraud, n8f)
fraud8["Amount"] = rng.uniform(150, 500, size=n8f).round(2)
df8 = pd.concat([legit8, fraud8]).sample(frac=1, random_state=7).reset_index(drop=True)
df8 = inject_noise(df8, v_scale=0.20, amount_pct=0.15, flip_prob=0.10)
save(df8, "08_ecommerce_returns_abuse.csv",
     f"150 rows | {n8f} fraud | E-commerce returns abuse")

# ═══════════════════════════════════════════════════════════════════════
# CSV 9: INTERNATIONAL — Travel Card, night transactions (200 rows)
# ═══════════════════════════════════════════════════════════════════════
n9f = 10
legit9 = pick(df_legit, 190)
legit9["Time"] = rng.integers(64800, 86400, size=190)  # late night
fraud9 = pick(df_fraud, n9f)
fraud9["Time"] = rng.integers(64800, 86400, size=n9f)
fraud9["Amount"] = rng.lognormal(5.5, 0.8, size=n9f).clip(100, 10000).round(2)
df9 = pd.concat([legit9, fraud9]).sort_values("Time").reset_index(drop=True)
df9 = inject_noise(df9, v_scale=0.18, amount_pct=0.10, time_drift=600)
save(df9, "09_international_travel_card.csv",
     f"200 rows | {n9f} fraud | International travel (night txns)")

# ═══════════════════════════════════════════════════════════════════════
# CSV 10: ATM — 50/50 Split Stress Test (100 rows)
# ═══════════════════════════════════════════════════════════════════════
legit10 = pick(df_legit, 50)
legit10["Amount"] = rng.choice([20, 40, 50, 60, 80, 100, 200], size=50).astype(float)
fraud10 = pick(df_fraud, 50)
fraud10["Amount"] = rng.choice([20, 40, 50, 60, 80, 100, 200], size=50).astype(float)
df10 = pd.concat([legit10, fraud10]).sample(frac=1, random_state=99).reset_index(drop=True)
df10 = inject_noise(df10, v_scale=0.25, amount_pct=0.05, flip_prob=0.15)
save(df10, "10_atm_withdrawals_50_50.csv",
     "100 rows | 50 fraud | ATM 50/50 stress test")


# ═══════════════════════════════════════════════════════════════════════
# Known fraud test file
# ═══════════════════════════════════════════════════════════════════════
known = pd.concat([pick(df_fraud, 7), pick(df_legit, 3)])
known = known.sample(frac=1, random_state=555).reset_index(drop=True)
known = inject_noise(known, v_scale=0.12, amount_pct=0.05)
for d in [OUT_DATA]:
    known.to_csv(os.path.join(d, "test_known_fraud.csv"), index=False)
print(f"  [OK] {'test_known_fraud.csv':48s} 10 rows  |  7 fraud | Known fraud test")


print(f"\n[DONE] All sample CSVs regenerated with noise injection.")
print(f"  -> {OUT_DATA}")
print(f"  -> {OUT_FE}")
print(f"\n  Noise config: V-scale=0.12-0.25, Amount=+/-5-20%, Time=+/-30-600s, Sign-flip=8-15%")
