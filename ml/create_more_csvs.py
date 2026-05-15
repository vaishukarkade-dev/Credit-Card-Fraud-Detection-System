"""
create_more_csvs.py — generates 5 additional scenario CSVs
"""
import os, numpy as np, pandas as pd

rng  = np.random.default_rng(777)
DATA = os.path.join(os.path.dirname(__file__), "..", "data", "creditcard.csv")
FE   = os.path.join(os.path.dirname(__file__), "..", "frontend", "samples")
OUT  = os.path.join(os.path.dirname(__file__), "..", "data", "samples")
os.makedirs(FE,  exist_ok=True)
os.makedirs(OUT, exist_ok=True)

df_all   = pd.read_csv(DATA)
df_legit = df_all[df_all["Class"] == 0].drop(columns=["Class"])
df_fraud = df_all[df_all["Class"] == 1].drop(columns=["Class"])
print(f"[INFO] Pool: {len(df_legit):,} legit | {len(df_fraud)} fraud")

def pick(src, n, seed=None):
    return src.sample(n=n, replace=True, random_state=rng.integers(0,99999)).reset_index(drop=True)

def noise(df, s=0.015):
    num = [c for c in df.columns if c != "Time"]
    df = df.copy()
    df[num] = df[num] + rng.normal(0, s, size=(len(df), len(num)))
    df["Amount"] = df["Amount"].clip(lower=0.5).round(2)
    df["Time"]   = df["Time"].astype(int)
    return df

def save(df, name, desc):
    for d in [OUT, FE]:
        df.to_csv(os.path.join(d, name), index=False)
    print(f"[OK] {name:46s} {desc}")


# ── CSV 6: High-value transactions — travel & luxury (20% fraud) ───────────
n6f = 20
legit6 = pick(df_legit, 80)
legit6["Amount"] = rng.uniform(500, 8000, size=80).round(2)  # high value
fraud6 = pick(df_fraud, n6f)
fraud6["Amount"] = rng.uniform(2000, 15000, size=n6f).round(2)
df6 = pd.concat([legit6, fraud6]).sample(frac=1, random_state=42).reset_index(drop=True)
df6 = noise(df6)
save(df6, "06_high_value_transactions.csv",
     "100 rows | 20 fraud | High-value luxury & travel spend")


# ── CSV 7: Rapid-fire — 200 txns in 10 min (card testing attack) ──────────
n7f = 60
legit7 = pick(df_legit, 140)
legit7["Time"] = np.sort(rng.integers(0, 600, size=140))   # 10 min window
fraud7 = pick(df_fraud, n7f)
fraud7["Time"] = rng.integers(0, 600, size=n7f)
fraud7["Amount"] = rng.uniform(0.5, 3.0, size=n7f).round(2)  # micro amounts
df7 = pd.concat([legit7, fraud7]).sort_values("Time").reset_index(drop=True)
df7 = noise(df7, s=0.01)
save(df7, "07_card_testing_attack.csv",
     "200 rows | 60 fraud | Rapid card-testing attack in 10 min")


# ── CSV 8: E-commerce returns abuse (10% fraud, mid amounts) ──────────────
n8f = 15
legit8 = pick(df_legit, 135)
legit8["Amount"] = rng.uniform(20, 300, size=135).round(2)
fraud8 = pick(df_fraud, n8f)
fraud8["Amount"] = rng.uniform(150, 500, size=n8f).round(2)
df8 = pd.concat([legit8, fraud8]).sample(frac=1, random_state=7).reset_index(drop=True)
df8 = noise(df8)
save(df8, "08_ecommerce_returns_abuse.csv",
     "150 rows | 15 fraud | E-commerce with returns abuse pattern")


# ── CSV 9: International travel — unusual locations (5% fraud) ────────────
n9f = 10
legit9 = pick(df_legit, 190)
# Shift time to late night / early morning (international timezone)
legit9["Time"] = rng.integers(64800, 86400, size=190)
fraud9 = pick(df_fraud, n9f)
fraud9["Time"] = rng.integers(64800, 86400, size=n9f)
fraud9["Amount"] = rng.lognormal(5.5, 0.8, size=n9f).clip(100, 10000).round(2)
df9 = pd.concat([legit9, fraud9]).sort_values("Time").reset_index(drop=True)
df9 = noise(df9)
save(df9, "09_international_travel_card.csv",
     "200 rows | 10 fraud | International travel card (night txns)")


# ── CSV 10: ATM withdrawals — 50/50 split test ────────────────────────────
legit10 = pick(df_legit, 50)
legit10["Amount"] = rng.choice([20,40,50,60,80,100,200], size=50).astype(float)
fraud10 = pick(df_fraud, 50)
fraud10["Amount"] = rng.choice([20,40,50,60,80,100,200], size=50).astype(float)
df10 = pd.concat([legit10, fraud10]).sample(frac=1, random_state=99).reset_index(drop=True)
df10 = noise(df10, s=0.02)
save(df10, "10_atm_withdrawals_50_50.csv",
     "100 rows | 50 fraud | ATM withdrawals — 50/50 split stress test")


print(f"\nAll 5 new CSVs saved to:\n  {OUT}\n  {FE}")
