"""
debug_model.py — traces exactly what happens to known fraud rows through the pipeline
"""
import pandas as pd
import numpy as np
import joblib

# Load
df        = pd.read_csv("data/creditcard.csv")
scaler    = joblib.load("backend/model/scaler.pkl")
model     = joblib.load("backend/model/best_model.pkl")

FEATURE_COLS = ["Time"] + [f"V{i}" for i in range(1, 29)] + ["Amount"]

# Get 20 known fraud rows (raw from CSV — same as what API would receive)
fraud = df[df["Class"] == 1].head(20).copy()
legit = df[df["Class"] == 0].head(20).copy()

print("="*55)
print("  DEBUG: Model prediction on raw CSV rows")
print("="*55)

for label, subset in [("FRAUD", fraud), ("LEGIT", legit)]:
    X = subset.drop(columns=["Class"]).copy()
    # Apply scaler exactly as predictor does
    X[["Amount", "Time"]] = scaler.transform(X[["Amount", "Time"]])
    X = X[FEATURE_COLS]
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    correct = sum(preds == (1 if label == "FRAUD" else 0))
    print(f"\n  {label} rows (20 rows):")
    print(f"  Probabilities: {np.round(probs, 3)}")
    print(f"  Predictions:   {preds}")
    print(f"  Correct:       {correct}/20")
    print(f"  Prob range:    {probs.min():.3f} - {probs.max():.3f}")
