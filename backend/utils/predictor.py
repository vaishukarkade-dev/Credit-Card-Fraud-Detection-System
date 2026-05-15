"""
predictor.py
------------
Loads the trained model + scaler at startup (singleton pattern) and
exposes two inference functions:
  • predict_single(transaction_dict) → FraudResult
  • predict_batch(dataframe)         → list[FraudResult]
"""

import os
import json
import logging
from functools import lru_cache
from typing import Any

import joblib
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ── Paths (resolved relative to this file) ─────────────────────────────
BASE_DIR    = os.path.dirname(os.path.dirname(__file__))   # backend/
MODEL_PATH  = os.path.join(BASE_DIR, "model", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "model", "scaler.pkl")
METRICS_PATH = os.path.join(BASE_DIR, "model", "metrics.json")

# ── Expected feature order (matches training) ───────────────────────────
FEATURE_COLS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)


# ── Singleton loaders ───────────────────────────────────────────────────
@lru_cache(maxsize=1)
def _load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            "Run `python ml/model_training.py --data <path>` first."
        )
    logger.info(f"Loading model from {MODEL_PATH}")
    return joblib.load(MODEL_PATH)


@lru_cache(maxsize=1)
def _load_scaler():
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError(
            f"Scaler not found at {SCALER_PATH}. "
            "Run the training script first."
        )
    logger.info(f"Loading scaler from {SCALER_PATH}")
    return joblib.load(SCALER_PATH)


def get_metrics() -> dict:
    """Return stored evaluation metrics (from training)."""
    if not os.path.exists(METRICS_PATH):
        return {}
    with open(METRICS_PATH) as f:
        return json.load(f)


# ── Pre-processing helpers ──────────────────────────────────────────────
def _build_df(data: dict | list[dict]) -> pd.DataFrame:
    """Convert input dict(s) to a properly ordered DataFrame."""
    if isinstance(data, dict):
        data = [data]
    df = pd.DataFrame(data)

    # Ensure all expected columns exist (fill missing with 0)
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0

    return df[FEATURE_COLS].astype(float)


def _scale(df: pd.DataFrame) -> pd.DataFrame:
    scaler = _load_scaler()
    df = df.copy()
    scale_cols = [c for c in ["Amount", "Time"] if c in df.columns]
    df[scale_cols] = scaler.transform(df[scale_cols])
    return df


def _risk_label(prob: float) -> tuple[str, str]:
    """Map fraud probability → (label, confidence)."""
    if prob >= 0.80:
        return "FRAUD",    "HIGH"
    elif prob >= 0.50:
        return "FRAUD",    "MEDIUM"
    elif prob >= 0.30:
        return "LEGIT",    "LOW RISK"
    else:
        return "LEGIT",    "SAFE"


# ── Public API ──────────────────────────────────────────────────────────
def predict_single(transaction: dict) -> dict:
    """
    Predict fraud for one transaction.

    Args:
        transaction: dict with keys matching FEATURE_COLS

    Returns:
        {
          "fraud_probability": float,   # 0.0 – 1.0
          "risk_score":        float,   # 0.0 – 100.0
          "label":             str,     # "FRAUD" | "LEGIT"
          "confidence":        str,     # "HIGH" | "MEDIUM" | "LOW RISK" | "SAFE"
        }
    """
    model  = _load_model()
    df     = _build_df(transaction)
    df_sc  = _scale(df)

    prob   = float(model.predict_proba(df_sc)[0, 1])
    label, confidence = _risk_label(prob)

    return {
        "fraud_probability": round(prob, 4),
        "risk_score":        round(prob * 100, 2),
        "label":             label,
        "confidence":        confidence,
    }


def predict_batch(df_input: pd.DataFrame) -> list[dict]:
    """
    Predict fraud for a batch of transactions (from CSV upload).

    Args:
        df_input: DataFrame — must contain same columns as training data
                  'Class' column is ignored if present.

    Returns:
        List of prediction dicts, one per row.
    """
    model  = _load_model()

    # Drop target column if accidentally included
    df = df_input.drop(columns=["Class"], errors="ignore").copy()

    # Ensure column alignment
    for col in FEATURE_COLS:
        if col not in df.columns:
            df[col] = 0.0
    df = df[FEATURE_COLS].astype(float)

    df_sc = _scale(df)
    probs  = model.predict_proba(df_sc)[:, 1]

    results = []
    for i, prob in enumerate(probs):
        label, confidence = _risk_label(float(prob))
        row_data = df_input.iloc[i].to_dict() if i < len(df_input) else {}
        results.append({
            "row":               i + 1,
            "amount":            round(float(df_input.get("Amount", pd.Series([0])).iloc[i] if "Amount" in df_input else 0), 2),
            "fraud_probability": round(float(prob), 4),
            "risk_score":        round(float(prob) * 100, 2),
            "label":             label,
            "confidence":        confidence,
        })

    return results
