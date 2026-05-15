"""
model_training.py
-----------------
Train, evaluate, and save the best fraud detection model.

Models compared:
  • Logistic Regression  (baseline)
  • Random Forest        (ensemble)
  • XGBoost              (gradient boosting — usually wins)

Metrics:
  Accuracy | Precision | Recall | F1-score | ROC-AUC

Usage:
  python model_training.py --data path/to/creditcard.csv
"""

import os
import sys
import argparse
import json
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")          # headless
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve
)
import xgboost as xgb

# local import
sys.path.insert(0, os.path.dirname(__file__))
from preprocessing import preprocess_pipeline

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────
MODEL_DIR   = os.path.join(os.path.dirname(__file__), "..", "backend", "model")
REPORT_DIR  = os.path.join(os.path.dirname(__file__), "reports")
BEST_MODEL  = os.path.join(MODEL_DIR, "best_model.pkl")
METRICS_JSON = os.path.join(MODEL_DIR, "metrics.json")

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(REPORT_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# Model definitions
# ─────────────────────────────────────────────
def get_models() -> dict:
    return {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, class_weight="balanced",
            n_jobs=-1, random_state=42
        ),
        "XGBoost": xgb.XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=100,   # handles imbalance directly
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=42, n_jobs=-1
        ),
    }


# ─────────────────────────────────────────────
# Evaluation helpers
# ─────────────────────────────────────────────
def evaluate(model, X_test, y_test, name: str) -> dict:
    """Compute all metrics and return as dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        "model"    : name,
        "accuracy" : round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall"   : round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1"       : round(f1_score(y_test, y_pred, zero_division=0), 4),
        "roc_auc"  : round(roc_auc_score(y_test, y_prob), 4),
    }
    print(f"\n{'='*55}")
    print(f"  {name}")
    print(f"{'='*55}")
    for k, v in metrics.items():
        if k != "model":
            print(f"  {k:<12}: {v}")
    return metrics, y_pred, y_prob


def plot_confusion(y_test, y_pred, name: str):
    """Save confusion matrix PNG."""
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Legit", "Fraud"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(f"{name} - Confusion Matrix", fontsize=12)
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    path = os.path.join(REPORT_DIR, f"cm_{safe_name}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[INFO] Confusion matrix -> {path}")


def plot_roc_curves(results: list, X_test, y_test):
    """Overlay ROC curves for all models."""
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1)

    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["probs"])
        ax.plot(fpr, tpr, lw=2, label=f"{r['name']} (AUC={r['roc_auc']:.3f})")

    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve Comparison")
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(REPORT_DIR, "roc_curves.png")
    plt.savefig(path, dpi=120)
    plt.close()
    print(f"[INFO] ROC curves -> {path}")


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Save feature importance chart (RF / XGBoost only)."""
    if not hasattr(model, "feature_importances_"):
        return
    imp = pd.Series(model.feature_importances_, index=feature_names)
    top = imp.nlargest(15)
    fig, ax = plt.subplots(figsize=(7, 5))
    top.sort_values().plot(kind="barh", ax=ax, color="#0EA5E9")
    ax.set_title(f"{model_name} - Top 15 Feature Importances")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    safe_name = model_name.lower().replace(" ", "_")
    path = os.path.join(REPORT_DIR, f"importance_{safe_name}.png")
    plt.savefig(path, dpi=100)
    plt.close()
    print(f"[INFO] Feature importance -> {path}")


# ─────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────
def train(data_path: str):
    print("\n[STEP 1] Preprocessing ...")
    X_train, X_test, y_train, y_test, scaler = preprocess_pipeline(data_path)
    feature_names = list(X_train.columns)

    print("\n[STEP 2] Training models ...")
    models   = get_models()
    all_metrics  = []
    roc_data     = []

    best_auc   = -1
    best_model_obj = None
    best_name  = ""

    for name, model in models.items():
        print(f"\n  -> Fitting {name} ...", end=" ", flush=True)
        model.fit(X_train, y_train)
        print("done.")

        metrics, y_pred, y_prob = evaluate(model, X_test, y_test, name)
        plot_confusion(y_test, y_pred, name)
        plot_feature_importance(model, feature_names, name)

        roc_data.append({"name": name, "probs": y_prob, "roc_auc": metrics["roc_auc"]})
        all_metrics.append(metrics)

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_model_obj = model
            best_name = name

    plot_roc_curves(roc_data, X_test, y_test)

    print(f"\n[RESULT] Best model: {best_name} (ROC-AUC={best_auc:.4f})")

    print(f"\n[STEP 3] Saving best model -> {BEST_MODEL}")
    joblib.dump(best_model_obj, BEST_MODEL)

    # Save metrics summary
    summary = {
        "best_model" : best_name,
        "best_roc_auc": best_auc,
        "feature_names": feature_names,
        "all_metrics"  : all_metrics,
    }
    with open(METRICS_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] Metrics saved -> {METRICS_JSON}")

    print("\n[OK] Training complete!\n")
    return best_model_obj, scaler, summary


# ─────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Credit Card Fraud — Model Training")
    parser.add_argument("--data", required=True,
                        help="Path to creditcard.csv from Kaggle")
    args = parser.parse_args()
    train(args.data)
