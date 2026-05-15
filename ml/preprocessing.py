"""
preprocessing.py
----------------
Preprocessing pipeline for Credit Card Fraud Detection.

Steps:
  1. Load raw CSV (expects Kaggle 'creditcard.csv' columns)
  2. Drop duplicates / handle NaNs
  3. Scale Amount & Time features
  4. Split into train / test sets (stratified)
  5. Apply SMOTE to the training set to handle class imbalance
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib


# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RANDOM_STATE = 42
TEST_SIZE     = 0.20
SCALER_PATH   = os.path.join(os.path.dirname(__file__), "..", "backend", "model", "scaler.pkl")


def load_data(filepath: str) -> pd.DataFrame:
    """Load and do basic sanity-checks on the raw dataset."""
    print(f"[INFO] Loading data from: {filepath}")
    df = pd.read_csv(filepath)

    # Kaggle dataset columns: Time, V1…V28, Amount, Class
    required_cols = {"Time", "Amount", "Class"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    print(f"[INFO] Shape: {df.shape} | Fraud rate: {df['Class'].mean()*100:.4f}%")

    # Drop exact duplicates
    before = len(df)
    df = df.drop_duplicates()
    print(f"[INFO] Dropped {before - len(df)} duplicate rows.")

    # Drop NaNs
    df = df.dropna()
    return df


def scale_features(df: pd.DataFrame, fit: bool = True,
                   scaler: StandardScaler = None) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Scale 'Amount' and 'Time' columns (V1-V28 are already PCA-transformed).
    If fit=True, a new scaler is created and returned (for training).
    If fit=False, the provided scaler is applied (for inference).
    """
    cols_to_scale = ["Amount", "Time"]

    if fit:
        scaler = StandardScaler()
        df[cols_to_scale] = scaler.fit_transform(df[cols_to_scale])
        # Persist scaler so backend can use it at inference time
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[INFO] Scaler saved -> {SCALER_PATH}")
    else:
        if scaler is None:
            raise ValueError("A fitted scaler must be provided when fit=False.")
        df[cols_to_scale] = scaler.transform(df[cols_to_scale])

    return df, scaler


def split_data(df: pd.DataFrame) -> tuple:
    """Stratified train/test split."""
    X = df.drop(columns=["Class"])
    y = df["Class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(f"[INFO] Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"[INFO] Train fraud %: {y_train.mean()*100:.4f}%")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series,
                sampling_strategy: float = 0.1) -> tuple:
    """
    Apply SMOTE oversampling only on training set.
    sampling_strategy=0.1 -> minority : majority = 1:10 (avoids extreme oversampling).
    """
    print("[INFO] Applying SMOTE ...")
    sm = SMOTE(random_state=RANDOM_STATE, sampling_strategy=sampling_strategy)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(f"[INFO] After SMOTE -> X: {X_res.shape} | Fraud %: {y_res.mean()*100:.2f}%")
    return X_res, y_res


def preprocess_pipeline(filepath: str) -> tuple:
    """
    End-to-end preprocessing:
        raw CSV -> cleaned -> scaled -> split -> SMOTE
    Returns: X_train_res, X_test, y_train_res, y_test, scaler
    """
    df = load_data(filepath)
    df, scaler = scale_features(df, fit=True)
    X_train, X_test, y_train, y_test = split_data(df)
    X_train_res, y_train_res = apply_smote(X_train, y_train)
    return X_train_res, X_test, y_train_res, y_test, scaler


def preprocess_inference(df: pd.DataFrame, scaler: StandardScaler) -> pd.DataFrame:
    """
    Minimal preprocessing for a batch of transactions at inference time.
    Expects same column schema as training data (minus 'Class').
    """
    df = df.copy()

    # Fill missing with column medians (robust to missing fields)
    for col in df.columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)

    # Scale Amount + Time
    if "Amount" in df.columns and "Time" in df.columns:
        df, _ = scale_features(df, fit=False, scaler=scaler)

    return df
