"""
validators.py
-------------
Pydantic v2 models for input validation and response serialisation.
"""

from pydantic import BaseModel, Field, field_validator
from typing import Literal, Optional


# ── Request schemas ─────────────────────────────────────────────────────

class TransactionIn(BaseModel):
    """Single transaction input — matches Kaggle creditcard.csv schema."""

    Time  : float = Field(0.0,   description="Seconds elapsed from first transaction")
    V1    : float = Field(0.0)
    V2    : float = Field(0.0)
    V3    : float = Field(0.0)
    V4    : float = Field(0.0)
    V5    : float = Field(0.0)
    V6    : float = Field(0.0)
    V7    : float = Field(0.0)
    V8    : float = Field(0.0)
    V9    : float = Field(0.0)
    V10   : float = Field(0.0)
    V11   : float = Field(0.0)
    V12   : float = Field(0.0)
    V13   : float = Field(0.0)
    V14   : float = Field(0.0)
    V15   : float = Field(0.0)
    V16   : float = Field(0.0)
    V17   : float = Field(0.0)
    V18   : float = Field(0.0)
    V19   : float = Field(0.0)
    V20   : float = Field(0.0)
    V21   : float = Field(0.0)
    V22   : float = Field(0.0)
    V23   : float = Field(0.0)
    V24   : float = Field(0.0)
    V25   : float = Field(0.0)
    V26   : float = Field(0.0)
    V27   : float = Field(0.0)
    V28   : float = Field(0.0)
    Amount: float = Field(..., ge=0, description="Transaction amount in USD")

    @field_validator("Amount")
    @classmethod
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount must be ≥ 0")
        return v

    model_config = {"json_schema_extra": {
        "example": {
            "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.53,
            "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
            "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
            "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
            "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
            "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11,
            "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13,
            "V28": -0.02, "Amount": 149.62
        }
    }}


# ── Response schemas ────────────────────────────────────────────────────

class PredictionOut(BaseModel):
    """Single-transaction prediction response."""
    fraud_probability: float = Field(..., ge=0.0, le=1.0)
    risk_score       : float = Field(..., ge=0.0, le=100.0)
    label            : Literal["FRAUD", "LEGIT"]
    confidence       : Literal["HIGH", "MEDIUM", "LOW RISK", "SAFE"]


class BatchRowOut(BaseModel):
    """One row from a batch prediction response."""
    row              : int
    amount           : float
    fraud_probability: float
    risk_score       : float
    label            : Literal["FRAUD", "LEGIT"]
    confidence       : Literal["HIGH", "MEDIUM", "LOW RISK", "SAFE"]


class BatchPredictionOut(BaseModel):
    """Batch prediction response wrapper."""
    total_rows   : int
    fraud_count  : int
    legit_count  : int
    fraud_rate   : float
    predictions  : list[BatchRowOut]


class HealthOut(BaseModel):
    """Health check response."""
    status       : str
    model_loaded : bool
    model_name   : Optional[str] = None
    version      : str = "1.0.0"


class ErrorOut(BaseModel):
    """Standard error response."""
    detail: str
