"""
main.py
-------
FastAPI application — Credit Card Fraud Detection API.

Production-ready:
  • Serves frontend static files (single deployment)
  • CORS configured via env var
  • Model pre-loaded at startup
  • Structured logging
  • Request size limits on batch upload

Routes:
  GET  /           → serves index.html (frontend)
  GET  /health     → server + model status
  GET  /metrics    → model evaluation metrics
  POST /predict    → single transaction prediction
  POST /batch_predict → CSV batch prediction
"""

import io
import os
import logging
from pathlib import Path
from contextlib import asynccontextmanager

import pandas as pd
from fastapi import FastAPI, File, UploadFile, HTTPException, status, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

# Local utilities
import sys
sys.path.insert(0, os.path.dirname(__file__))
from utils.validators import (
    TransactionIn, PredictionOut,
    BatchPredictionOut, BatchRowOut,
    HealthOut, ErrorOut,
)
from utils.predictor import predict_single, predict_batch, get_metrics, _load_model, _load_scaler

# ── Logging ─────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "info").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("fraud_api")

# ── Config ──────────────────────────────────────────────────────────────
FRONTEND_ORIGIN = os.getenv("FRONTEND_ORIGIN", "*")
FRONTEND_DIR = Path(__file__).resolve().parent.parent / "frontend"


# ── Lifespan — warm up model at startup ─────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Pre-load model & scaler into memory so first request is instant."""
    logger.info("🚀 Starting up — loading model & scaler …")
    try:
        _load_model()
        _load_scaler()
        logger.info("✅ Model and scaler loaded successfully.")
    except FileNotFoundError as e:
        logger.warning(f"⚠️  Model not found at startup: {e}")
        logger.warning("    Run the training script first, or the /predict endpoint will fail.")
    yield
    logger.info("⏹️  Shutting down.")


# ── App ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title       = "FraudShield AI — Credit Card Fraud Detection API",
    description = "ML-powered fraud detection — single & batch predictions. "
                  "Upload CSV files or send single transactions for instant risk scoring.",
    version     = "2.0.0",
    docs_url    = "/docs",
    redoc_url   = "/redoc",
    lifespan    = lifespan,
)

# ── Middleware ───────────────────────────────────────────────────────────

# GZip compression for responses > 500 bytes (huge perf gain for batch results)
app.add_middleware(GZipMiddleware, minimum_size=500)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins     = [FRONTEND_ORIGIN] if FRONTEND_ORIGIN != "*" else ["*"],
    allow_credentials = True,
    allow_methods     = ["*"],
    allow_headers     = ["*"],
)


# ── Security headers middleware ──────────────────────────────────────────
@app.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
    # Cache static assets aggressively, API responses not at all
    if request.url.path.startswith(("/styles/", "/scripts/", "/assets/", "/samples/")):
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    elif request.url.path.startswith(("/predict", "/batch_predict", "/health", "/metrics")):
        response.headers["Cache-Control"] = "no-store"
    return response


# ─────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────

@app.get(
    "/health",
    response_model=HealthOut,
    summary="Health Check",
    tags=["Status"],
)
async def health():
    """Check if the server is running and the model is loaded."""
    model_loaded = False
    model_name   = None

    try:
        _load_model()
        model_loaded = True
        metrics = get_metrics()
        model_name = metrics.get("best_model", "Unknown")
    except Exception:
        pass

    return HealthOut(
        status      = "ok",
        model_loaded= model_loaded,
        model_name  = model_name,
    )


@app.get(
    "/metrics",
    summary="Model Metrics",
    tags=["Status"],
)
async def metrics():
    """Return model evaluation metrics from training."""
    data = get_metrics()
    if not data:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Metrics not found. Train the model first."
        )
    return JSONResponse(content=data)


@app.post(
    "/predict",
    response_model=PredictionOut,
    summary="Single Transaction Prediction",
    tags=["Prediction"],
)
async def predict(transaction: TransactionIn):
    """
    Predict whether a single transaction is fraudulent.

    Returns fraud probability (0–1), risk score (0–100%), label, and confidence.
    """
    try:
        result = predict_single(transaction.model_dump())
        return PredictionOut(**result)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {e}. Train the model first."
        )
    except Exception as e:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Prediction error: {str(e)}"
        )


@app.post(
    "/batch_predict",
    response_model=BatchPredictionOut,
    summary="Batch CSV Prediction",
    tags=["Prediction"],
)
async def batch_predict(file: UploadFile = File(...)):
    """
    Upload a CSV file of transactions and get fraud predictions for all rows.

    - CSV must contain at minimum: Time, V1–V28, Amount
    - 'Class' column is ignored if present
    - Returns predictions with fraud probability and risk score for each row
    """
    # ── Validate file type ──────────────────────────────────────────────
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only .csv files are accepted."
        )

    # ── Read CSV ────────────────────────────────────────────────────────
    try:
        content = await file.read()
        df = pd.read_csv(io.StringIO(content.decode("utf-8")))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Could not parse CSV: {e}"
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="The uploaded CSV file is empty."
        )

    # ── Cap rows to prevent abuse ───────────────────────────────────────
    MAX_ROWS = 50_000
    if len(df) > MAX_ROWS:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"CSV exceeds maximum of {MAX_ROWS:,} rows."
        )

    # ── Predict ─────────────────────────────────────────────────────────
    try:
        results = predict_batch(df)
    except FileNotFoundError as e:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Model not loaded: {e}"
        )
    except Exception as e:
        logger.exception("Batch prediction failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch prediction error: {str(e)}"
        )

    # ── Assemble response ───────────────────────────────────────────────
    fraud_count = sum(1 for r in results if r["label"] == "FRAUD")
    legit_count = len(results) - fraud_count

    return BatchPredictionOut(
        total_rows  = len(results),
        fraud_count = fraud_count,
        legit_count = legit_count,
        fraud_rate  = round(fraud_count / len(results) * 100, 4) if results else 0.0,
        predictions = [BatchRowOut(**r) for r in results],
    )


# ─────────────────────────────────────────────────────────────────────────
# Static Frontend Serving (production — single deployment)
# ─────────────────────────────────────────────────────────────────────────

if FRONTEND_DIR.exists():
    # Serve sub-directories as static mounts
    for subdir in ["styles", "scripts", "assets", "samples"]:
        sub_path = FRONTEND_DIR / subdir
        if sub_path.exists():
            app.mount(f"/{subdir}", StaticFiles(directory=str(sub_path)), name=subdir)

    # Serve HTML pages explicitly so SPA-like routing works
    @app.get("/", include_in_schema=False)
    async def serve_index():
        return FileResponse(str(FRONTEND_DIR / "index.html"), media_type="text/html")

    @app.get("/upload.html", include_in_schema=False)
    async def serve_upload():
        return FileResponse(str(FRONTEND_DIR / "upload.html"), media_type="text/html")

    @app.get("/dashboard.html", include_in_schema=False)
    async def serve_dashboard():
        return FileResponse(str(FRONTEND_DIR / "dashboard.html"), media_type="text/html")

    @app.get("/upload", include_in_schema=False)
    async def serve_upload_clean():
        return FileResponse(str(FRONTEND_DIR / "upload.html"), media_type="text/html")

    @app.get("/dashboard", include_in_schema=False)
    async def serve_dashboard_clean():
        return FileResponse(str(FRONTEND_DIR / "dashboard.html"), media_type="text/html")

    @app.get("/favicon.ico", include_in_schema=False)
    async def favicon():
        favicon_path = FRONTEND_DIR / "assets" / "favicon.ico"
        if favicon_path.exists():
            return FileResponse(str(favicon_path))
        # Return shield emoji as SVG favicon fallback
        svg = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
            <text y=".9em" font-size="90">🛡️</text></svg>'''
        return JSONResponse(content=svg, media_type="image/svg+xml",
                           headers={"Content-Type": "image/svg+xml"})

    logger.info(f"📁 Serving frontend from: {FRONTEND_DIR}")
else:
    # No frontend directory — redirect to API docs
    @app.get("/", include_in_schema=False)
    async def root():
        from fastapi.responses import RedirectResponse
        return RedirectResponse(url="/docs")

    logger.warning(f"⚠️  Frontend directory not found at {FRONTEND_DIR}. Serving API-only mode.")
