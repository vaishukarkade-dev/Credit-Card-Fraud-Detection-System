# 🛡️ FraudShield AI — Credit Card Fraud Detection

> **AI-powered credit card fraud detection system** with a production-grade FastAPI backend, Random Forest ML model (99.74% ROC-AUC), and a premium glassmorphism frontend.

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3+-F7931E?logo=scikit-learn&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)
![Render](https://img.shields.io/badge/Render-Deploy-46E3B7?logo=render&logoColor=white)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Live Demo](#live-demo)
- [Architecture](#architecture)
- [Features](#features)
- [Model Performance](#model-performance)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [API Reference](#api-reference)
- [Deployment](#deployment)
- [Configuration](#configuration)
- [Tech Stack](#tech-stack)

---

## Overview

FraudShield AI is an end-to-end fraud detection system that:
1. **Trains** a Random Forest classifier on 284,807 real-world credit card transactions (Kaggle dataset)
2. **Serves** predictions via a FastAPI REST API with sub-50ms latency
3. **Visualizes** results in a premium dark-mode dashboard with Chart.js

The system handles both **batch CSV uploads** (up to 50,000 rows) and **single-transaction** real-time predictions.

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    FraudShield AI                        │
├──────────────────┬──────────────────────────────────────┤
│   Frontend       │         Backend (FastAPI)             │
│  ┌────────────┐  │  ┌──────────┐  ┌──────────────────┐ │
│  │ index.html │──┼─▶│ /predict │  │ Random Forest    │ │
│  │ upload.html│──┼─▶│ /batch   │──│ + StandardScaler │ │
│  │ dashboard  │  │  │ /health  │  │ + SMOTE          │ │
│  └────────────┘  │  └──────────┘  └──────────────────┘ │
│  Tailwind CSS    │  GZip · CORS · Security Headers     │
│  Chart.js        │  Gunicorn + Uvicorn Workers          │
└──────────────────┴──────────────────────────────────────┘
```

---

## Features

| Feature | Description |
|---------|-------------|
| 📊 **Batch CSV Upload** | Drag & drop CSV, process up to 50K transactions |
| ⚡ **Single Prediction** | Real-time fraud check with PCA feature inputs |
| 🎯 **Risk Scoring** | 0–100% risk score with HIGH/MEDIUM/SAFE labels |
| 📈 **Analytics Dashboard** | Pie charts, bar charts, line graphs, top-20 fraud table |
| 📥 **CSV Export** | Download all predictions as CSV |
| 🌗 **Dark/Light Mode** | Theme toggle persisted in localStorage |
| 🔒 **Security Headers** | X-Content-Type-Options, X-Frame-Options, Referrer-Policy |
| 🗜️ **GZip Compression** | Automatic response compression for batch results |
| 🐳 **Docker Ready** | Single-command deployment with Docker Compose |
| ☁️ **Render/Heroku** | One-click deploy via render.yaml or Procfile |

---

## Model Performance

Trained on **284,807 transactions** with **SMOTE** class balancing:

| Model | Precision | Recall | F1 | ROC-AUC |
|-------|-----------|--------|----|---------|
| Logistic Regression | 0.87 | 0.62 | 0.73 | 0.9700 |
| XGBoost | 0.94 | 0.83 | 0.88 | 0.9870 |
| **Random Forest ✅** | **0.95** | **0.79** | **0.86** | **0.9974** |

> **Best Model:** Random Forest — auto-selected by highest ROC-AUC score.

---

## Quick Start

### Prerequisites
- Python 3.11+
- pip

### 1. Clone & Install

```bash
git clone https://github.com/your-username/fraudshield-ai.git
cd fraudshield-ai
pip install -r backend/requirements.txt
```

### 2. Train the Model (if no pre-trained model exists)

```bash
python ml/model_training.py --data data/creditcard.csv
```

This generates:
- `backend/model/best_model.pkl` — trained Random Forest
- `backend/model/scaler.pkl` — StandardScaler for Amount/Time
- `backend/model/metrics.json` — evaluation results

### 3. Start the Backend

```bash
cd backend
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open the Frontend

**Option A** — Served by FastAPI (recommended for production):
```
http://localhost:8000
```

**Option B** — Live Server (for development):
```
Open frontend/index.html with VS Code Live Server on port 5500
```

> The frontend auto-detects the API URL. When served by FastAPI (same origin), it uses relative URLs. When on Live Server (port 5500), it falls back to `http://localhost:8000`.

---

## Project Structure

```
fraudshield-ai/
├── backend/
│   ├── main.py              # FastAPI app — routes + static serving
│   ├── requirements.txt     # Python dependencies
│   ├── model/
│   │   ├── best_model.pkl   # Trained Random Forest
│   │   ├── scaler.pkl       # StandardScaler
│   │   └── metrics.json     # Training metrics
│   └── utils/
│       ├── predictor.py     # Model inference (single + batch)
│       └── validators.py    # Pydantic v2 request/response schemas
├── frontend/
│   ├── index.html           # Landing page
│   ├── upload.html          # Upload + single predict page
│   ├── dashboard.html       # Analytics dashboard
│   ├── styles/
│   │   └── main.css         # Design system (dark/light, animations)
│   ├── scripts/
│   │   ├── main.js          # Shared utilities (theme, scroll, counters)
│   │   ├── upload.js        # CSV upload + prediction logic
│   │   └── dashboard.js     # Chart.js visualizations
│   └── samples/             # 10 pre-built test CSV files
├── ml/
│   ├── model_training.py    # Training pipeline (SMOTE + 3 models)
│   ├── preprocessing.py     # Data cleaning & feature engineering
│   ├── generate_dataset.py  # Synthetic dataset generator
│   └── generate_sample.py   # Sample CSV creator
├── data/
│   ├── creditcard.csv       # Kaggle dataset (not in git)
│   └── samples/             # Generated test CSVs
├── Dockerfile               # Production container
├── docker-compose.yml       # One-command deployment
├── Procfile                 # Render/Heroku process definition
├── render.yaml              # Render.com blueprint
├── .env.example             # Environment variable template
├── .gitignore               # Git exclusions
└── README.md                # This file
```

---

## API Reference

Base URL: `http://localhost:8000`

### `GET /health`
Health check — returns model status.

```json
{ "status": "ok", "model_loaded": true, "model_name": "RandomForest", "version": "2.0.0" }
```

### `GET /metrics`
Returns training evaluation metrics.

### `POST /predict`
Single transaction prediction.

**Request:**
```json
{
  "Time": 0, "V1": -1.36, "V2": -0.07, "V3": 2.53,
  "V4": 1.38, "V5": -0.34, "V6": 0.46, "V7": 0.24,
  "V8": 0.10, "V9": 0.36, "V10": 0.09, "V11": -0.55,
  "V12": -0.62, "V13": -0.99, "V14": -0.31, "V15": 1.47,
  "V16": -0.47, "V17": 0.21, "V18": 0.03, "V19": 0.40,
  "V20": 0.25, "V21": -0.02, "V22": 0.28, "V23": -0.11,
  "V24": 0.07, "V25": 0.13, "V26": -0.19, "V27": 0.13,
  "V28": -0.02, "Amount": 149.62
}
```

**Response:**
```json
{
  "fraud_probability": 0.0023,
  "risk_score": 0.23,
  "label": "LEGIT",
  "confidence": "SAFE"
}
```

### `POST /batch_predict`
Upload CSV file for batch predictions.

**Request:** `multipart/form-data` with `file` field (CSV)

**Response:**
```json
{
  "total_rows": 100,
  "fraud_count": 12,
  "legit_count": 88,
  "fraud_rate": 12.0,
  "predictions": [
    { "row": 1, "amount": 149.62, "fraud_probability": 0.95, "risk_score": 95.0, "label": "FRAUD", "confidence": "HIGH" }
  ]
}
```

### Interactive Docs
- **Swagger UI:** `http://localhost:8000/docs`
- **ReDoc:** `http://localhost:8000/redoc`

---

## Deployment

### 🐳 Docker (Recommended)

```bash
# Build and run
docker compose up --build

# Or without compose
docker build -t fraudshield-ai .
docker run -p 8000:8000 fraudshield-ai
```

### ☁️ Render.com

1. Push to GitHub
2. Go to [Render Dashboard](https://dashboard.render.com) → **New > Blueprint**
3. Connect your repo — `render.yaml` auto-configures everything
4. Deploy 🚀

### 🟣 Heroku

```bash
heroku create fraudshield-ai
git push heroku main
```

### Production Settings

```bash
# Copy env template
cp .env.example .env

# Edit for production
FRONTEND_ORIGIN=https://your-domain.com
WEB_CONCURRENCY=4
LOG_LEVEL=warning
```

---

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `PORT` | `8000` | Server port |
| `HOST` | `0.0.0.0` | Bind address |
| `FRONTEND_ORIGIN` | `*` | CORS allowed origin |
| `WEB_CONCURRENCY` | `2` | Gunicorn worker count |
| `LOG_LEVEL` | `info` | Logging level |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| **Backend** | FastAPI, Gunicorn, Uvicorn |
| **ML** | scikit-learn, XGBoost, imbalanced-learn (SMOTE) |
| **Frontend** | HTML5, Tailwind CSS 3.4, Chart.js 4.4 |
| **Data** | pandas, NumPy |
| **Deployment** | Docker, Render, Heroku |

---

## Dataset

[Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions (492 frauds = 0.17%)
- 30 features: Time, V1–V28 (PCA), Amount
- Anonymized European cardholder transactions (Sept 2013)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <strong>🛡️ FraudShield AI</strong> · Built with FastAPI + Random Forest + ❤️
</p>
#   C r e d i t - C a r d - F r a u d - D e t e c t i o n - S y s t e m  
 