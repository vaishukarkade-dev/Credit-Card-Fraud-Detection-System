# ── FraudShield AI — Production Dockerfile ──────────────────────────────
# Multi-stage build: slim Python image serving FastAPI + static frontend

FROM python:3.11-slim AS base

# Prevent Python from writing .pyc files and buffering stdout/stderr
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# ── Install dependencies ────────────────────────────────────────────────
COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ── Copy application code ──────────────────────────────────────────────
COPY backend/ ./backend/
COPY frontend/ ./frontend/

# ── Expose port ─────────────────────────────────────────────────────────
ENV PORT=8000
EXPOSE ${PORT}

# ── Health check ────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:${PORT}/health')" || exit 1

# ── Run with Gunicorn + Uvicorn workers ─────────────────────────────────
CMD ["sh", "-c", "gunicorn backend.main:app --bind 0.0.0.0:${PORT} --workers ${WEB_CONCURRENCY:-2} --worker-class uvicorn.workers.UvicornWorker --timeout 120 --access-logfile - --error-logfile -"]
