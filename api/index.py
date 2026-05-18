"""
api/index.py
------------
Vercel Serverless Function entry point for FraudShield AI FastAPI backend.
Vercel @vercel/python builder looks for a variable named 'app' at module level.
"""

import os
import sys

# ── Resolve paths so imports work inside Vercel's serverless sandbox ───
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BACKEND = os.path.join(ROOT, "backend")

for path in [ROOT, BACKEND]:
    if path not in sys.path:
        sys.path.insert(0, path)

# ── Import FastAPI app ──────────────────────────────────────────────────
from main import app  # noqa: E402  (backend/main.py)

# Mount prefix dynamically for Vercel experimentalServices routing
app.root_path = "/_/backend"

