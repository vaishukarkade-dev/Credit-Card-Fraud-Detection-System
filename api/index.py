import os
import sys

# Dynamic path injection for serverless execution environment
current_dir = os.path.dirname(__file__)
root_dir = os.path.abspath(os.path.join(current_dir, ".."))
backend_dir = os.path.abspath(os.path.join(root_dir, "backend"))

sys.path.insert(0, backend_dir)
sys.path.insert(0, root_dir)

# Import the FastAPI application singleton
from backend.main import app

# Vercel serverless requires a variable named 'handler' or 'app' at module level
app = app
