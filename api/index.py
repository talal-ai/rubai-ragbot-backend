"""
Vercel serverless handler for FastAPI app
"""
import sys
import os
from pathlib import Path

# Add parent directory to path so we can import main
sys.path.insert(0, str(Path(__file__).parent.parent))

# Initialize database connection on first import (for serverless)
try:
    from database import init_db
    # Initialize DB connection pool (will be reused across invocations)
    init_db()
except Exception as e:
    print(f"Warning: Database initialization failed: {e}")

from main import app
from mangum import Mangum

# Create Mangum handler for Vercel
# lifespan="off" disables startup/shutdown events (handled manually above)
handler = Mangum(app, lifespan="off")

