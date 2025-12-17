"""Clean up test data from database"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from database import SessionLocal
from sqlalchemy import text

db = SessionLocal()

# Delete test documents
deleted = db.execute(
    text("DELETE FROM documents WHERE filename LIKE 'phase1%' OR filename LIKE 'test_%'")
).rowcount

db.commit()
print(f"Cleaned: Deleted {deleted} test chunks")

total = db.execute(text("SELECT COUNT(*) FROM documents")).scalar()
print(f"Total documents in DB now: {total}")

db.close()
