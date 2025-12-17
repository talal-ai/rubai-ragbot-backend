"""
Script to clean up old admin/client users and re-seed with UUID primary keys.
Run this ONCE to fix the foreign key constraint issues.
"""
import sys
from pathlib import Path
from sqlalchemy import text

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_and_reseed():
    """Delete old admin/client users and re-seed with UUID"""
    db = SessionLocal()
    try:
        logger.info("🧹 Cleaning up old admin/client users...")
        
        # Delete existing admin and client users (they have integer IDs)
        result = db.execute(text("DELETE FROM users WHERE role IN ('admin', 'client')"))
        deleted_count = result.rowcount
        db.commit()
        
        logger.info(f"✅ Deleted {deleted_count} old admin/client users")
        
        # Now run the seed script
        logger.info("🌱 Re-seeding users with UUID primary keys...")
        from seed_admin_users import seed_users
        seed_users()
        
        logger.info("✅ Cleanup and re-seed completed successfully!")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error during cleanup: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()

if __name__ == "__main__":
    cleanup_and_reseed()
