
import sys
from pathlib import Path
from sqlalchemy import text
import logging

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def update_role_constraint():
    """
    Update the check_valid_role constraint to include 'client' role.
    """
    db = SessionLocal()
    try:
        logger.info("🔄 Updating 'check_valid_role' constraint...")
        
        # 1. Drop existing constraint
        logger.info("Dropping existing constraint...")
        try:
            db.execute(text("ALTER TABLE users DROP CONSTRAINT IF EXISTS check_valid_role"))
            db.commit()
            logger.info("✅ Dropped existing constraint (if it existed)")
        except Exception as e:
            logger.warning(f"⚠️ Warning dropping constraint: {e}")
            db.rollback()

        # 2. Add new constraint
        logger.info("Adding new constraint with 'client' role...")
        try:
            db.execute(text("""
                ALTER TABLE users 
                ADD CONSTRAINT check_valid_role 
                CHECK (role IN ('user', 'admin', 'client'))
            """))
            db.commit()
            logger.info("✅ Successfully added new constraint allowing ('user', 'admin', 'client')")
        except Exception as e:
            logger.error(f"❌ Failed to add new constraint: {e}")
            db.rollback()
            raise

    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    update_role_constraint()
