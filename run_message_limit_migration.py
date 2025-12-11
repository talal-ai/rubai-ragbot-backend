#!/usr/bin/env python3
"""
Run the message limit migration SQL script.
This script reads the SQL migration file and executes it against the database.
"""
import os
import sys
from pathlib import Path
from dotenv import load_dotenv
from sqlalchemy import create_engine, text
import logging

# Load environment variables
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rubai_db")

def run_migration():
    """Run the message limit migration"""
    try:
        # Read the SQL migration file
        migration_file = backend_dir / "migrations" / "add_message_limit.sql"
        
        if not migration_file.exists():
            logger.error(f"Migration file not found: {migration_file}")
            return False
        
        logger.info(f"Reading migration file: {migration_file}")
        with open(migration_file, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # Create database connection
        logger.info("Connecting to database...")
        engine = create_engine(DATABASE_URL)
        
        with engine.connect() as conn:
            # Begin transaction
            trans = conn.begin()
            try:
                logger.info("Executing migration SQL...")
                
                # Execute the entire SQL script at once
                # PostgreSQL functions with $$ need to be executed as a whole block
                logger.info("Executing complete migration script...")
                conn.execute(text(sql_script))
                
                # Commit transaction
                trans.commit()
                logger.info("✅ Migration completed successfully!")
                return True
                
            except Exception as e:
                trans.rollback()
                logger.error(f"❌ Migration failed: {e}")
                raise
        
    except Exception as e:
        logger.error(f"❌ Error running migration: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Message Limit Migration Script")
    logger.info("=" * 60)
    
    success = run_migration()
    
    if success:
        logger.info("✅ Migration completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Migration failed!")
        sys.exit(1)

