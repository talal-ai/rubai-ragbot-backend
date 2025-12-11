"""
Script to create admin users (admin1, admin2, admin3) with password 123456789
Run this script to set up admin accounts.
"""
import sys
from pathlib import Path
from sqlalchemy.orm import Session

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal
from auth import hash_password, get_user_by_username, create_user
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

ADMIN_PASSWORD = "123456789"
ADMIN_USERS = ["admin1", "admin2", "admin3"]

def seed_admin_users():
    """Create admin users if they don't exist"""
    db: Session = SessionLocal()
    try:
        created_count = 0
        for username in ADMIN_USERS:
            # Check if admin user already exists
            existing_user = get_user_by_username(db, username)
            
            if existing_user:
                logger.info(f"Admin user '{username}' already exists, skipping...")
                # Update password hash if needed
                if not existing_user.password_hash:
                    existing_user.password_hash = hash_password(ADMIN_PASSWORD)
                    existing_user.role = 'admin'
                    db.commit()
                    logger.info(f"Updated password for existing admin '{username}'")
                continue
            
            # Create new admin user
            password_hash = hash_password(ADMIN_PASSWORD)
            email = f"{username}@admin.local"  # Placeholder email
            
            # Import User model
            from auth import User
            
            # Create user with admin role directly
            admin_user = User(
                email=email,
                name=f"Admin {username}",
                username=username,
                password_hash=password_hash,
                role='admin',
                google_id=None
            )
            
            db.add(admin_user)
            db.commit()
            db.refresh(admin_user)
            
            created_count += 1
            logger.info(f"✅ Created admin user: {username} (email: {email})")
        
        logger.info(f"\n🎉 Successfully created/updated {created_count} admin users")
        logger.info(f"📝 Admin login credentials:")
        for username in ADMIN_USERS:
            logger.info(f"   Username: {username}, Password: {ADMIN_PASSWORD}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error seeding admin users: {e}")
        raise
    finally:
        db.close()

if __name__ == "__main__":
    logger.info("🌱 Starting admin user seeding...")
    seed_admin_users()
    logger.info("✅ Admin user seeding completed!")

