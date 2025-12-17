"""
Script to create seed users (admins and clients) with password authentication
Uses UUID primary keys to match Supabase schema and foreign key constraints.
Run this script to set up admin and client accounts.
"""
import sys
import uuid
from pathlib import Path
from sqlalchemy.orm import Session
from sqlalchemy import text

# Add backend directory to path
backend_dir = Path(__file__).parent
sys.path.insert(0, str(backend_dir))

from database import SessionLocal
from auth import hash_password, get_user_by_username, User
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Default password for all seed users
DEFAULT_PASSWORD = "123456789"

# Seed users configuration: (username, role, email_domain)
SEED_USERS = [
    # Admin users
    ("admin1", "admin", "therubai.com"),
    ("admin2", "admin", "therubai.com"),
    ("admin3", "admin", "therubai.com"),
    # Client users - full access except admin dashboard
    ("client", "client", "therubai.com"),
]

def seed_users():
    """Create seed users with UUID primary keys if they don't exist"""
    db: Session = SessionLocal()
    try:
        created_count = 0
        updated_count = 0
        
        for username, role, email_domain in SEED_USERS:
            # Check if user already exists
            existing_user = get_user_by_username(db, username)
            
            if existing_user:
                logger.info(f"User '{username}' already exists (id: {existing_user.id}), checking updates...")
                # Update password hash and role if needed
                needs_update = False
                if not existing_user.password_hash:
                    existing_user.password_hash = hash_password(DEFAULT_PASSWORD)
                    needs_update = True
                if existing_user.role != role:
                    existing_user.role = role
                    needs_update = True
                if needs_update:
                    db.commit()
                    updated_count += 1
                    logger.info(f"Updated user '{username}' (role: {role})")
                continue
            
            # Create new user with UUID
            password_hash = hash_password(DEFAULT_PASSWORD)
            email = f"{username.lower()}@{email_domain}"
            user_uuid = str(uuid.uuid4())  # Generate proper UUID
            
            # Create user with specified role and UUID
            new_user = User(
                id=user_uuid,  # Explicitly set UUID
                email=email,
                name=f"{role.title()} - {username}",
                username=username,
                password_hash=password_hash,
                role=role,
                google_id=None
            )
            
            db.add(new_user)
            db.commit()
            db.refresh(new_user)
            
            created_count += 1
            logger.info(f"✅ Created {role} user: {username} (id: {user_uuid}, email: {email})")
        
        logger.info(f"\n🎉 Seed completed: {created_count} created, {updated_count} updated")
        logger.info(f"\n📝 Login credentials (all use password: {DEFAULT_PASSWORD}):")
        for username, role, _ in SEED_USERS:
            logger.info(f"   [{role.upper()}] Username: {username}")
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Error seeding users: {e}")
        import traceback
        traceback.print_exc()
        raise
    finally:
        db.close()

# Keep old function name for backwards compatibility
def seed_admin_users():
    """Backwards compatible alias for seed_users()"""
    seed_users()

if __name__ == "__main__":
    logger.info("🌱 Starting user seeding with UUID primary keys...")
    seed_users()
    logger.info("✅ User seeding completed!")
