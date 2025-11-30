"""
Authentication module for RubAI Backend
Simple Google OAuth + JWT implementation
"""
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt  # type: ignore
from datetime import datetime, timedelta
from typing import Optional, Any
from pydantic import BaseModel
from sqlalchemy.orm import Session
from sqlalchemy import Column, String, Text, BigInteger
from sqlalchemy.dialects.postgresql import TIMESTAMP
from sqlalchemy.sql import func
import httpx
import logging
import os

from database import Base, get_db

logger = logging.getLogger(__name__)

# Configuration
JWT_SECRET = os.getenv("JWT_SECRET", "your-super-secret-jwt-key-change-in-production")
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24 * 7  # 7 days

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")

# Security
security = HTTPBearer(auto_error=False)


# ==================== USER MODEL ====================

class User(Base):
    """User model for authentication"""
    __tablename__ = "users"
    
    id = Column(BigInteger, primary_key=True, index=True)
    email = Column(String(255), unique=True, nullable=False, index=True)
    name = Column(String(255), nullable=True)
    avatar_url = Column(Text, nullable=True)
    google_id = Column(String(255), unique=True, nullable=True, index=True)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())


# ==================== PYDANTIC MODELS ====================

class GoogleAuthRequest(BaseModel):
    """Request model for Google OAuth"""
    credential: str  # Google ID token from frontend


class TokenResponse(BaseModel):
    """Response model for authentication"""
    access_token: str
    token_type: str = "bearer"
    user: dict


class UserResponse(BaseModel):
    """User data response"""
    id: int
    email: str
    name: Optional[str] = None
    avatar_url: Optional[str] = None


# ==================== JWT FUNCTIONS ====================

def create_access_token(user_id: str, email: str) -> str:
    """Create JWT access token"""
    expire = datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS)
    payload = {
        "sub": str(user_id),
        "email": email,
        "exp": expire,
        "iat": datetime.utcnow()
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def verify_token(token: str) -> Optional[dict]:
    """Verify JWT token and return payload"""
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except JWTError as e:
        logger.warning(f"Token verification failed: {e}")
        return None


# ==================== GOOGLE OAUTH ====================

async def verify_google_token(credential: str) -> Optional[dict]:
    """
    Verify Google ID token and return user info
    Uses Google's tokeninfo endpoint for simplicity
    """
    try:
        # Verify token with Google
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"https://oauth2.googleapis.com/tokeninfo?id_token={credential}"
            )
            
            if response.status_code != 200:
                logger.error(f"Google token verification failed: {response.text}")
                return None
            
            data = response.json()
            
            # Verify audience (client ID) if configured
            if GOOGLE_CLIENT_ID and data.get("aud") != GOOGLE_CLIENT_ID:
                logger.error("Token audience mismatch")
                return None
            
            return {
                "google_id": data.get("sub"),
                "email": data.get("email"),
                "name": data.get("name"),
                "avatar_url": data.get("picture")
            }
            
    except Exception as e:
        logger.error(f"Error verifying Google token: {e}")
        return None


# ==================== USER CRUD ====================

def get_user_by_email(db: Session, email: str) -> Optional[User]:
    """Get user by email"""
    return db.query(User).filter(User.email == email).first()


def get_user_by_id(db: Session, user_id: int) -> Optional[User]:
    """Get user by ID"""
    return db.query(User).filter(User.id == user_id).first()


def get_user_by_google_id(db: Session, google_id: str) -> Optional[User]:
    """Get user by Google ID"""
    return db.query(User).filter(User.google_id == google_id).first()


def create_user(
    db: Session, 
    email: str, 
    name: Optional[str] = None, 
    avatar_url: Optional[str] = None, 
    google_id: Optional[str] = None
) -> User:
    """Create new user"""
    user = User(
        email=email,
        name=name,
        avatar_url=avatar_url,
        google_id=google_id
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"Created new user: {email}")
    return user


def update_user(
    db: Session, 
    user: User, 
    name: Optional[str] = None, 
    avatar_url: Optional[str] = None
) -> User:
    """Update existing user"""
    if name is not None:
        setattr(user, 'name', name)
    if avatar_url is not None:
        setattr(user, 'avatar_url', avatar_url)
    db.commit()
    db.refresh(user)
    return user


# ==================== AUTH DEPENDENCIES ====================

async def get_current_user_optional(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
    db: Session = Depends(get_db)
) -> Optional[User]:
    """
    Get current user from JWT token (OPTIONAL - returns None if no token)
    Use this for endpoints that work with or without auth
    """
    if not credentials:
        return None
    
    payload = verify_token(credentials.credentials)
    if not payload:
        return None
    
    sub = payload.get("sub")
    if sub is None:
        return None
    
    user_id = int(sub)
    user = get_user_by_id(db, user_id)
    return user


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(HTTPBearer()),
    db: Session = Depends(get_db)
) -> User:
    """
    Get current user from JWT token (REQUIRED)
    Use this for endpoints that require authentication
    """
    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    payload = verify_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    sub = payload.get("sub")
    if sub is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token payload"
        )
    
    user_id = int(sub)
    user = get_user_by_id(db, user_id)
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found"
        )
    
    return user


def user_to_dict(user: User) -> dict:
    """Convert User model to dict"""
    return {
        "id": user.id,
        "email": user.email,
        "name": user.name,
        "avatar_url": user.avatar_url
    }
