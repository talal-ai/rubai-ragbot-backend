from sqlalchemy import create_engine, Column, BigInteger, String, Text, CheckConstraint, UniqueConstraint, Index, text
from sqlalchemy.dialects.postgresql import TIMESTAMP, JSONB, UUID
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from sqlalchemy.sql import func
from pgvector.sqlalchemy import Vector
import os
from dotenv import load_dotenv
import logging

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rubai_db")

# Create SQLAlchemy engine with connection pooling
engine = create_engine(
    DATABASE_URL,
    pool_size=10,  # Number of connections to keep open
    max_overflow=20,  # Additional connections if pool is full
    pool_pre_ping=True,  # Verify connections before using
    pool_recycle=3600,  # Recycle connections after 1 hour
    echo=False  # Set to True for SQL query logging
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# Document model with vector embeddings
class Document(Base):
    __tablename__ = "documents"
    
    id = Column(BigInteger, primary_key=True, index=True)
    user_id = Column(UUID(as_uuid=False), nullable=True, index=True)  # Supabase auth user UUID
    filename = Column(String(500), nullable=False)
    content = Column(Text, nullable=False)
    chunk_index = Column(BigInteger, nullable=False)
    page_number = Column(BigInteger, nullable=True)  # Page number for PDFs, NULL for text files
    embedding = Column(Vector(768), nullable=True)  # Gemini embedding dimension (legacy - deprecated)
    embedding_new = Column(Vector(1024), nullable=True)  # BGE-large embeddings (1024-dim) - ACTIVE
    text_id = Column(String(255), nullable=True, index=True)  # LlamaIndex node ID
    storage_url = Column(String(2048), nullable=True)  # Supabase Storage file URL
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), nullable=False)
    doc_metadata = Column('metadata', JSONB, server_default='{}', nullable=False)
    
    # Table constraints
    __table_args__ = (
        CheckConstraint('chunk_index >= 0', name='documents_chunk_index_positive'),
        CheckConstraint("length(trim(filename)) > 0", name='documents_filename_not_empty'),
        CheckConstraint("length(trim(content)) > 0", name='documents_content_not_empty'),
        UniqueConstraint('user_id', 'filename', 'chunk_index', name='documents_unique_user_chunk'),
        Index('idx_documents_user_id', 'user_id'),
        Index('idx_documents_filename', 'filename'),
        Index('idx_documents_created_at', 'created_at'),
        Index('idx_documents_user_filename', 'user_id', 'filename'),
    )

# Initialize database connection test
def init_db():
    """Test database connection"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        logger.info("✅ Database connection successful")
        return True
    except Exception as e:
        logger.error(f"❌ Database connection error: {e}")
        raise

# Dependency to get DB session
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Health check for database
def check_db_health():
    """Check if database is accessible"""
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception:
        return False
