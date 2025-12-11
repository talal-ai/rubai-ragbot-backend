"""
Configuration settings for RubAI Backend
"""
import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from backend directory
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

# API Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/rubai_db")
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

# Supabase Configuration
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://tonodshnbuztozteuaon.supabase.co")
SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InRvbm9kc2huYnV6dG96dGV1YW9uIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQwNDk2MTYsImV4cCI6MjA3OTYyNTYxNn0.mH9YYTj1qB-dpZYmsgJWKPyIn-ndCQpZv7SlbALc9AE")
SUPABASE_STORAGE_BUCKET = "knowledge-base"
SUPABASE_STORAGE_URL = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}"

# LLM Provider Configuration
# Options: "gemini" or "ollama"
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini")

# Ollama Cloud Configuration
OLLAMA_API_KEY = os.getenv("OLLAMA_API_KEY")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ministral-3:8b-cloud").strip()
OLLAMA_API_BASE_URL = os.getenv("OLLAMA_API_BASE_URL", "https://ollama.com/api").strip()

# File Upload Limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
ALLOWED_FILE_TYPES = {'.pdf', '.txt'}
ALLOWED_MIME_TYPES = {'application/pdf', 'text/plain'}

# Message Limits
MAX_MESSAGE_LENGTH = 10000
MAX_QUERY_LENGTH = 500

# RAG Configuration
# Slightly larger chunks = fewer embeddings = faster ingestion
DEFAULT_CHUNK_SIZE = 1500
DEFAULT_CHUNK_OVERLAP = 250
DEFAULT_TOP_K = 5
MAX_TOP_K = 20
SIMILARITY_THRESHOLD = 0.3  # Lowered from 0.5 for better retrieval on vague queries

# Database Connection Pool
DB_POOL_SIZE = 10
DB_MAX_OVERFLOW = 20
DB_POOL_RECYCLE = 3600  # 1 hour

# Gemini Model (Free Tier)
# gemini-2.5-flash: Stable version (June 2025), 1M input tokens, 65K output tokens
# Alternative: gemini-2.0-flash-001 (stable, Jan 2025) or gemini-2.0-flash
GEMINI_MODEL = 'gemini-2.5-flash'
EMBEDDING_MODEL = 'models/text-embedding-004'
EMBEDDING_DIMENSION = 768

