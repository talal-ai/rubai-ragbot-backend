from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List, Tuple
import os
from dotenv import load_dotenv
from fastapi.responses import StreamingResponse
import json
from sqlalchemy.orm import Session
import logging
from pathlib import Path
from sqlalchemy import text
import mimetypes
import re
from jose import jwt, JWTError, ExpiredSignatureError  # type: ignore

# Import our RAG modules
from database import Document, SessionLocal, engine, get_db, check_db_health, init_db
from document_processor import process_document, extract_text_from_pdf, chunk_text
from vector_store import store_document_chunks, search_similar_chunks, get_all_documents, delete_document
from config import LLM_PROVIDER, GEMINI_API_KEY
from llm_provider import create_llm_provider
from embeddings import generate_embedding

# Import LlamaIndex RAG service
from llamaindex_service import RAGService
from chat_sync import sync_chat_session
from supabase_storage import VALID_CATEGORIES, list_files_in_category

# Import auth module
from auth import (
    GoogleAuthRequest, TokenResponse, User, AdminLoginRequest, UnifiedLoginRequest,
    verify_google_token, get_user_by_google_id, get_user_by_email,
    get_user_by_username, verify_password, hash_password, is_admin,
    create_user, update_user, user_to_dict,
    get_current_user, get_current_user_optional, create_access_token,
    get_user_by_email_or_username
)

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants for validation
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_MESSAGE_LENGTH = 10000
ALLOWED_FILE_TYPES = {'.pdf', '.txt'}
ALLOWED_MIME_TYPES = {
    'application/pdf',
    'text/plain',
}

# Supabase JWT secret (from environment)
SUPABASE_JWT_SECRET = os.getenv("SUPABASE_JWT_SECRET", "")

# ==================== AUTH HELPERS ====================

async def get_supabase_user_id(authorization: Optional[str] = Header(None)) -> Optional[str]:
    """
    Extract user_id from Supabase JWT token in Authorization header.
    Returns None if no valid token is provided (allows anonymous access).
    Verifies JWT signature if SUPABASE_JWT_SECRET is configured.
    Also checks for admin JWT tokens (from our backend auth).
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        # First, try to decode as admin token (uses our JWT_SECRET)
        try:
            from auth import JWT_SECRET, JWT_ALGORITHM, verify_token
            payload = verify_token(token)
            if payload:
                # Check if this is an admin token (has role='admin')
                if payload.get("role") == "admin":
                    logger.info(f"Admin user detected from token: {payload.get('email')}")
                    # Return a special marker for admin users
                    return "admin"
                # If it's our backend token but not admin, extract user_id
                user_id = payload.get("sub")
                if user_id:
                    logger.info(f"Backend user token detected: {user_id}")
                    return user_id
        except:
            pass
        
        # If JWT secret is configured, verify the token properly (Supabase token)
        if SUPABASE_JWT_SECRET:
            try:
                payload = jwt.decode(
                    token, 
                    SUPABASE_JWT_SECRET, 
                    algorithms=["HS256"],
                    options={"verify_aud": False}  # Supabase doesn't always set aud
                )
                user_id = payload.get("sub")
                if user_id:
                    logger.info(f"Verified user_id from Supabase token: {user_id} (type: {type(user_id)})")
                else:
                    logger.warning(f"No 'sub' claim found in JWT payload. Available keys: {list(payload.keys())}")
                return user_id
            except ExpiredSignatureError:
                logger.warning("JWT token has expired")
                return None
            except JWTError as e:
                logger.warning(f"JWT verification failed: {e}")
                return None
        else:
            # Fallback: Decode without verification (development only)
            logger.warning("SUPABASE_JWT_SECRET not set - token not verified!")
            import base64
            parts = token.split('.')
            if len(parts) != 3:
                logger.warning("Invalid JWT format")
                return None
            
            payload_encoded = parts[1]
            padding = len(payload_encoded) % 4
            if padding:
                payload_encoded += '=' * (4 - padding)
            
            payload_bytes = base64.urlsafe_b64decode(payload_encoded)
            payload = json.loads(payload_bytes.decode('utf-8'))
            
            user_id = payload.get("sub")
            if user_id:
                logger.info(f"Extracted user_id from Supabase token (unverified): {user_id} (type: {type(user_id)})")
            else:
                logger.warning(f"No 'sub' claim found in JWT payload. Available keys: {list(payload.keys())}")
            return user_id
    except Exception as e:
        logger.warning(f"Failed to extract user_id from token: {e}")
        return None


async def is_admin_user(authorization: Optional[str] = Header(None)) -> bool:
    """Check if the current user is an admin from JWT token"""
    if not authorization:
        return False
    
    try:
        from auth import verify_token
        token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
        payload = verify_token(token)
        return payload and payload.get("role") == "admin"
    except:
        return False


async def require_supabase_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Require a valid Supabase user. Raises 401 if not authenticated.
    Admin users are allowed (they return "admin" string).
    """
    user_id = await get_supabase_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id

def get_admin_chat_user_id(admin_user_id: str) -> str:
    """
    Convert admin user_id to a UUID for chat_sessions table.
    Uses a consistent UUID based on admin identifier.
    All admin users share the same chat sessions UUID.
    """
    # Use a fixed UUID for all admin users: 00000000-0000-0000-0000-000000000001
    # This allows admin chat sessions to be stored in chat_sessions table
    return "00000000-0000-0000-0000-000000000001"

app = FastAPI(  # type: ignore[call-arg]
    title="RubAI Backend API",
    description="Professional RAG Chatbot System",
    version="1.0.0"
)

# Initialize database on startup
@app.on_event("startup")
async def startup_event():
    try:
        init_db()
        logger.info("✅ Application started successfully")
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise

# CORS Configuration - Update with your frontend URL in production
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:5173").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== MODELS ====================

class Attachment(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    type: str = Field(..., min_length=1)
    data: str = Field(..., min_length=1)

class ChatMessage(BaseModel):
    message: str = Field(..., min_length=1, max_length=MAX_MESSAGE_LENGTH)
    tone: Optional[str] = Field(default="Professional", max_length=50)
    attachment: Optional[Attachment] = None
    selected_documents: Optional[List[str]] = None  # Document filenames to search within
    category: Optional[str] = None  # Storage category to search within (privacy-policies, cvs, etc.)
    knowledge_base_mode: Optional[str] = Field(default="none", description="Knowledge base access mode: 'none', 'folder', 'file', 'all'")
    chat_id: str = Field(..., min_length=1, description="Unique chat session identifier for conversation memory")  # NEW: Required for memory
    language: Optional[str] = Field(default="en", description="Language preference for LLM responses: 'en', 'uz', 'ru'")
    
    @validator('message')
    def validate_message(cls, v):
        if not v.strip():
            raise ValueError('Message cannot be empty')
        return v.strip()
    
    @validator('tone')
    def validate_tone(cls, v):
        if v:
            allowed_tones = ['Professional', 'Casual', 'Technical', 'Friendly']
            if v not in allowed_tones:
                return 'Professional'
        return v

class ChatResponse(BaseModel):
    response: str
    error: Optional[str] = None
    sources: Optional[List[dict]] = None

# System instruction for RubAI
SYSTEM_INSTRUCTION = """You are RubAI, a sophisticated and determined AI solutions assistant designed to provide precise, high-context answers. 
Your design aesthetic is dark, modern, and minimalist. 
When answering coding questions, provide clean, commented code blocks. 
You are capable of analyzing uploaded files (PDFs, Images, Text) and answering questions about them.

CRITICAL for document analysis:
- When given document content, DIRECTLY present the information - don't meta-comment about it
- Instead of "this page contains a section about X", say "Here's what it covers: [actual content]"
- Share specific data, quotes, statistics, and findings from the documents
- Be informative and detailed, not vague or descriptive about what's there

Maintain a helpful, professional, yet slightly futuristic tone suited for an AI Automation Agency."""

# Initialize LLM Provider based on configuration
try:
    llm_provider = create_llm_provider(
        provider_type=LLM_PROVIDER,
        system_instruction=SYSTEM_INSTRUCTION
    )
    logger.info(f"LLM Provider initialized: {LLM_PROVIDER}")
except Exception as e:
    logger.error(f"Failed to initialize LLM provider: {e}")
    raise ValueError(f"LLM provider initialization failed: {str(e)}")

# ==================== HELPER FUNCTIONS ====================

def sanitize_filename(filename: str) -> str:
    """Sanitize filename to prevent path traversal attacks"""
    # Remove path components
    filename = os.path.basename(filename)
    # Remove potentially dangerous characters
    filename = re.sub(r'[^\w\s\-\.]', '', filename)
    # Limit length
    if len(filename) > 255:
        name, ext = os.path.splitext(filename)
        filename = name[:250] + ext
    return filename

def validate_file(file: UploadFile, file_bytes: bytes) -> None:
    """Validate uploaded file"""
    # Check file size
    if len(file_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Maximum size is {MAX_FILE_SIZE / 1024 / 1024}MB"
        )
    
    # Check file extension
    filename: str = file.filename if file.filename else "unknown"
    file_ext = Path(filename).suffix.lower()
    if file_ext not in ALLOWED_FILE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"File type not allowed. Allowed types: {', '.join(ALLOWED_FILE_TYPES)}"
        )
    
    # Check MIME type
    if file.content_type not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"MIME type not allowed. File type: {file.content_type}"
        )
    
    # Check if file is empty
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="File is empty")

# ==================== ENDPOINTS ====================

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "RubAI Backend API is running",
        "status": "healthy",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    """Health check with database status"""
    db_healthy = check_db_health()
    return {
        "status": "healthy" if db_healthy else "degraded",
        "service": "RubAI Backend",
        "database": "connected" if db_healthy else "disconnected"
    }

# ==================== AUTH ENDPOINTS ====================

@app.post("/auth/google", response_model=TokenResponse)
async def google_auth(request: GoogleAuthRequest, db: Session = Depends(get_db)):
    """
    Authenticate with Google OAuth
    Receives Google ID token from frontend, verifies it, creates/updates user
    """
    try:
        # Verify Google token
        google_user = await verify_google_token(request.credential)
        
        if not google_user:
            raise HTTPException(
                status_code=401,
                detail="Invalid Google credentials"
            )
        
        # Check if user exists by Google ID
        user = get_user_by_google_id(db, google_user["google_id"])
        
        if not user:
            # Check by email (might have signed up differently before)
            user = get_user_by_email(db, google_user["email"])
            
            if user:
                # Link Google account to existing user
                user.google_id = google_user["google_id"]
                avatar = google_user.get("avatar_url")
                if avatar:
                    user.avatar_url = avatar
                db.commit()
            else:
                # Create new user
                user = create_user(
                    db,
                    email=google_user["email"],
                    name=google_user.get("name"),
                    avatar_url=google_user.get("avatar_url"),
                    google_id=google_user["google_id"]
                )
        else:
            # Update user info if changed
            update_user(
                db, user,
                name=google_user.get("name"),
                avatar_url=google_user.get("avatar_url")
            )
        
        # Create JWT token with role
        access_token = create_access_token(str(user.id), str(user.email), user.role)
        
        logger.info(f"User authenticated: {user.email} (role: {user.role})")
        
        return TokenResponse(
            access_token=access_token,
            user=user_to_dict(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Google auth error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@app.get("/auth/me")
async def get_current_user_info(current_user: User = Depends(get_current_user)):
    """
    Get current authenticated user info
    """
    return user_to_dict(current_user)


@app.post("/auth/logout")
async def logout():
    """
    Logout endpoint (client should clear token)
    """
    return {"message": "Logged out successfully"}


@app.post("/auth/admin/login", response_model=TokenResponse)
async def admin_login(request: AdminLoginRequest, db: Session = Depends(get_db)):
    """
    Admin login endpoint
    Authenticate admin users with username and password
    """
    try:
        # Find user by username
        user = get_user_by_username(db, request.username)
        
        if not user:
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Check if user is admin
        if user.role != 'admin':
            raise HTTPException(
                status_code=403,
                detail="Access denied. Admin privileges required."
            )
        
        # Verify password
        if not user.password_hash:
            raise HTTPException(
                status_code=401,
                detail="Password not set for this admin account"
            )
        
        if not verify_password(request.password, user.password_hash):
            raise HTTPException(
                status_code=401,
                detail="Invalid username or password"
            )
        
        # Create JWT token with admin role
        access_token = create_access_token(str(user.id), str(user.email), user.role)
        
        logger.info(f"Admin authenticated: {user.username} ({user.email})")
        
        return TokenResponse(
            access_token=access_token,
            user=user_to_dict(user)
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Admin login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@app.post("/auth/login", response_model=TokenResponse)
async def unified_login(request: UnifiedLoginRequest, db: Session = Depends(get_db)):
    """
    Unified login endpoint that auto-detects admin vs regular users.
    - If email/username is an admin: authenticates with password hash and returns admin token
    - If not admin: returns error indicating to use Supabase auth (frontend handles this)
    This keeps the role separation hidden from users.
    """
    try:
        # Check if user exists by email or username
        user = get_user_by_email_or_username(db, request.email_or_username)
        
        # If user exists and is admin, authenticate as admin
        if user and user.role == 'admin':
            # Verify password
            if not user.password_hash:
                raise HTTPException(
                    status_code=401,
                    detail="Invalid email or password"
                )
            
            if not verify_password(request.password, user.password_hash):
                raise HTTPException(
                    status_code=401,
                    detail="Invalid email or password"
                )
            
            # Create admin JWT token
            access_token = create_access_token(str(user.id), str(user.email), user.role)
            
            logger.info(f"Admin authenticated via unified login: {user.username or user.email}")
            
            return TokenResponse(
                access_token=access_token,
                user=user_to_dict(user)
            )
        
        # If not admin, return error that frontend will catch and use Supabase auth
        # We use a generic error message so users don't know about role separation
        raise HTTPException(
            status_code=401,
            detail="Invalid email or password"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unified login error: {e}")
        raise HTTPException(status_code=500, detail="Authentication failed")


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatMessage):
    """
    Send a message to LLM and get a response (non-streaming)
    """
    try:
        logger.info(f"Chat request received: {request.message[:50]}...")
        
        # Build prompt with tone instruction
        tone_instruction = f"[System: Respond in a {request.tone} tone] "
        prompt = tone_instruction + request.message
        
        # Note: Attachment handling may need provider-specific implementation
        # For now, we'll include attachment info in the prompt if present
        if request.attachment:
            prompt = f"[Attachment: {request.attachment.name} ({request.attachment.type})]\n\n{prompt}"
        
        # Generate response using LLM provider
        response_text = llm_provider.generate_content(prompt)
        
        logger.info("Chat response generated successfully")
        return ChatResponse(response=response_text)
    
    except Exception as e:
        logger.error(f"Chat error: {e}")
        raise HTTPException(status_code=500, detail="Failed to generate response. Please try again.")

@app.post("/chat/stream")
async def chat_stream(request: ChatMessage):
    """
    Send a message to LLM and stream the response
    """
    async def generate():
        try:
            logger.info(f"Stream chat request: {request.message[:50]}...")
            
            # Build prompt with tone instruction
            tone_instruction = f"[System: Respond in a {request.tone} tone] "
            prompt = tone_instruction + request.message
            
            # Note: Attachment handling may need provider-specific implementation
            if request.attachment:
                prompt = f"[Attachment: {request.attachment.name} ({request.attachment.type})]\n\n{prompt}"
            
            # Stream response using LLM provider
            for chunk in llm_provider.generate_content_stream(prompt):
                yield f"data: {json.dumps({'text': chunk})}\n\n"
            
            # Send done signal
            yield f"data: {json.dumps({'done': True})}\n\n"
            logger.info("Stream completed successfully")
            
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': 'Failed to generate response'})}\n\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")

# ==================== RAG ENDPOINTS ====================

@app.post("/documents/upload")
async def upload_document(
    file: UploadFile = File(...),
    category: str = Query("privacy-policies", description="Document category: privacy-policies, cvs, terms-and-conditions, ai-docs"),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Upload and process a document (PDF or TXT).
    If authenticated, document is associated with the user.
    Original file is stored in Supabase Storage organized by category.
    """
    import time
    start_time = time.time()
    
    try:
        from supabase_storage import upload_file_to_storage, validate_category, VALID_CATEGORIES
        
        logger.info(f"Upload request for file: {file.filename} (user: {user_id}, category: {category})")
        
        # Validate and normalize category
        validated_category = validate_category(category)
        if validated_category != category:
            logger.info(f"Category normalized: {category} -> {validated_category}")
        category = validated_category
        
        # Read file
        file_bytes = await file.read()
        read_time = time.time() - start_time
        
        # Validate file
        validate_file(file, file_bytes)
        
        # Sanitize filename
        filename = sanitize_filename(file.filename or "document.pdf")
        
        # Check document upload limit for authenticated users (skip for admin)
        if user_id and user_id != "admin":
            is_allowed, error_msg, current_count = check_document_upload_limit(db, user_id, upload_limit=3, authorization=authorization)
            if not is_allowed:
                logger.warning(f"Document upload limit reached for user {user_id}: {current_count} documents")
                raise HTTPException(status_code=429, detail=error_msg or "Document upload limit reached")
        
        # Admin users store documents as global (user_id=NULL), so check for existing global documents
        # Regular users check for their own documents
        effective_user_id = None if user_id == "admin" else user_id
        if effective_user_id:
            existing = db.execute(
                text("SELECT 1 FROM documents WHERE filename = :filename AND user_id = :user_id LIMIT 1"),
                {"filename": filename, "user_id": effective_user_id}
            ).first()
        else:
            existing = db.execute(
                text("SELECT 1 FROM documents WHERE filename = :filename AND user_id IS NULL LIMIT 1"),
                {"filename": filename}
            ).first()
            
        if existing:
            logger.info(f"Document '{filename}' already exists for user {user_id}, skipping re-upload")
            return {
                "message": "Document already uploaded",
                "filename": filename,
                "chunks_stored": 0,
                "processing_time": 0.0,
                "category": category
            }
        
        # **PHASE 2**: Upload original file to Supabase Storage
        storage_upload_start = time.time()
        content_type = file.content_type or "application/octet-stream"
        upload_success, upload_msg, storage_url = upload_file_to_storage(
            file_content=file_bytes,
            filename=filename,
            category=category,
            content_type=content_type
        )
        storage_upload_time = time.time() - storage_upload_start
        
        if not upload_success or not storage_url:
            logger.warning(f"File will be stored in database but NOT in Supabase Storage: {filename} - {upload_msg}")
            storage_url = None
        else:
            logger.info(f"File uploaded to Supabase Storage: {storage_url}")
        
        # Determine file type
        file_extension = Path(filename).suffix[1:].lower()
        
        # Process document (extract text and chunk)
        process_start = time.time()
        chunks = process_document(file_bytes, filename, file_extension)
        process_time = time.time() - process_start
        
        if not chunks:
            raise HTTPException(status_code=400, detail="No content extracted from document")
        
        # **PHASE 2.1**: Enhanced metadata with category and storage_url
        metadata = {
            "original_filename": file.filename,
            "file_type": file_extension,
            "file_size": len(file_bytes),
            "mime_type": file.content_type,
            "category": category,  # NEW: Document category
            "upload_type": "knowledge_base",  # NEW: Upload type (vs attachment)
            "storage_url": storage_url  # NEW: Supabase Storage URL
        }
        
        store_start = time.time()
        
        # Use Gemini embeddings (768-dim) for consistency with existing data
        from embeddings import generate_embedding
        
        # Generate embeddings and store chunks
        stored_count = 0
        for chunk_text, chunk_idx, page_num in chunks:
            try:
                # Generate Gemini embedding (768-dim)
                embedding = generate_embedding(chunk_text)
                
                # Store in database with 768-dim embedding and storage_url
                # Admin users store documents as global (user_id=NULL)
                doc_user_id = None if user_id == "admin" else user_id
                doc = Document(
                    user_id=doc_user_id,
                    filename=filename,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    embedding=embedding,  # 768-dim Gemini embedding
                    embedding_new=None,  # Explicitly set new embedding column to NULL
                    text_id=None,  # LlamaIndex node ID (set later if needed)
                    storage_url=storage_url,  # Supabase Storage URL
                    doc_metadata=metadata
                )
                db.add(doc)
                stored_count += 1
            except Exception as e:
                logger.error(f"Error processing chunk {chunk_idx}: {e}")
                continue
        
        db.commit()
        store_time = time.time() - store_start
        
        # Increment document count after successful upload (skip for admin)
        if user_id and user_id != "admin":
            increment_document_count(db, user_id)
        
        total_time = time.time() - start_time
        
        logger.info(
            f"✅ Document uploaded: {filename}, {stored_count} chunks, category: {category} (user: {user_id}). "
            f"Timing - Read: {read_time:.2f}s, Storage: {storage_upload_time:.2f}s, Process: {process_time:.2f}s, "
            f"Store: {store_time:.2f}s, Total: {total_time:.2f}s"
        )
        
        return {
            "message": "Document uploaded successfully",
            "filename": filename,
            "chunks_stored": stored_count,
            "category": category,
            "storage_url": storage_url,
            "processing_time": round(total_time, 2)
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail="Failed to process document")

@app.get("/documents")
async def list_documents(
    category: Optional[str] = Query(None),
    language: Optional[str] = Query(None, description="Filter by language (en, ru, uz)"),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Get list of all uploaded documents for the current user.
    Shows user's own documents plus shared documents (no owner).
    Admin users see all documents.
    Optional category filter to show documents from specific folder.
    Optional language filter to show only documents in a specific language.
    """
    try:
        # Admin users see all documents (pass None to get_all_documents)
        # Also handle case where user_id is "admin" string
        effective_user_id = None if (await is_admin_user(authorization) or user_id == "admin") else user_id
        documents = get_all_documents(db, user_id=effective_user_id, category=category, language=language)
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"List documents error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{filename}")
async def remove_document(
    filename: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Delete a document:
    - Personal documents: Hard delete (removes from DB and storage permanently)
    - Global documents: Soft delete (hides from user's view only)
    
    Users can only delete their own personal documents.
    Global documents (starter pack) can only be hidden, not deleted.
    """
    try:
        # URL decode filename if needed (FastAPI should do this, but ensure it's decoded)
        import urllib.parse
        filename = urllib.parse.unquote(filename)
        
        # Sanitize filename
        filename = sanitize_filename(filename)
        
        logger.info(f"Delete request for filename: '{filename}' (user: {user_id})")
        
        # Check ownership type and user permission
        result = db.execute(
            text("SELECT ownership_type, user_id FROM documents WHERE filename = :filename LIMIT 1"),
            {"filename": filename}
        ).first()
        
        if not result:
            logger.warning(f"Document not found: '{filename}'")
            raise HTTPException(status_code=404, detail="Document not found")
        
        ownership_type = result[0]
        doc_user_id = result[1]
        
        logger.info(f"Document found: ownership_type={ownership_type}, doc_user_id={doc_user_id}, request_user_id={user_id}")
        
        # Require authentication for any delete operation
        if not user_id:
            logger.warning(f"Unauthenticated user attempted to delete document: {filename}")
            raise HTTPException(status_code=401, detail="Authentication required to delete documents")
        
        # Admin users can delete any document (bypass ownership checks)
        is_admin = (user_id == "admin")
        
        if not is_admin:
            # GLOBAL DOCUMENTS: Hide instead of delete (for regular users)
            if ownership_type == 'global':
                # Hide the document for this user
                db.execute(
                    text("""
                        INSERT INTO user_document_visibility (user_id, filename, is_hidden, hidden_at)
                        VALUES (:user_id, :filename, TRUE, NOW())
                        ON CONFLICT (user_id, filename) 
                        DO UPDATE SET is_hidden = TRUE, hidden_at = NOW()
                    """),
                    {"user_id": user_id, "filename": filename}
                )
                db.commit()
                
                logger.info(f"Global document hidden: {filename} (user: {user_id})")
                return {
                    "message": "Document hidden from your view (global documents cannot be permanently deleted)",
                    "filename": filename,
                    "hidden": True,
                    "chunks_deleted": 0,
                    "storage_deleted": False
                }
            
            # PERSONAL DOCUMENTS: Hard delete (existing logic)
            # Verify user owns the document
            # Convert both to strings for comparison (in case of UUID type mismatch)
            doc_user_id_str = str(doc_user_id) if doc_user_id else None
            user_id_str = str(user_id) if user_id else None
            
            if doc_user_id_str != user_id_str:
                logger.warning(f"Permission denied: doc_user_id={doc_user_id_str} (type: {type(doc_user_id)}), request_user_id={user_id_str} (type: {type(user_id)}), filename={filename}")
                raise HTTPException(status_code=403, detail="You don't have permission to delete this document")
            
            logger.info(f"Permission granted: user {user_id_str} owns document {filename}")
        
        # Admin users can delete any document, regular users can only delete their own
        # First, get document metadata to find category for storage deletion
        from vector_store import get_document_metadata
        doc_metadata = get_document_metadata(db, filename, user_id=user_id)
        
        if not doc_metadata:
            raise HTTPException(status_code=404, detail="Document not found or you don't have permission to delete it")
        
        # Get text_ids (node_ids) for this document BEFORE deleting (for LlamaIndex cleanup)
        node_ids = []
        try:
            # For admin, get all text_ids regardless of user_id
            if is_admin:
                result = db.execute(
                    text("SELECT text_id FROM documents WHERE filename = :filename AND text_id IS NOT NULL"),
                    {"filename": filename}
                )
            else:
                result = db.execute(
                    text("SELECT text_id FROM documents WHERE filename = :filename AND user_id = :user_id AND text_id IS NOT NULL"),
                    {"filename": filename, "user_id": user_id}
                )
            node_ids = [row[0] for row in result.fetchall() if row[0]]
        except Exception as e:
            logger.warning(f"Failed to get node_ids for LlamaIndex cleanup (non-critical): {e}")
        
        # Delete from Supabase storage if category is available
        storage_deleted = False
        if doc_metadata.get('category'):
            from supabase_storage import delete_file_from_storage
            storage_deleted, storage_message = delete_file_from_storage(filename, doc_metadata['category'])
            if not storage_deleted:
                logger.warning(f"Failed to delete from storage: {storage_message}")
        
        # Delete from database (chunks) - pass None for admin to allow deletion of any document
        effective_user_id = None if is_admin else user_id
        deleted_count = delete_document(db, filename, user_id=effective_user_id)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found or you don't have permission to delete it")
        
        # Also delete from LlamaIndex data_documents table if nodes exist
        if node_ids:
            try:
                db.execute(
                    text("DELETE FROM data_documents WHERE node_id = ANY(:node_ids)"),
                    {"node_ids": node_ids}
                )
                db.commit()
                logger.info(f"Deleted {len(node_ids)} LlamaIndex nodes for document: {filename}")
            except Exception as e:
                logger.warning(f"Failed to delete LlamaIndex nodes (non-critical): {e}")
                # Don't fail the whole operation if LlamaIndex cleanup fails
        
        # Decrement document count after successful deletion (skip for admin)
        if user_id and user_id != "admin":
            decrement_document_count(db, user_id)
        
        logger.info(f"Personal document deleted: {filename}, {deleted_count} chunks (user: {user_id}), storage: {'✓' if storage_deleted else '✗'}")
        
        return {
            "message": "Document deleted successfully",
            "filename": filename,
            "chunks_deleted": deleted_count,
            "storage_deleted": storage_deleted
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/documents/hide")
async def hide_document(
    filename: str = Query(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """
    Hide a global document from user's view (soft delete).
    Only works for global documents (starter pack).
    Personal documents should be deleted permanently using DELETE endpoint.
    """
    try:
        # Sanitize filename
        filename = sanitize_filename(filename)
        
        # Check if document exists and is global
        result = db.execute(
            text("SELECT ownership_type FROM documents WHERE filename = :filename LIMIT 1"),
            {"filename": filename}
        ).first()
        
        if not result:
            raise HTTPException(status_code=404, detail="Document not found")
        
        if result[0] != 'global':
            raise HTTPException(
                status_code=400, 
                detail="Can only hide global documents. Use DELETE endpoint for personal documents."
            )
        
        # Insert or update visibility record
        db.execute(
            text("""
                INSERT INTO user_document_visibility (user_id, filename, is_hidden, hidden_at)
                VALUES (:user_id, :filename, TRUE, NOW())
                ON CONFLICT (user_id, filename) 
                DO UPDATE SET is_hidden = TRUE, hidden_at = NOW()
            """),
            {"user_id": user_id, "filename": filename}
        )
        db.commit()
        
        logger.info(f"Document hidden: {filename} (user: {user_id})")
        return {
            "message": f"Document '{filename}' hidden from your view",
            "filename": filename,
            "is_hidden": True
        }
        
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Hide document error: {e}")
        raise HTTPException(status_code=500, detail="Failed to hide document")


@app.post("/documents/unhide")
async def unhide_document(
    filename: str = Query(...),
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """
    Restore a hidden global document to user's view.
    """
    try:
        # Sanitize filename
        filename = sanitize_filename(filename)
        
        # Update visibility record
        db.execute(
            text("""
                UPDATE user_document_visibility
                SET is_hidden = FALSE, hidden_at = NULL
                WHERE user_id = :user_id AND filename = :filename
            """),
            {"user_id": user_id, "filename": filename}
        )
        db.commit()
        
        logger.info(f"Document unhidden: {filename} (user: {user_id})")
        return {
            "message": f"Document '{filename}' restored to your view",
            "filename": filename,
            "is_hidden": False
        }
        
    except Exception as e:
        db.rollback()
        logger.error(f"Unhide document error: {e}")
        raise HTTPException(status_code=500, detail="Failed to unhide document")

@app.post("/search")
async def search_documents(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(default=5, ge=1, le=20),
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Search for similar document chunks within user's documents.
    """
    try:
        results = search_similar_chunks(db, query.strip(), top_k, user_id=user_id)
        return {"results": results, "count": len(results)}
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail="Search failed")

@app.post("/chat/rag", response_model=ChatResponse)
async def chat_with_rag(
    request: ChatMessage,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Chat with RAG using LlamaIndex with conversation memory.
    
    NEW: This endpoint now maintains conversation context!
    - No need to re-attach documents for follow-up questions
    - Conversation history is automatically included
    - Ask "tell me more" and it will understand the context
    
    Args:
        request: ChatMessage with message, chat_id, selected_documents, tone
        user_id: Extracted from Supabase JWT (optional for anonymous)
        
    Returns:
        ChatResponse with answer and source citations
    """
    try:
        logger.info(f"RAG chat with LlamaIndex: user={user_id}, chat={request.chat_id}, msg='{request.message[:50]}...'")
        
        # For anonymous users, pass None (not "anonymous") since user_id is UUID column
        # This allows the SQL query to match user_id IS NULL for public documents
        effective_user_id = user_id  # None for anonymous, UUID string for authenticated
        
        # Check message limit for authenticated users (skip for admin)
        if effective_user_id and effective_user_id != "admin":
            is_allowed, cooldown_msg, current_count = check_message_limit(db, effective_user_id, daily_limit=20, authorization=authorization)
            if not is_allowed:
                logger.warning(f"Message limit reached for user {effective_user_id}: {current_count} messages today")
                raise HTTPException(status_code=429, detail=cooldown_msg or "Daily message limit reached")
            
            # Increment message count before processing (skip for admin)
            increment_message_count(db, effective_user_id)
        
        # Initialize RAG service
        rag_service = RAGService(db)
        
        # Map language codes to full names
        language_names = {
            'en': 'English',
            'uz': 'Uzbek',
            'ru': 'Russian'
        }
        language_name = language_names.get(request.language or 'en', 'English')
        
        # Build system prompt with tone and language
        language_instruction = ""
        if request.language == 'uz':
            language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in Uzbek (O'zbek tili). All text, explanations, and responses must be in Uzbek. Do not mix languages."
        elif request.language == 'ru':
            language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in Russian (Русский язык). All text, explanations, and responses must be in Russian. Do not mix languages."
        elif request.language == 'en':
            language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in English. All text, explanations, and responses must be in English."
        
        system_prompt = f"[System: Respond in a {request.tone} tone. IMPORTANT: Please respond in {language_name} language.]{language_instruction}\n\n{rag_service._get_default_system_prompt()}"
        
        # Chat with direct retrieval (uses our vector_store search, not LlamaIndex PGVectorStore)
        # For admin users, pass "admin" to search_similar_chunks so it shows all documents
        # search_similar_chunks already handles "admin" correctly (shows all docs, respects filters)
        rag_user_id = effective_user_id  # Keep "admin" as-is for search_similar_chunks
        response_text, sources = rag_service.chat_direct(
            user_id=rag_user_id,
            chat_id=request.chat_id,
            message=request.message,
            selected_documents=request.selected_documents,
            category=request.category,
            knowledge_base_mode=request.knowledge_base_mode or "none",
            system_prompt=system_prompt,
            language=request.language
        )
        
        logger.info(f"✅ RAG response generated with direct retrieval and {len(sources)} sources")
        
        # **PHASE 2.3**: Sources are now extracted from LlamaIndex response
        # Only display sources if they exist and have metadata
        
        # Log source URLs for debugging citation links
        for i, source in enumerate(sources):
            logger.debug(f"Source {i+1}: {source.get('filename')} | Page: {source.get('page_number')} | URL: {source.get('storage_url', 'MISSING')}")
        
        # PHASE 3: Synchronize with chat_sessions after response
        # Convert admin user_id to UUID for chat_sessions table
        try:
            if effective_user_id:
                sync_user_id = get_admin_chat_user_id(effective_user_id) if effective_user_id == "admin" else effective_user_id
                await sync_chat_session(
                    db=db,
                    user_id=sync_user_id,
                    chat_id=request.chat_id,
                    user_message=request.message,
                    ai_response=response_text,
                    sources=sources if sources else None,
                    tone=request.tone
                )
                logger.info("✅ Chat session synchronized")
        except Exception as e:
            logger.error(f"Error synchronizing chat session: {e}")
            # Don't fail the whole request, just log the error
        
        return ChatResponse(
            response=response_text,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"RAG chat error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to generate RAG response")


# ==================== DOCUMENT UPLOAD LIMIT MANAGEMENT ====================

def check_document_upload_limit(db: Session, user_id: str, upload_limit: int = 3, authorization: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Check if user has reached document upload limit.
    Admin users bypass the limit.
    
    Args:
        db: Database session
        user_id: User identifier (Supabase UUID)
        upload_limit: Maximum documents per user (default: 3)
        authorization: Optional Authorization header to check for admin role
        
    Returns:
        tuple: (is_allowed, error_message, current_count)
        - is_allowed: True if user can upload document, False if limit reached
        - error_message: Message to display if limit reached
        - current_count: Current document count
    """
    if not user_id:
        # Anonymous users: no limit (or implement separate logic if needed)
        return True, None, None
    
    # Check if user is admin (from token or user_id marker)
    if user_id == "admin":
        logger.info(f"Admin user bypassing document upload limit")
        return True, None, None
    
    # Check if user is admin from JWT token
    if authorization:
        try:
            from auth import verify_token
            token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
            payload = verify_token(token)
            if payload and payload.get("role") == "admin":
                logger.info(f"Admin user bypassing document upload limit")
                return True, None, None
        except Exception as e:
            logger.debug(f"Could not check admin status from token: {e}")
    
    try:
        # Get current document count
        result = db.execute(
            text("""
                SELECT document_count 
                FROM user_document_counts 
                WHERE user_id = :user_id
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()
        current_count = row[0] if row else 0
        
        # Check if limit reached
        if current_count >= upload_limit:
            error_msg = f"You've reached your document upload limit of {upload_limit} documents. Please delete some documents before uploading new ones."
            return False, error_msg, current_count
        
        return True, None, current_count
        
    except Exception as e:
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_document_counts table does not exist. Please run the migration: backend/migrations/add_document_upload_limit.sql")
            logger.warning(f"   Document uploads will work but won't be counted until migration is applied.")
        else:
            logger.error(f"Error checking document upload limit: {e}")
        # On error, allow the upload (fail open) - don't block users if table doesn't exist
        return True, None, None


def increment_document_count(db: Session, user_id: str) -> int:
    """
    Increment document count for user.
    
    Args:
        db: Database session
        user_id: User identifier
        
    Returns:
        int: New document count after increment
    """
    if not user_id:
        return 0
    
    try:
        # Use PostgreSQL function to increment count atomically
        result = db.execute(
            text("SELECT increment_document_count(:user_id)"),
            {"user_id": user_id}
        )
        new_count = result.scalar()
        db.commit()
        logger.info(f"Document count incremented for user {user_id}: {new_count} documents")
        return new_count or 0
    except Exception as e:
        db.rollback()
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_document_counts table does not exist. Document counting disabled until migration is run.")
            return 0
        
        logger.error(f"Error incrementing document count: {e}")
        # Try fallback: direct insert/update
        try:
            result = db.execute(
                text("""
                    INSERT INTO user_document_counts (user_id, document_count, last_upload_at)
                    VALUES (:user_id, 1, NOW())
                    ON CONFLICT (user_id)
                    DO UPDATE SET 
                        document_count = user_document_counts.document_count + 1,
                        last_upload_at = NOW(),
                        updated_at = NOW()
                    RETURNING document_count
                """),
                {"user_id": user_id}
            )
            new_count = result.scalar()
            db.commit()
            return new_count or 0
        except Exception as e2:
            db.rollback()
            error_str2 = str(e2).lower()
            if "does not exist" in error_str2 or "relation" in error_str2 or "undefinedtable" in error_str2:
                logger.warning(f"⚠️ user_document_counts table does not exist. Document counting disabled.")
            else:
                logger.error(f"Fallback increment also failed: {e2}")
            return 0


def decrement_document_count(db: Session, user_id: str) -> int:
    """
    Decrement document count for user (when document is deleted).
    
    Args:
        db: Database session
        user_id: User identifier
        
    Returns:
        int: New document count after decrement
    """
    if not user_id:
        return 0
    
    try:
        # Use PostgreSQL function to decrement count atomically
        result = db.execute(
            text("SELECT decrement_document_count(:user_id)"),
            {"user_id": user_id}
        )
        new_count = result.scalar()
        db.commit()
        logger.info(f"Document count decremented for user {user_id}: {new_count} documents")
        return new_count or 0
    except Exception as e:
        db.rollback()
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_document_counts table does not exist. Document counting disabled.")
            return 0
        
        logger.error(f"Error decrementing document count: {e}")
        # Try fallback: direct update
        try:
            result = db.execute(
                text("""
                    UPDATE user_document_counts
                    SET document_count = GREATEST(document_count - 1, 0),
                        updated_at = NOW()
                    WHERE user_id = :user_id
                    RETURNING document_count
                """),
                {"user_id": user_id}
            )
            row = result.fetchone()
            db.commit()
            return row[0] if row else 0
        except Exception as e2:
            db.rollback()
            error_str2 = str(e2).lower()
            if "does not exist" in error_str2 or "relation" in error_str2 or "undefinedtable" in error_str2:
                logger.warning(f"⚠️ user_document_counts table does not exist. Document counting disabled.")
            else:
                logger.error(f"Fallback decrement also failed: {e2}")
            return 0


# ==================== MESSAGE LIMIT MANAGEMENT ====================

def check_message_limit(db: Session, user_id: str, daily_limit: int = 20, authorization: Optional[str] = None) -> Tuple[bool, Optional[str], Optional[int]]:
    """
    Check if user has reached daily message limit.
    Admin users bypass the limit.
    
    Args:
        db: Database session
        user_id: User identifier (Supabase UUID)
        daily_limit: Maximum messages per day (default: 20)
        authorization: Optional Authorization header to check for admin role
        
    Returns:
        tuple: (is_allowed, cooldown_message, current_count)
        - is_allowed: True if user can send message, False if limit reached
        - cooldown_message: Message to display if limit reached (includes next reset time)
        - current_count: Current message count for today
    """
    if not user_id:
        # Anonymous users: no limit (or implement separate logic if needed)
        return True, None, None
    
    # Check if user is admin (from token or user_id marker)
    if user_id == "admin":
        logger.info(f"Admin user bypassing message limit")
        return True, None, None
    
    # Check if user is admin from JWT token
    if authorization:
        try:
            from auth import verify_token
            token = authorization.replace("Bearer ", "") if authorization.startswith("Bearer ") else authorization
            payload = verify_token(token)
            if payload and payload.get("role") == "admin":
                logger.info(f"Admin user bypassing message limit")
                return True, None, None
        except Exception as e:
            logger.debug(f"Could not check admin status from token: {e}")
    
    try:
        from datetime import datetime, timedelta, timezone
        
        # Get current message count for today
        result = db.execute(
            text("""
                SELECT message_count 
                FROM user_message_counts 
                WHERE user_id = :user_id AND date = CURRENT_DATE
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()
        current_count = row[0] if row else 0
        
        # Check if limit reached
        if current_count >= daily_limit:
            # Calculate next reset time (12:00 AM tomorrow)
            now = datetime.now(timezone.utc)
            tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
            
            # Format date for display
            reset_date_str = tomorrow.strftime("%B %d, %Y")
            reset_time_str = tomorrow.strftime("%I:%M %p")
            
            cooldown_msg = f"Cool down! You've reached your daily limit of {daily_limit} messages. Please come back after 12:00 AM ({reset_date_str})."
            return False, cooldown_msg, current_count
        
        return True, None, current_count
        
    except Exception as e:
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_message_counts table does not exist. Please run the migration: backend/migrations/add_message_limit.sql")
            logger.warning(f"   Messages will work but won't be counted until migration is applied.")
        else:
            logger.error(f"Error checking message limit: {e}")
        # On error, allow the message (fail open) - don't block users if table doesn't exist
        return True, None, None


def increment_message_count(db: Session, user_id: str) -> int:
    """
    Increment message count for user for today.
    
    Args:
        db: Database session
        user_id: User identifier
        
    Returns:
        int: New message count after increment
    """
    if not user_id:
        return 0
    
    try:
        # Use PostgreSQL function to increment count atomically
        result = db.execute(
            text("SELECT increment_message_count(:user_id)"),
            {"user_id": user_id}
        )
        new_count = result.scalar()
        db.commit()
        logger.info(f"Message count incremented for user {user_id}: {new_count} messages today")
        return new_count or 0
    except Exception as e:
        db.rollback()
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_message_counts table does not exist. Message counting disabled until migration is run.")
            return 0
        
        logger.error(f"Error incrementing message count: {e}")
        # Try fallback: direct insert/update
        try:
            result = db.execute(
                text("""
                    INSERT INTO user_message_counts (user_id, date, message_count, last_message_at)
                    VALUES (:user_id, CURRENT_DATE, 1, NOW())
                    ON CONFLICT (user_id, date)
                    DO UPDATE SET 
                        message_count = user_message_counts.message_count + 1,
                        last_message_at = NOW(),
                        updated_at = NOW()
                    RETURNING message_count
                """),
                {"user_id": user_id}
            )
            new_count = result.scalar()
            db.commit()
            return new_count or 0
        except Exception as e2:
            db.rollback()
            error_str2 = str(e2).lower()
            if "does not exist" in error_str2 or "relation" in error_str2 or "undefinedtable" in error_str2:
                logger.warning(f"⚠️ user_message_counts table does not exist. Message counting disabled.")
            else:
                logger.error(f"Fallback increment also failed: {e2}")
            return 0


# ==================== PHASE 3: CHAT SESSION SYNCHRONIZATION ====================

async def sync_chat_session(
    db: Session,
    user_id: str,
    chat_id: str,
    user_message: str,
    ai_response: str,
    attachment: Optional[dict] = None,
    sources: Optional[List[dict]] = None,
    tone: Optional[str] = None
):
    """
    Synchronize chat conversation to chat_sessions table for UI persistence.
    
    This keeps chat_sessions in sync with llamaindex_chat_store after each exchange.
    - llamaindex_chat_store = RAG memory (LlamaIndex uses this)
    - chat_sessions = UI display (Frontend uses this)
    
    Args:
        db: Database session
        user_id: User identifier (skips sync if "admin" due to UUID type mismatch)
        chat_id: Chat session ID
        user_message: The user's message
        ai_response: The AI's response
        attachment: Optional attachment data
        sources: Optional source citations
        tone: Optional tone used
    """
    # Convert admin user_id to UUID for chat_sessions table
    # Admin users use a special UUID to store chat sessions
    effective_user_id = get_admin_chat_user_id(user_id) if user_id == "admin" else user_id
    
    try:
        import time
        
        # Build user message object
        user_msg = {
            "id": f"user_{int(time.time() * 1000)}",
            "role": "user",
            "content": user_message,
            "timestamp": int(time.time() * 1000)
        }
        
        if attachment:
            user_msg["attachment"] = attachment
        
        if tone:
            user_msg["tone"] = tone
        
        # Build AI message object
        ai_msg = {
            "id": f"model_{int(time.time() * 1000) + 1}",
            "role": "model",
            "content": ai_response,
            "timestamp": int(time.time() * 1000) + 1
        }
        
        if sources and len(sources) > 0:
            ai_msg["sources"] = sources
        
        # Check if session exists (use effective_user_id for admin UUID conversion)
        result = db.execute(
            text("SELECT messages FROM chat_sessions WHERE id = :chat_id AND user_id = :user_id"),
            {"chat_id": chat_id, "user_id": effective_user_id}
        )
        row = result.fetchone()
        
        if row:
            # Session exists - append messages
            existing_messages = row[0] if row[0] else []
            existing_messages.append(user_msg)
            existing_messages.append(ai_msg)
            
            db.execute(
                text("""
                    UPDATE chat_sessions 
                    SET messages = :messages, updated_at = now()
                    WHERE id = :chat_id AND user_id = :user_id
                """),
                {
                    "messages": json.dumps(existing_messages),
                    "chat_id": chat_id,
                    "user_id": effective_user_id
                }
            )
            logger.info(f"Updated existing chat session {chat_id} with {len(existing_messages)} messages")
        else:
            # Session doesn't exist - create new
            messages = [user_msg, ai_msg]
            
            db.execute(
                text("""
                    INSERT INTO chat_sessions (id, user_id, title, messages)
                    VALUES (:chat_id, :user_id, :title, :messages)
                """),
                {
                    "chat_id": chat_id,
                    "user_id": effective_user_id,
                    "title": user_message[:50] if len(user_message) > 50 else user_message,
                    "messages": json.dumps(messages)
                }
            )
            logger.info(f"Created new chat session {chat_id}")
        
        db.commit()
        
    except Exception as e:
        db.rollback()
        logger.error(f"Failed to sync chat session: {e}")
        raise


# ==================== RAG STREAMING ENDPOINT ====================

@app.post("/chat/rag/stream")
async def chat_with_rag_stream(
    request: ChatMessage,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Chat with RAG using LlamaIndex with conversation memory (streaming version).
    
    PHASE 3: Dual storage synchronization
    - llamaindex_chat_store: RAG memory (conversation context for LLM)
    - chat_sessions: UI persistence (full messages for frontend display)
    - Both are synchronized after each message exchange
    """
    async def generate():
        full_response = ""
        sources_list = []
        
        try:
            logger.info(f"RAG stream with LlamaIndex: user={user_id}, chat={request.chat_id}, msg='{request.message[:50]}...'")
            
            # For anonymous users, pass None since user_id is UUID column
            effective_user_id = user_id  # None for anonymous, UUID string for authenticated
            
            # Check message limit for authenticated users (skip for admin)
            if effective_user_id and effective_user_id != "admin":
                is_allowed, cooldown_msg, current_count = check_message_limit(db, effective_user_id, daily_limit=20, authorization=authorization)
                if not is_allowed:
                    logger.warning(f"Message limit reached for user {effective_user_id}: {current_count} messages today")
                    yield f"data: {json.dumps({'error': cooldown_msg or 'Daily message limit reached'})}\n\n"
                    return
                
                # Increment message count before processing (skip for admin)
                increment_message_count(db, effective_user_id)
            
            # Initialize RAG service
            rag_service = RAGService(db)
            
            # Process attachment if present
            attachment_data = None
            if request.attachment:
                logger.info(f"Processing attachment: {request.attachment.name}")
                try:
                    from document_processor import process_base64_attachment
                    
                    # Extract attachment data
                    base64_data = request.attachment.data
                    filename = request.attachment.name
                    
                    # Process attachment into chunks
                    chunks = process_base64_attachment(base64_data, filename)
                    
                    # Add chunks to vector store (skip for admin)
                    if effective_user_id and effective_user_id != "admin":
                        rag_service.add_temporary_documents(
                            chunks=chunks,
                            filename=filename,
                            user_id=effective_user_id,
                            chat_id=request.chat_id
                        )
                    
                    logger.info(f"✅ Attachment processed: {len(chunks)} chunks added to vector store")
                    
                    # Store attachment data for chat_sessions
                    attachment_data = {
                        "name": request.attachment.name,
                        "type": request.attachment.type
                    }
                    
                except Exception as e:
                    logger.error(f"Error processing attachment: {e}")
                    # Continue without attachment rather than failing completely
                    yield f"data: {json.dumps({'text': '[Note: Could not process attachment] '})}\n\n"
            
            # Map language codes to full names
            language_names = {
                'en': 'English',
                'uz': 'Uzbek',
                'ru': 'Russian'
            }
            language_name = language_names.get(request.language, 'English')
            
            # Build system prompt with tone and language
            language_instruction = ""
            if request.language == 'uz':
                language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in Uzbek (O'zbek tili). All text, explanations, and responses must be in Uzbek. Do not mix languages."
            elif request.language == 'ru':
                language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in Russian (Русский язык). All text, explanations, and responses must be in Russian. Do not mix languages."
            elif request.language == 'en':
                language_instruction = f"\n\nCRITICAL LANGUAGE REQUIREMENT: You MUST respond entirely in English. All text, explanations, and responses must be in English."
            
            system_prompt = f"[System: Respond in a {request.tone} tone. IMPORTANT: Please respond in {language_name} language.]{language_instruction}\n\n{rag_service._get_default_system_prompt()}"
            
            # Stream response with direct retrieval (uses our vector_store, not LlamaIndex PGVectorStore)
            # For admin users, pass "admin" to search_similar_chunks so it shows all documents
            # search_similar_chunks already handles "admin" correctly (shows all docs, respects filters)
            rag_user_id = effective_user_id  # Keep "admin" as-is for search_similar_chunks
            token_count = 0
            for data in rag_service.chat_stream_direct(
                user_id=rag_user_id,
                chat_id=request.chat_id,
                message=request.message,
                selected_documents=request.selected_documents,
                category=request.category,
                knowledge_base_mode=request.knowledge_base_mode or "none",
                system_prompt=system_prompt
            ):
                # Handle token streaming
                if "token" in data:
                    token_count += 1
                    full_response += data["token"]
                    payload = json.dumps({'text': data["token"]})
                    logger.debug(f"Sending token {token_count}: {payload}")
                    yield f"data: {payload}\n\n"
                
                # Handle source citations
                elif "sources" in data:
                    sources_list = data['sources']
                    logger.info(f"Sending {len(sources_list)} source citations")
                    payload = json.dumps({'sources': sources_list})
                    yield f"data: {payload}\n\n"
            
            # PHASE 3: Synchronize with chat_sessions after streaming completes
            # Convert admin user_id to UUID for chat_sessions table
            try:
                if effective_user_id:
                    sync_user_id = get_admin_chat_user_id(effective_user_id) if effective_user_id == "admin" else effective_user_id
                    await sync_chat_session(
                        db=db,
                        user_id=sync_user_id,
                        chat_id=request.chat_id,
                        user_message=request.message,
                        ai_response=full_response,
                        attachment=attachment_data,
                        sources=sources_list,
                        tone=request.tone
                    )
                    logger.info("✅ Chat session synchronized")
            except Exception as e:
                logger.error(f"Error synchronizing chat session: {e}")
                # Don't fail the whole request, just log the error
            
            # Quality validation: Check for abnormally short responses
            if token_count < 5 and not sources_list:
                logger.error(f"⚠️ Abnormally short response detected: {token_count} tokens, response='{full_response[:200]}'")
                logger.error(f"   Message was: '{request.message}', Tone: {request.tone}")
            
            # Send done signal
            yield f"data: {json.dumps({'done': True})}\\n\\n"
            logger.info(f"✅ RAG stream completed with conversation memory - {token_count} tokens sent")
            
        except Exception as e:
            logger.error(f"RAG stream error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            yield f"data: {json.dumps({'error': 'Failed to generate RAG response'})}\\n\\n"
    
    return StreamingResponse(generate(), media_type="text/event-stream")


# ==================== DOCUMENT COUNT ENDPOINT ====================

@app.get("/user/document-count")
async def get_document_count(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Get current document upload count and limit.
    Admin users have no limit (returns limit: null).
    
    Returns:
        - count: Current document count
        - limit: Document upload limit (3 for regular users, null for admin)
        - remaining: Remaining documents that can be uploaded
    """
    # Check if user is admin
    if await is_admin_user(authorization):
        logger.info("Admin user - returning unlimited document count")
        return {
            "count": 0,
            "limit": None,
            "remaining": None
        }
    
    if not user_id or user_id == "admin":
        # Anonymous users or admin (already handled above) have no limit
        return {
            "count": 0,
            "limit": None,
            "remaining": None
        }
    
    try:
        # Get current document count
        result = db.execute(
            text("""
                SELECT document_count 
                FROM user_document_counts 
                WHERE user_id = :user_id
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()
        current_count = row[0] if row else 0
        
        upload_limit = 3
        remaining = max(0, upload_limit - current_count)
        
        return {
            "count": current_count,
            "limit": upload_limit,
            "remaining": remaining
        }
    except Exception as e:
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_document_counts table does not exist. Please run the migration: backend/migrations/add_document_upload_limit.sql")
            # Return safe defaults - table doesn't exist yet
            return {
                "count": 0,
                "limit": 3,
                "remaining": 3
            }
        else:
            logger.error(f"Error getting document count: {e}")
            # Return safe defaults on error
            return {
                "count": 0,
                "limit": 3,
                "remaining": 3
            }


# ==================== MESSAGE COUNT ENDPOINT ====================

@app.get("/user/message-count")
async def get_message_count(
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id),
    authorization: Optional[str] = Header(None)
):
    """
    Get current message count for today and daily limit.
    Admin users have no limit (returns limit: null).
    
    Returns:
        - count: Current message count for today
        - limit: Daily message limit (20 for regular users, null for admin)
        - remaining: Remaining messages for today
        - reset_time: When the count resets (12:00 AM tomorrow)
    """
    # Check if user is admin
    if await is_admin_user(authorization):
        logger.info("Admin user - returning unlimited message count")
        return {
            "count": 0,
            "limit": None,
            "remaining": None,
            "reset_time": None
        }
    
    if not user_id or user_id == "admin":
        # Anonymous users or admin (already handled above) have no limit
        return {
            "count": 0,
            "limit": None,
            "remaining": None,
            "reset_time": None
        }
    
    try:
        from datetime import datetime, timedelta, timezone
        
        # Get current message count for today
        result = db.execute(
            text("""
                SELECT message_count 
                FROM user_message_counts 
                WHERE user_id = :user_id AND date = CURRENT_DATE
            """),
            {"user_id": user_id}
        )
        row = result.fetchone()
        current_count = row[0] if row else 0
        
        # Calculate reset time (12:00 AM tomorrow)
        now = datetime.now(timezone.utc)
        tomorrow = (now + timedelta(days=1)).replace(hour=0, minute=0, second=0, microsecond=0)
        
        daily_limit = 20
        remaining = max(0, daily_limit - current_count)
        
        return {
            "count": current_count,
            "limit": daily_limit,
            "remaining": remaining,
            "reset_time": tomorrow.isoformat()
        }
    except Exception as e:
        # Check if it's a table doesn't exist error
        error_str = str(e).lower()
        if "does not exist" in error_str or "relation" in error_str or "undefinedtable" in error_str:
            logger.warning(f"⚠️ user_message_counts table does not exist. Please run the migration: backend/migrations/add_message_limit.sql")
            # Return safe defaults - table doesn't exist yet
            return {
                "count": 0,
                "limit": 20,
                "remaining": 20,
                "reset_time": None
            }
        else:
            logger.error(f"Error getting message count: {e}")
            # Return safe defaults on error
            return {
                "count": 0,
                "limit": 20,
                "remaining": 20,
                "reset_time": None
            }


# ==================== CHAT HISTORY MANAGEMENT (LlamaIndex) ====================

@app.get("/chat/{chat_id}/history")
async def get_chat_history(
    chat_id: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Get conversation history for a chat session using LlamaIndex chat store.
    
    Returns:
        - history: List of messages with role, content, and timestamp
        - count: Total number of messages
    """
    try:
        # For chat history, we need an identifier
        # Admin users use "admin" as identifier for LlamaIndex memory
        effective_user_id = user_id if user_id else "anonymous"
        
        # Initialize RAG service
        rag_service = RAGService(db)
        
        # Get chat history
        history = rag_service.get_chat_history(effective_user_id, chat_id)
        
        logger.info(f"Retrieved {len(history)} messages for chat {chat_id}")
        
        return {
            "history": history,
            "count": len(history)
        }
    except Exception as e:
        logger.error(f"Error fetching chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to fetch chat history")


@app.delete("/chat/{chat_id}/history")
async def delete_chat_history(
    chat_id: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Delete conversation history for a chat session.
    
    This clears all messages from the LlamaIndex chat store for the specified chat.
    """
    try:
        # For chat history deletion
        # Admin users use "admin" as identifier for LlamaIndex memory
        effective_user_id = user_id if user_id else "anonymous"
        
        # Initialize RAG service
        rag_service = RAGService(db)
        
        # Delete chat history
        rag_service.delete_chat_history(effective_user_id, chat_id)
        
        logger.info(f"Deleted chat history for chat {chat_id}")
        
        return {"message": "Chat history deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting chat history: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat history")


# ==================== CHAT SESSION ENDPOINTS ====================

class ChatSession(BaseModel):
    """Chat session model"""
    id: Optional[str] = None
    user_id: str
    title: str = "New Chat"
    messages: List[dict] = Field(default_factory=list)
    selected_documents: List[str] = Field(default_factory=list)
    created_at: Optional[str] = None
    updated_at: Optional[str] = None

class CreateSessionRequest(BaseModel):
    """Request to create a new chat session"""
    title: str = "New Chat"
    messages: List[dict] = Field(default_factory=list)
    selected_documents: List[str] = Field(default_factory=list)

class UpdateSessionRequest(BaseModel):
    """Request to update an existing chat session"""
    title: Optional[str] = None
    messages: Optional[List[dict]] = None
    selected_documents: Optional[List[str]] = None


@app.get("/chat/sessions")
async def list_chat_sessions(
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """Get all chat sessions for the current user"""
    try:
        # Convert admin user_id to UUID for chat_sessions table
        effective_user_id = get_admin_chat_user_id(user_id) if user_id == "admin" else user_id
        logger.info(f"Listing chat sessions for user_id={user_id} (effective_user_id={effective_user_id})")
        
        result = db.execute(
            text("""
                SELECT id, title, messages, selected_documents, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = :user_id
                ORDER BY updated_at DESC
            """),
            {"user_id": effective_user_id}
        )
        
        sessions = []
        for row in result:
            sessions.append({
                "id": str(row[0]),
                "title": row[1],
                "messages": row[2],
                "selected_documents": row[3] or [],
                "created_at": str(row[4]),
                "updated_at": str(row[5])
            })
        
        logger.info(f"Found {len(sessions)} chat sessions for user_id={user_id}")
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve chat sessions")


@app.post("/chat/sessions")
async def create_chat_session(
    request: CreateSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """Create a new chat session"""
    try:
        # Convert admin user_id to UUID for chat_sessions table
        effective_user_id = get_admin_chat_user_id(user_id) if user_id == "admin" else user_id
        
        result = db.execute(
            text("""
                INSERT INTO chat_sessions (user_id, title, messages, selected_documents)
                VALUES (:user_id, :title, :messages, :selected_documents)
                RETURNING id, title, messages, selected_documents, created_at, updated_at
            """),
            {
                "user_id": effective_user_id,
                "title": request.title,
                "messages": json.dumps(request.messages),
                "selected_documents": request.selected_documents
            }
        )
        db.commit()
        
        row = result.fetchone()
        if not row:
            raise HTTPException(status_code=500, detail="Failed to create session")
            
        return {
            "id": str(row[0]),
            "title": row[1],
            "messages": row[2],
            "selected_documents": row[3] or [],
            "created_at": str(row[4]),
            "updated_at": str(row[5])
        }
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating chat session: {e}")
        raise HTTPException(status_code=500, detail="Failed to create chat session")


@app.put("/chat/sessions/{session_id}")
async def update_chat_session(
    session_id: str,
    request: UpdateSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """Update an existing chat session"""
    try:
        # Convert admin user_id to UUID for chat_sessions table
        effective_user_id = get_admin_chat_user_id(user_id) if user_id == "admin" else user_id
        
        # Build update query dynamically based on provided fields
        updates = []
        params = {"session_id": session_id, "user_id": effective_user_id}
        
        if request.title is not None:
            updates.append("title = :title")
            params["title"] = request.title
        
        if request.messages is not None:
            updates.append("messages = :messages")
            params["messages"] = json.dumps(request.messages)
        
        if request.selected_documents is not None:
            # Store as array - SQLAlchemy will handle the conversion
            updates.append("selected_documents = :selected_documents")
            # Type ignore for SQLAlchemy parameter binding
            params["selected_documents"] = request.selected_documents  # type: ignore
        
        if not updates:
            raise HTTPException(status_code=400, detail="No fields to update")
        
        updates.append("updated_at = now()")
        update_sql = ", ".join(updates)
        
        result = db.execute(
            text(f"""
                UPDATE chat_sessions
                SET {update_sql}
                WHERE id = :session_id AND user_id = :user_id
                RETURNING id, title, messages, selected_documents, created_at, updated_at
            """),
            params
        )
        db.commit()
        
        row = result.first()
        if not row:
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return {
            "id": str(row[0]),
            "title": row[1],
            "messages": row[2],
            "selected_documents": row[3] or [],
            "created_at": str(row[4]),
            "updated_at": str(row[5])
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating chat session: {e}")
        raise HTTPException(status_code=500, detail="Failed to update chat session")


@app.delete("/chat/sessions/{session_id}")
async def delete_chat_session(
    session_id: str,
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """Delete a chat session"""
    try:
        # Convert admin user_id to UUID for chat_sessions table
        effective_user_id = get_admin_chat_user_id(user_id) if user_id == "admin" else user_id
        
        result = db.execute(
            text("""
                DELETE FROM chat_sessions
                WHERE id = :session_id AND user_id = :user_id
                RETURNING id
            """),
            {"session_id": session_id, "user_id": effective_user_id}
        )
        db.commit()
        
        if not result.first():
            raise HTTPException(status_code=404, detail="Chat session not found")
        
        return {"message": "Chat session deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting chat session: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete chat session")


@app.get("/storage/categories")
async def list_storage_categories():
    """
    Get list of available storage categories (folders).
    """
    try:
        return {"categories": VALID_CATEGORIES}
    except Exception as e:
        logger.error(f"List categories error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve categories")


@app.get("/storage/files/{category}")
async def list_storage_files(category: str):
    """
    Get list of files in a storage category.
    """
    try:
        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category. Must be one of: {VALID_CATEGORIES}")
        
        files = list_files_in_category(category)
        return {"files": files, "category": category}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"List files error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve files")


@app.get("/storage/file-url/{category}/{filename}")
async def get_storage_file_url(category: str, filename: str):
    """
    Get the public URL for a storage file.
    """
    try:
        if category not in VALID_CATEGORIES:
            raise HTTPException(status_code=400, detail=f"Invalid category")
        
        from supabase_storage import get_file_url
        url = get_file_url(filename, category)
        if not url:
            raise HTTPException(status_code=404, detail="File not found")
        return {"url": url}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get file URL error: {e}")
        raise HTTPException(status_code=500, detail="Failed to get file URL")


@app.get("/debug/sources/{filename}")
async def debug_document_sources(
    filename: str,
    db: Session = Depends(get_db)
):
    """
    Debug endpoint to check source metadata for a specific document.
    Shows what sources would be returned for citation links.
    """
    try:
        result = db.execute(
            text("""
                SELECT 
                    filename,
                    chunk_index,
                    page_number,
                    storage_url,
                    metadata,
                    LEFT(content, 100) as content_preview
                FROM documents
                WHERE filename = :filename
                ORDER BY chunk_index
            """),
            {"filename": filename}
        )
        
        chunks = result.fetchall()
        
        if not chunks:
            raise HTTPException(status_code=404, detail=f"Document '{filename}' not found")
        
        sources = []
        for chunk in chunks:
            source = {
                "filename": chunk.filename,
                "chunk_index": chunk.chunk_index,
                "page_number": chunk.page_number,
                "storage_url": chunk.storage_url,
                "category": chunk.metadata.get("category") if chunk.metadata else None,
                "content_preview": chunk.content_preview
            }
            sources.append(source)
        
        return {
            "filename": filename,
            "total_chunks": len(sources),
            "sources": sources,
            "unique_pages": list(set([s["page_number"] for s in sources if s["page_number"] is not None])),
            "storage_url_status": {
                "all_have_urls": all(s["storage_url"] for s in sources),
                "missing_urls": sum(1 for s in sources if not s["storage_url"]),
                "with_urls": sum(1 for s in sources if s["storage_url"])
            }
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Debug sources error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to retrieve debug information")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
