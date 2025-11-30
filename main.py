from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, Query, Header
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import Optional, List
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
    GoogleAuthRequest, TokenResponse, User,
    verify_google_token, get_user_by_google_id, get_user_by_email,
    create_user, update_user, user_to_dict,
    get_current_user, get_current_user_optional, create_access_token
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
    """
    if not authorization:
        return None
    
    try:
        # Extract token from "Bearer <token>"
        if authorization.startswith("Bearer "):
            token = authorization[7:]
        else:
            token = authorization
        
        # If JWT secret is configured, verify the token properly
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
                    logger.debug(f"Verified user_id from Supabase token: {user_id}")
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
                logger.debug(f"Extracted user_id from Supabase token (unverified): {user_id}")
            return user_id
    except Exception as e:
        logger.warning(f"Failed to extract user_id from token: {e}")
        return None


async def require_supabase_user(authorization: Optional[str] = Header(None)) -> str:
    """
    Require a valid Supabase user. Raises 401 if not authenticated.
    """
    user_id = await get_supabase_user_id(authorization)
    if not user_id:
        raise HTTPException(status_code=401, detail="Authentication required")
    return user_id

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
    chat_id: str = Field(..., min_length=1, description="Unique chat session identifier for conversation memory")  # NEW: Required for memory
    
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
        
        # Create JWT token
        access_token = create_access_token(str(user.id), str(user.email))
        
        logger.info(f"User authenticated: {user.email}")
        
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
    user_id: Optional[str] = Depends(get_supabase_user_id)
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
        
        # If this document is already in the knowledge base for this user, skip heavy work
        if user_id:
            existing = db.execute(
                text("SELECT 1 FROM documents WHERE filename = :filename AND user_id = :user_id LIMIT 1"),
                {"filename": filename, "user_id": user_id}
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
                doc = Document(
                    user_id=user_id,
                    filename=filename,
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    embedding=embedding,  # 768-dim Gemini embedding
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
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Get list of all uploaded documents for the current user.
    Shows user's own documents plus shared documents (no owner).
    Optional category filter to show documents from specific folder.
    """
    try:
        documents = get_all_documents(db, user_id=user_id, category=category)
        return {"documents": documents, "count": len(documents)}
    except Exception as e:
        logger.error(f"List documents error: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

@app.delete("/documents/{filename}")
async def remove_document(
    filename: str,
    db: Session = Depends(get_db),
    user_id: Optional[str] = Depends(get_supabase_user_id)
):
    """
    Delete a document and all its chunks.
    User can only delete their own documents or shared documents.
    """
    try:
        # Sanitize filename
        filename = sanitize_filename(filename)
        
        deleted_count = delete_document(db, filename, user_id=user_id)
        
        if deleted_count == 0:
            raise HTTPException(status_code=404, detail="Document not found or you don't have permission to delete it")
        
        logger.info(f"Document deleted: {filename}, {deleted_count} chunks (user: {user_id})")
        
        return {
            "message": "Document deleted successfully",
            "filename": filename,
            "chunks_deleted": deleted_count
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete error: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document")

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
    user_id: Optional[str] = Depends(get_supabase_user_id)
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
        
        # Initialize RAG service
        rag_service = RAGService(db)
        
        # Build system prompt with tone
        system_prompt = f"[System: Respond in a {request.tone} tone]\n\n{rag_service._get_default_system_prompt()}"
        
        # Chat with direct retrieval (uses our vector_store search, not LlamaIndex PGVectorStore)
        response_text, sources = rag_service.chat_direct(
            user_id=effective_user_id,
            chat_id=request.chat_id,
            message=request.message,
            selected_documents=request.selected_documents,
            category=request.category,
            system_prompt=system_prompt
        )
        
        logger.info(f"✅ RAG response generated with direct retrieval and {len(sources)} sources")
        
        # **PHASE 2.3**: Sources are now extracted from LlamaIndex response
        # Only display sources if they exist and have metadata
        
        return ChatResponse(
            response=response_text,
            sources=sources
        )
    
    except Exception as e:
        logger.error(f"RAG chat error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail="Failed to generate RAG response")


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
        user_id: User identifier
        chat_id: Chat session ID
        user_message: The user's message
        ai_response: The AI's response
        attachment: Optional attachment data
        sources: Optional source citations
        tone: Optional tone used
    """
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
        
        # Check if session exists
        result = db.execute(
            text("SELECT messages FROM chat_sessions WHERE id = :chat_id AND user_id = :user_id"),
            {"chat_id": chat_id, "user_id": user_id}
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
                    "user_id": user_id
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
                    "user_id": user_id,
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
    user_id: Optional[str] = Depends(get_supabase_user_id)
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
                    
                    # Add chunks to vector store
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
            
            # Build system prompt with tone
            system_prompt = f"[System: Respond in a {request.tone} tone]\n\n{rag_service._get_default_system_prompt()}"
            
            # Stream response with direct retrieval (uses our vector_store, not LlamaIndex PGVectorStore)
            token_count = 0
            for data in rag_service.chat_stream_direct(
                user_id=effective_user_id,
                chat_id=request.chat_id,
                message=request.message,
                selected_documents=request.selected_documents,
                category=request.category,
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
            try:
                await sync_chat_session(
                    db=db,
                    user_id=effective_user_id,
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
        # For chat history, we need an identifier - use "anonymous" as fallback for key generation
        effective_user_id = user_id or "anonymous"
        
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
        # For chat history deletion, use "anonymous" as fallback for key generation
        effective_user_id = user_id or "anonymous"
        
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
        result = db.execute(
            text("""
                SELECT id, title, messages, selected_documents, created_at, updated_at
                FROM chat_sessions
                WHERE user_id = :user_id
                ORDER BY updated_at DESC
            """),
            {"user_id": user_id}
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
        
        return {"sessions": sessions, "count": len(sessions)}
    except Exception as e:
        logger.error(f"Error listing chat sessions: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve chat sessions")


@app.post("/chat/sessions")
async def create_chat_session(
    request: CreateSessionRequest,
    db: Session = Depends(get_db),
    user_id: str = Depends(require_supabase_user)
):
    """Create a new chat session"""
    try:
        result = db.execute(
            text("""
                INSERT INTO chat_sessions (user_id, title, messages, selected_documents)
                VALUES (:user_id, :title, :messages, :selected_documents)
                RETURNING id, title, messages, selected_documents, created_at, updated_at
            """),
            {
                "user_id": user_id,
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
        # Build update query dynamically based on provided fields
        updates = []
        params = {"session_id": session_id, "user_id": user_id}
        
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
        result = db.execute(
            text("""
                DELETE FROM chat_sessions
                WHERE id = :session_id AND user_id = :user_id
                RETURNING id
            """),
            {"session_id": session_id, "user_id": user_id}
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
