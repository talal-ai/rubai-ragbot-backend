from sqlalchemy.orm import Session
from sqlalchemy import text, func
from sqlalchemy.exc import IntegrityError
from typing import List, Dict, Optional
from database import Document
from embeddings import generate_embedding, generate_query_embedding, generate_embeddings_batch
import logging
from config import SIMILARITY_THRESHOLD

logger = logging.getLogger(__name__)

def store_document_chunks(
    db: Session,
    filename: str,
    chunks: List[tuple],
    metadata: Optional[Dict] = None,
    user_id: Optional[str] = None
) -> int:
    """
    Store document chunks with embeddings in the database using batch embedding generation.
    Handles duplicates gracefully.
    
    Args:
        db: Database session
        filename: Name of the document
        chunks: List of (chunk_text, chunk_index) or (chunk_text, chunk_index, page_number) tuples
        metadata: Optional metadata dictionary
        user_id: Optional user ID to associate with the document
    
    Returns:
        Number of chunks stored
    """
    if not chunks:
        logger.warning(f"No chunks to store for {filename}")
        return 0
    
    stored_count = 0
    metadata_json = metadata or {}
    
    logger.info(f"Storing {len(chunks)} chunks for {filename} (user_id={user_id})")
    
    # Parse chunks and extract texts for batch embedding
    parsed_chunks = []
    chunk_texts = []
    
    for chunk_data in chunks:
        # Support both old format (text, index) and new format (text, index, page_number)
        if len(chunk_data) == 3:
            chunk_text, chunk_index, page_number = chunk_data
        else:
            chunk_text, chunk_index = chunk_data
            page_number = None
        
        parsed_chunks.append({
            'text': chunk_text,
            'index': chunk_index,
            'page_number': page_number
        })
        chunk_texts.append(chunk_text)
    
    # Generate embeddings in batch (MUCH faster!)
    try:
        logger.info(f"Generating embeddings in batch for {len(chunk_texts)} chunks...")
        embeddings = generate_embeddings_batch(chunk_texts)
        logger.info(f"Batch embedding generation complete")
    except Exception as e:
        logger.error(f"Failed to generate batch embeddings: {e}")
        raise
    
    # Store chunks with their embeddings
    for chunk_info, embedding in zip(parsed_chunks, embeddings):
        try:
            # Create document record (Gemini embeddings: 768-dim)
            doc = Document(
                filename=filename,
                content=chunk_info['text'],
                chunk_index=chunk_info['index'],
                page_number=chunk_info['page_number'],
                embedding=embedding,  # Use 768-dim Gemini embeddings
                doc_metadata=metadata_json,
                user_id=user_id
            )
            
            db.add(doc)
            db.flush()  # Flush to catch unique constraint violations
            stored_count += 1
            
        except IntegrityError:
            # Handle duplicate (filename, chunk_index, user_id)
            logger.warning(f"Duplicate chunk skipped: {filename} chunk {chunk_info['index']}")
            continue
        except Exception as e:
            logger.error(f"Error storing chunk {chunk_info['index']} of {filename}: {e}")
            continue
    
    # Commit all successfully added documents
    if stored_count > 0:
        try:
            db.commit()
            logger.info(f"Successfully stored {stored_count}/{len(chunks)} chunks for {filename}")
        except Exception as e:
            db.rollback()
            logger.error(f"Failed to commit chunks for {filename}: {e}")
            raise
    else:
        logger.warning(f"No chunks were successfully stored for {filename}")
        
    return stored_count

def search_similar_chunks(
    db: Session,
    query: str,
    top_k: int = 5,
    similarity_threshold: float = None,
    filename_filter: str = None,
    page_filter: int = None,
    user_id: Optional[str] = None,
    filenames: Optional[List[str]] = None,
    category: Optional[str] = None
) -> List[Dict]:
    """
    Search for similar document chunks using vector similarity
    
    Args:
        db: Database session
        query: Search query
        top_k: Number of top results to return
        similarity_threshold: Minimum similarity score (0-1, default from config)
        filename_filter: Optional filter by single filename
        page_filter: Optional filter by page number
        user_id: Optional user ID to filter documents (if None, searches all docs)
        filenames: Optional list of filenames to search within (for chat context isolation)
        category: Optional filter by storage category (privacy-policies, cvs, terms-and-conditions, ai-docs)
    
    Returns:
        List of similar chunks with metadata including page numbers
    """
    try:
        if not query or not query.strip():
            logger.warning("Empty query provided")
            return []
        
        threshold = similarity_threshold if similarity_threshold is not None else SIMILARITY_THRESHOLD
        
        logger.info(f"Searching for query: '{query[:50]}...' (top_k={top_k}, threshold={threshold}, user_id={user_id})")
        
        # Generate query embedding
        query_embedding = generate_query_embedding(query)
        
        # Build the SQL query with optional filters (use embedding: 768-dim Gemini)
        where_clauses = ["1 - (embedding <=> :query_embedding) > :threshold"]
        params = {
            "query_embedding": str(query_embedding),
            "threshold": threshold,
            "limit": top_k
        }
        
        # Filter by user_id if provided
        if user_id:
            where_clauses.append("(user_id = :user_id OR user_id IS NULL)")
            params["user_id"] = user_id
        
        if filename_filter:
            where_clauses.append("filename = :filename")
            params["filename"] = filename_filter
        
        # Filter by multiple filenames (for chat context isolation)
        if filenames and len(filenames) > 0:
            placeholders = ", ".join([f":fn_{i}" for i in range(len(filenames))])
            where_clauses.append(f"filename IN ({placeholders})")
            for i, fn in enumerate(filenames):
                params[f"fn_{i}"] = fn
        
        if page_filter is not None:
            where_clauses.append("page_number = :page_number")
            params["page_number"] = page_filter
        
        # Filter by category if provided
        if category:
            where_clauses.append("metadata->>'category' = :category")
            params["category"] = category
        
        where_sql = " AND ".join(where_clauses)
        
        # Perform vector similarity search using cosine distance with embedding
        # Note: <=> is the cosine distance operator in pgvector
        sql = text(f"""
            SELECT 
                id,
                filename,
                content,
                chunk_index,
                page_number,
                1 - (embedding <=> :query_embedding) as similarity
            FROM documents
            WHERE {where_sql}
            ORDER BY embedding <=> :query_embedding
            LIMIT :limit
        """)
        
        result = db.execute(sql, params)
        
        chunks = []
        for row in result:
            chunks.append({
                "id": row[0],
                "filename": row[1],
                "content": row[2],
                "chunk_index": row[3],
                "page_number": row[4],
                "similarity": float(row[5])
            })
        
        logger.info(f"Found {len(chunks)} similar chunks")
        return chunks
        
    except Exception as e:
        logger.error(f"Error searching chunks: {e}")
        db.rollback()
        raise RuntimeError(f"Search failed: {str(e)}")

def get_all_documents(db: Session, user_id: Optional[str] = None, category: Optional[str] = None) -> List[Dict]:
    """
    Get list of all unique documents with full metadata for a user.
    
    Args:
        user_id: Optional user ID to filter documents
        category: Optional category filter (privacy-policies, cvs, terms-and-conditions, ai-docs)
    
    Returns documents with:
    - filename, chunk_count, uploaded_at (basic info)
    - storage_url: Supabase Storage public URL
    - category: Document category (privacy-policies, cvs, terms-and-conditions, ai-docs)
    - file_size: File size in bytes
    - file_type: File extension
    - upload_type: Type of upload (knowledge_base, attachment)
    """
    try:
        # Build where clause for user and category filtering
        where_clauses = []
        params = {}
        if user_id:
            where_clauses.append("(d.user_id = :user_id OR d.user_id IS NULL)")
            params["user_id"] = user_id
        if category:
            where_clauses.append("d.metadata->>'category' = :category")
            params["category"] = category
        
        user_filter = "WHERE " + " AND ".join(where_clauses) if where_clauses else ""
        
        # Use a subquery to get the first metadata and storage_url for each file
        sql = text(f"""
            SELECT 
                d.filename, 
                COUNT(*) as chunk_count, 
                MAX(d.created_at) as uploaded_at,
                (
                    SELECT metadata 
                    FROM documents 
                    WHERE filename = d.filename 
                    LIMIT 1
                ) as doc_metadata,
                (
                    SELECT storage_url 
                    FROM documents 
                    WHERE filename = d.filename AND storage_url IS NOT NULL
                    LIMIT 1
                ) as storage_url
            FROM documents d
            {user_filter}
            GROUP BY d.filename
            ORDER BY uploaded_at DESC
        """)
        
        result = db.execute(sql, params)
        
        documents = []
        for row in result:
            doc = {
                "filename": row[0],
                "chunk_count": row[1],
                "uploaded_at": str(row[2]),
                "storage_url": row[4],  # Direct from storage_url column
                "category": None,
                "file_size": None,
                "file_type": None,
                "upload_type": None
            }
            
            # Extract metadata fields
            if row[3]:
                try:
                    meta = row[3]
                    # Storage URL (fallback to metadata if column is null)
                    if not doc['storage_url'] and 'storage_url' in meta:
                        doc['storage_url'] = meta['storage_url']
                    if not doc['storage_url'] and 'file_url' in meta:
                        doc['storage_url'] = meta['file_url']
                    
                    # Category
                    if 'category' in meta:
                        doc['category'] = meta['category']
                    
                    # File size
                    if 'file_size' in meta:
                        doc['file_size'] = meta['file_size']
                    
                    # File type
                    if 'file_type' in meta:
                        doc['file_type'] = meta['file_type']
                    
                    # Upload type
                    if 'upload_type' in meta:
                        doc['upload_type'] = meta['upload_type']
                except Exception as e:
                    logger.warning(f"Error parsing metadata for {row[0]}: {e}")
            
            documents.append(doc)
        
        logger.info(f"Retrieved {len(documents)} documents with full metadata")
        return documents
        
    except Exception as e:
        logger.error(f"Error getting documents: {e}")
        raise RuntimeError(f"Failed to retrieve documents: {str(e)}")

def get_document_metadata(db: Session, filename: str, user_id: Optional[str] = None) -> Optional[Dict]:
    """
    Get metadata for a specific document.
    
    Args:
        db: Database session
        filename: Name of the document
        user_id: Optional user ID filter
        
    Returns:
        Document metadata dict or None if not found
    """
    try:
        # Build query with user filter
        query = db.query(Document).filter(Document.filename == filename)
        if user_id:
            query = query.filter((Document.user_id == user_id) | (Document.user_id == None))
        
        # Get first document to extract metadata
        doc = query.first()
        if not doc:
            return None
            
        # Extract metadata
        metadata = doc.doc_metadata or {}
        return {
            'category': metadata.get('category'),
            'file_size': metadata.get('file_size'),
            'file_type': metadata.get('file_type'),
            'upload_type': metadata.get('upload_type'),
            'storage_url': doc.storage_url
        }
    except Exception as e:
        logger.error(f"Error getting document metadata for {filename}: {e}")
        return None

def delete_document(db: Session, filename: str, user_id: Optional[str] = None) -> int:
    """Delete all chunks of a document owned by the user"""
    try:
        if not filename or not filename.strip():
            raise ValueError("Filename cannot be empty")
        
        # Build query with user filter
        query = db.query(Document).filter(Document.filename == filename)
        if user_id:
            # Only delete if user owns the document or it's a shared document (no owner)
            query = query.filter((Document.user_id == user_id) | (Document.user_id == None))
        
        deleted = query.delete(synchronize_session=False)
        db.commit()
        
        logger.info(f"Deleted {deleted} chunks for document: {filename} (user: {user_id})")
        return deleted
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error deleting document {filename}: {e}")
        db.rollback()
        raise RuntimeError(f"Failed to delete document: {str(e)}")
