"""
Supabase Storage Client for RubAI Knowledge Base

Handles file uploads to Supabase Storage with organized folder structure:
- /privacy-policies
- /cvs  
- /terms-and-conditions
- /ai-docs (default)
"""

import os
from typing import Optional, Tuple, Any
try:
    from supabase import create_client, Client  # type: ignore
except ImportError:
    print("Warning: supabase package not installed. Install with: pip install supabase")
    create_client = None  # type: ignore
    Client = Any  # type: ignore
from config import SUPABASE_URL, SUPABASE_ANON_KEY, SUPABASE_STORAGE_BUCKET

# Valid categories for knowledge base organization
VALID_CATEGORIES = ["privacy-policies", "cvs", "terms-and-conditions", "ai-docs"]
DEFAULT_CATEGORY = "ai-docs"

# Initialize Supabase client
_supabase_client: Optional["Client"] = None  # type: ignore


def get_supabase_client() -> "Client":  # type: ignore
    """Get or create Supabase client singleton."""
    global _supabase_client
    if _supabase_client is None:
        if not SUPABASE_URL or not SUPABASE_ANON_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_ANON_KEY must be set")
        if create_client is None:  # type: ignore
            raise ImportError("supabase package not installed. Install with: pip install supabase")
        _supabase_client = create_client(SUPABASE_URL, SUPABASE_ANON_KEY)  # type: ignore
    return _supabase_client


def validate_category(category: Optional[str]) -> str:
    """Validate and return a valid category, defaulting to ai-docs."""
    if category is None or category not in VALID_CATEGORIES:
        return DEFAULT_CATEGORY
    return category


def upload_file_to_storage(
    file_content: bytes,
    filename: str,
    category: Optional[str] = None,
    content_type: str = "application/octet-stream"
) -> Tuple[bool, str, Optional[str]]:
    """
    Upload a file to Supabase Storage.
    
    Args:
        file_content: Raw bytes of the file
        filename: Original filename
        category: One of VALID_CATEGORIES (defaults to ai-docs)
        content_type: MIME type of the file
        
    Returns:
        Tuple of (success: bool, message: str, public_url: Optional[str])
    """
    try:
        client = get_supabase_client()
        category = validate_category(category)
        
        # Build storage path: category/filename
        storage_path = f"{category}/{filename}"
        
        # Upload to Supabase Storage
        response = client.storage.from_(SUPABASE_STORAGE_BUCKET).upload(
            path=storage_path,
            file=file_content,
            file_options={"content-type": content_type, "upsert": "true"}
        )
        
        # Generate public URL
        public_url = client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
        
        return True, f"File uploaded to {storage_path}", public_url
        
    except Exception as e:
        error_msg = str(e)
        # Handle duplicate file error gracefully
        if "Duplicate" in error_msg or "already exists" in error_msg.lower():
            # File exists, try to get its URL anyway
            try:
                client = get_supabase_client()
                category = validate_category(category)
                storage_path = f"{category}/{filename}"
                public_url = client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
                return True, f"File already exists at {storage_path}", public_url
            except:
                pass
        return False, f"Upload failed: {error_msg}", None


def delete_file_from_storage(filename: str, category: Optional[str] = None) -> Tuple[bool, str]:
    """
    Delete a file from Supabase Storage.
    
    Args:
        filename: Filename to delete
        category: Category folder the file is in
        
    Returns:
        Tuple of (success: bool, message: str)
    """
    try:
        client = get_supabase_client()
        category = validate_category(category)
        storage_path = f"{category}/{filename}"
        
        client.storage.from_(SUPABASE_STORAGE_BUCKET).remove([storage_path])
        return True, f"File deleted: {storage_path}"
        
    except Exception as e:
        return False, f"Delete failed: {str(e)}"


def get_file_url(filename: str, category: Optional[str] = None) -> Optional[str]:
    """
    Get the public URL for a file in storage.
    
    Args:
        filename: Filename to get URL for
        category: Category folder the file is in
        
    Returns:
        Public URL or None if error
    """
    try:
        client = get_supabase_client()
        category = validate_category(category)
        storage_path = f"{category}/{filename}"
        return client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
    except Exception as e:
        print(f"Error getting file URL: {e}")
        return None


def list_files_in_category(category: Optional[str] = None) -> list:
    """
    List all files in a category folder.
    
    Args:
        category: Category to list files from (defaults to ai-docs)
        
    Returns:
        List of file objects with name, size, etc.
    """
    try:
        client = get_supabase_client()
        category = validate_category(category)
        
        response = client.storage.from_(SUPABASE_STORAGE_BUCKET).list(path=category)
        # Filter out placeholder files
        return [f for f in response if not f.get('name', '').startswith('.')]
        
    except Exception as e:
        print(f"Error listing files: {e}")
        return []


def get_storage_metadata(filename: str, category: str, file_size: int, content_type: str) -> dict:
    """
    Generate metadata dict for storing with document chunks.
    
    Args:
        filename: Original filename
        category: Storage category
        file_size: File size in bytes
        content_type: MIME type
        
    Returns:
        Metadata dictionary for PostgreSQL storage
    """
    public_url = get_file_url(filename, category)
    
    return {
        "upload_type": "knowledge_base",
        "category": category,
        "file_type": content_type,
        "file_size": file_size,
        "mime_type": content_type,
        "original_filename": filename,
        "storage_path": f"{category}/{filename}",
        "file_url": public_url
    }
