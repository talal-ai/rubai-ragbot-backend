"""
Storage URL Validation and Debug Script

This script helps diagnose issues with storage URLs in the knowledge base.
It checks:
1. Which documents have storage URLs
2. Which documents are missing storage URLs
3. URL format validation
4. Provides backfill suggestions
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from database import SessionLocal, Document
from sqlalchemy import text
from urllib.parse import quote
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_STORAGE_BUCKET = os.getenv("SUPABASE_STORAGE_BUCKET", "knowledge-base")

def validate_storage_urls():
    """Validate all storage URLs in the database"""
    db = SessionLocal()
    
    try:
        logger.info("=" * 80)
        logger.info("STORAGE URL VALIDATION REPORT")
        logger.info("=" * 80)
        
        # Get all unique documents
        result = db.execute(text("""
            SELECT DISTINCT 
                filename,
                storage_url,
                metadata->>'category' as category,
                metadata->>'upload_type' as upload_type,
                COUNT(*) OVER (PARTITION BY filename) as chunk_count
            FROM documents
            ORDER BY filename
        """))
        
        docs = result.fetchall()
        
        if not docs:
            logger.warning("❌ No documents found in database!")
            return
        
        logger.info(f"\n📊 Total unique documents: {len(docs)}")
        logger.info(f"📦 Supabase URL: {SUPABASE_URL}")
        logger.info(f"🪣 Bucket: {SUPABASE_STORAGE_BUCKET}\n")
        
        # Statistics
        with_url = 0
        without_url = 0
        invalid_url = 0
        
        logger.info("=" * 80)
        logger.info("DOCUMENT STATUS:")
        logger.info("=" * 80)
        
        for doc in docs:
            filename = doc.filename
            storage_url = doc.storage_url
            category = doc.category or "ai-docs"
            upload_type = doc.upload_type
            chunk_count = doc.chunk_count
            
            # Check URL status
            has_url = bool(storage_url)
            is_valid = has_url and SUPABASE_URL in storage_url if SUPABASE_URL else False
            
            # Status icon
            if has_url and is_valid:
                status = "✅ VALID"
                with_url += 1
            elif has_url and not is_valid:
                status = "⚠️  INVALID"
                invalid_url += 1
            else:
                status = "❌ MISSING"
                without_url += 1
            
            logger.info(f"\n{status} | {filename}")
            logger.info(f"  📂 Category: {category}")
            logger.info(f"  📄 Type: {upload_type}")
            logger.info(f"  🧩 Chunks: {chunk_count}")
            
            if has_url:
                logger.info(f"  🔗 URL: {storage_url}")
                
                # Validate URL format
                if not is_valid and SUPABASE_URL:
                    logger.warning(f"  ⚠️  URL doesn't match expected pattern!")
                    logger.info(f"  Expected to contain: {SUPABASE_URL}")
            else:
                # Suggest URL
                encoded_filename = quote(filename)
                suggested_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{category}/{encoded_filename}"
                logger.info(f"  💡 Suggested URL: {suggested_url}")
        
        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("SUMMARY:")
        logger.info("=" * 80)
        logger.info(f"✅ Documents with valid URLs: {with_url}")
        logger.info(f"❌ Documents missing URLs: {without_url}")
        logger.info(f"⚠️  Documents with invalid URLs: {invalid_url}")
        logger.info(f"📊 Total documents: {len(docs)}")
        
        percentage = (with_url / len(docs) * 100) if docs else 0
        logger.info(f"\n🎯 Coverage: {percentage:.1f}%")
        
        if without_url > 0 or invalid_url > 0:
            logger.info("\n" + "=" * 80)
            logger.info("RECOMMENDED ACTIONS:")
            logger.info("=" * 80)
            logger.info("1. Run backfill script to populate missing URLs")
            logger.info("2. Check Supabase storage bucket for file existence")
            logger.info("3. Verify category structure matches: privacy-policies, cvs, terms-and-conditions, ai-docs")
            logger.info("\nTo backfill URLs, run: python backfill_storage_urls.py")
        else:
            logger.info("\n🎉 All documents have valid storage URLs!")
        
    except Exception as e:
        logger.error(f"Error validating storage URLs: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db.close()


def check_specific_document(filename: str):
    """Check storage URL for a specific document"""
    db = SessionLocal()
    
    try:
        logger.info(f"\n🔍 Checking document: {filename}")
        logger.info("=" * 80)
        
        result = db.execute(text("""
            SELECT 
                id,
                filename,
                chunk_index,
                page_number,
                storage_url,
                metadata
            FROM documents
            WHERE filename = :filename
            ORDER BY chunk_index
        """), {"filename": filename})
        
        chunks = result.fetchall()
        
        if not chunks:
            logger.warning(f"❌ No chunks found for document: {filename}")
            return
        
        logger.info(f"📄 Document: {filename}")
        logger.info(f"🧩 Total chunks: {len(chunks)}")
        
        # Check first chunk
        first_chunk = chunks[0]
        storage_url = first_chunk.storage_url
        metadata = first_chunk.metadata
        
        logger.info(f"\n📦 Metadata:")
        if metadata:
            for key, value in metadata.items():
                logger.info(f"  {key}: {value}")
        else:
            logger.info("  No metadata found")
        
        logger.info(f"\n🔗 Storage URL:")
        if storage_url:
            logger.info(f"  {storage_url}")
            logger.info(f"  ✅ URL is set")
        else:
            logger.warning(f"  ❌ No storage URL found")
            
            # Try to get from metadata
            if metadata and metadata.get("storage_url"):
                logger.info(f"  💡 Found in metadata: {metadata['storage_url']}")
            elif metadata and metadata.get("file_url"):
                logger.info(f"  💡 Found file_url in metadata: {metadata['file_url']}")
        
        logger.info(f"\n📃 Sample chunks:")
        for i, chunk in enumerate(chunks[:3], 1):
            logger.info(f"  Chunk {chunk.chunk_index}: Page {chunk.page_number or 'N/A'}, URL: {'SET' if chunk.storage_url else 'MISSING'}")
        
        if len(chunks) > 3:
            logger.info(f"  ... and {len(chunks) - 3} more chunks")
        
    except Exception as e:
        logger.error(f"Error checking document: {e}")
        import traceback
        logger.error(traceback.format_exc())
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        # Check specific document
        check_specific_document(sys.argv[1])
    else:
        # Validate all documents
        validate_storage_urls()
