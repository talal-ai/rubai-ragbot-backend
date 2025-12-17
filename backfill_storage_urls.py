"""
Storage URL Backfill Script

This script populates missing storage_url values in the documents table.
It reconstructs URLs from filename and category metadata.
"""
import os
from pathlib import Path
from dotenv import load_dotenv
from database import SessionLocal
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

def backfill_storage_urls(dry_run=True):
    """
    Backfill missing storage URLs in the documents table
    
    Args:
        dry_run: If True, only show what would be updated without making changes
    """
    db = SessionLocal()
    
    try:
        logger.info("=" * 80)
        logger.info(f"STORAGE URL BACKFILL {'(DRY RUN)' if dry_run else '(LIVE MODE)'}")
        logger.info("=" * 80)
        logger.info(f"Supabase URL: {SUPABASE_URL}")
        logger.info(f"Storage Bucket: {SUPABASE_STORAGE_BUCKET}\n")
        
        # Find documents with missing storage URLs
        result = db.execute(text("""
            SELECT 
                id,
                filename,
                chunk_index,
                storage_url,
                metadata
            FROM documents
            WHERE storage_url IS NULL OR storage_url = ''
            ORDER BY filename, chunk_index
        """))
        
        docs_missing_url = result.fetchall()
        
        if not docs_missing_url:
            logger.info("✅ All documents already have storage URLs!")
            return
        
        logger.info(f"📊 Found {len(docs_missing_url)} chunks missing storage URLs")
        
        # Group by filename
        by_filename = {}
        for doc in docs_missing_url:
            if doc.filename not in by_filename:
                by_filename[doc.filename] = []
            by_filename[doc.filename].append(doc)
        
        logger.info(f"📄 Affecting {len(by_filename)} unique documents\n")
        
        updated_count = 0
        error_count = 0
        
        for filename, chunks in by_filename.items():
            logger.info(f"Processing: {filename} ({len(chunks)} chunks)")
            
            # Get category from metadata
            first_chunk = chunks[0]
            metadata = first_chunk.metadata or {}
            category = metadata.get("category", "ai-docs")
            
            # Construct storage URL
            encoded_filename = quote(filename)
            storage_url = f"{SUPABASE_URL}/storage/v1/object/public/{SUPABASE_STORAGE_BUCKET}/{category}/{encoded_filename}"
            
            logger.info(f"  📂 Category: {category}")
            logger.info(f"  🔗 URL: {storage_url}")
            
            if not dry_run:
                try:
                    # Update all chunks for this filename
                    result = db.execute(
                        text("""
                            UPDATE documents 
                            SET storage_url = :storage_url
                            WHERE filename = :filename 
                            AND (storage_url IS NULL OR storage_url = '')
                        """),
                        {"storage_url": storage_url, "filename": filename}
                    )
                    db.commit()
                    rows_updated = result.rowcount if hasattr(result, 'rowcount') else 0  # type: ignore
                    updated_count += rows_updated
                    logger.info(f"  ✅ Updated {rows_updated} chunks")
                except Exception as e:
                    logger.error(f"  ❌ Error updating {filename}: {e}")
                    db.rollback()
                    error_count += len(chunks)
            else:
                logger.info(f"  💡 Would update {len(chunks)} chunks")
                updated_count += len(chunks)
            
            logger.info("")
        
        # Summary
        logger.info("=" * 80)
        logger.info("SUMMARY:")
        logger.info("=" * 80)
        
        if dry_run:
            logger.info(f"💡 Would update: {updated_count} chunks")
            logger.info(f"📄 Affecting: {len(by_filename)} documents")
            logger.info("\n⚠️  This was a DRY RUN. No changes were made.")
            logger.info("To apply changes, run: python backfill_storage_urls.py --apply")
        else:
            logger.info(f"✅ Updated: {updated_count} chunks")
            logger.info(f"❌ Errors: {error_count} chunks")
            logger.info(f"📄 Documents affected: {len(by_filename)}")
            logger.info("\n🎉 Backfill complete!")
        
    except Exception as e:
        logger.error(f"Error during backfill: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


def backfill_from_metadata():
    """
    Backfill storage URLs from metadata.storage_url or metadata.file_url
    This is useful if URLs are in metadata but not in the column
    """
    db = SessionLocal()
    
    try:
        logger.info("=" * 80)
        logger.info("BACKFILL FROM METADATA")
        logger.info("=" * 80)
        
        # Find documents where storage_url column is null but metadata has URL
        result = db.execute(text("""
            SELECT 
                id,
                filename,
                chunk_index,
                storage_url,
                metadata
            FROM documents
            WHERE (storage_url IS NULL OR storage_url = '')
            AND (
                metadata->>'storage_url' IS NOT NULL 
                OR metadata->>'file_url' IS NOT NULL
            )
        """))
        
        docs = result.fetchall()
        
        if not docs:
            logger.info("✅ No documents found with URLs in metadata but missing in column")
            return
        
        logger.info(f"📊 Found {len(docs)} chunks with URLs in metadata\n")
        
        updated_count = 0
        
        for doc in docs:
            metadata = doc.metadata or {}
            url_from_meta = metadata.get("storage_url") or metadata.get("file_url")
            
            if url_from_meta:
                logger.info(f"Updating: {doc.filename} (chunk {doc.chunk_index})")
                logger.info(f"  URL from metadata: {url_from_meta}")
                
                try:
                    db.execute(
                        text("UPDATE documents SET storage_url = :url WHERE id = :id"),
                        {"url": url_from_meta, "id": doc.id}
                    )
                    updated_count += 1
                except Exception as e:
                    logger.error(f"  ❌ Error: {e}")
        
        db.commit()
        logger.info(f"\n✅ Updated {updated_count} chunks from metadata")
        
    except Exception as e:
        logger.error(f"Error during metadata backfill: {e}")
        import traceback
        logger.error(traceback.format_exc())
        db.rollback()
    finally:
        db.close()


if __name__ == "__main__":
    import sys
    
    # Check for --apply flag
    apply = "--apply" in sys.argv or "-a" in sys.argv
    metadata_mode = "--metadata" in sys.argv or "-m" in sys.argv
    
    if metadata_mode:
        logger.info("Running in METADATA mode...")
        backfill_from_metadata()
    else:
        backfill_storage_urls(dry_run=not apply)
