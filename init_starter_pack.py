"""
Initialize Starter Pack Documents

This script processes files from Supabase Storage and adds them as global documents.

Workflow:
1. Lists files in Supabase Storage (knowledge-base bucket)
2. Downloads and processes each file
3. Stores in database with ownership_type='global' and user_id=NULL
4. All users automatically get access to these documents

Usage:
    python backend/init_starter_pack.py

Requirements:
    - Files must be uploaded to Supabase Storage first (via dashboard)
    - Database migration must be run first (add_starter_pack_support.sql)
"""

import os
import sys
from pathlib import Path
from sqlalchemy import text

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent))

from supabase_storage import get_supabase_client, VALID_CATEGORIES, SUPABASE_STORAGE_BUCKET
from document_processor import process_document
from database import SessionLocal, Document
from embeddings import generate_embedding
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def download_and_process_file(category: str, filename: str):
    """Download file from Supabase Storage and process it"""
    try:
        client = get_supabase_client()
        storage_path = f"{category}/{filename}"
        
        # Download file
        logger.info(f"📥 Downloading: {storage_path}")
        response = client.storage.from_(SUPABASE_STORAGE_BUCKET).download(storage_path)
        
        if not response:
            logger.error(f"❌ Failed to download {storage_path}")
            return None
        
        # Get file extension
        file_extension = Path(filename).suffix[1:].lower()
        
        # Process document (extract text and chunk)
        logger.info(f"⚙️  Processing: {filename}")
        chunks = process_document(response, filename, file_extension)
        
        if not chunks:
            logger.error(f"❌ No content extracted from {filename}")
            return None
        
        # Get public URL
        public_url = client.storage.from_(SUPABASE_STORAGE_BUCKET).get_public_url(storage_path)
        
        # Extract language from filename (e.g., "Privacy_EN.pdf" -> "en")
        language = None
        filename_upper = filename.upper()
        if '_EN' in filename_upper or 'EN.' in filename_upper or filename_upper.startswith('EN_'):
            language = 'en'
        elif '_RU' in filename_upper or 'RU.' in filename_upper or filename_upper.startswith('RU_'):
            language = 'ru'
        elif '_UZ' in filename_upper or 'UZ.' in filename_upper or filename_upper.startswith('UZ_'):
            language = 'uz'
        
        return {
            'filename': filename,
            'chunks': chunks,
            'category': category,
            'storage_url': public_url,
            'language': language,
            'file_size': len(response),
            'file_type': file_extension
        }
        
    except Exception as e:
        logger.error(f"❌ Error processing {filename}: {e}")
        return None


def store_global_document(db, doc_info):
    """Store document as global (ownership_type='global', user_id=NULL)"""
    try:
        metadata = {
            'category': doc_info['category'],
            'upload_type': 'knowledge_base',
            'ownership_type': 'global',
            'file_size': doc_info['file_size'],
            'file_type': doc_info['file_type'],
            'language': doc_info['language'],
            'storage_url': doc_info['storage_url']
        }
        
        stored_count = 0
        for chunk_text, chunk_idx, page_num in doc_info['chunks']:
            try:
                # Generate Gemini embedding (768-dim)
                embedding = generate_embedding(chunk_text)
                
                doc = Document(
                    user_id=None,  # NULL for global documents
                    filename=doc_info['filename'],
                    content=chunk_text,
                    chunk_index=chunk_idx,
                    page_number=page_num,
                    embedding=embedding,
                    storage_url=doc_info['storage_url'],
                    ownership_type='global',
                    doc_metadata=metadata
                )
                db.add(doc)
                stored_count += 1
            except Exception as e:
                logger.warning(f"⚠️  Failed to store chunk {chunk_idx}: {e}")
                continue
        
        db.commit()
        logger.info(f"✅ Stored {stored_count}/{len(doc_info['chunks'])} chunks for {doc_info['filename']}")
        return stored_count
        
    except Exception as e:
        db.rollback()
        logger.error(f"❌ Failed to store document: {e}")
        return 0


def main():
    print("\n" + "="*80)
    print("STARTER PACK INITIALIZATION")
    print("="*80)
    print("\nThis script will:")
    print("1. List files in Supabase Storage (knowledge-base bucket)")
    print("2. Process and store them as global documents")
    print("3. Make them available to all users\n")
    
    client = get_supabase_client()
    db = SessionLocal()
    
    total_files = 0
    total_chunks = 0
    skipped_files = 0
    
    try:
        # Process each category
        for category in VALID_CATEGORIES:
            print(f"\n📁 Category: {category}")
            print("-" * 80)
            
            # List files in category
            try:
                files = client.storage.from_(SUPABASE_STORAGE_BUCKET).list(path=category)
                files = [f for f in files if not f['name'].startswith('.')]
                
                if not files:
                    print(f"  ⚠️  No files found in {category}")
                    continue
                
                print(f"  Found {len(files)} files")
                
                for file_info in files:
                    filename = file_info['name']
                    
                    # Check if already exists as global document
                    existing = db.execute(
                        text("SELECT COUNT(*) FROM documents WHERE filename = :filename AND ownership_type = 'global'"),
                        {"filename": filename}
                    ).scalar()
                    
                    if existing > 0:
                        print(f"  ⏭️  Skipping: {filename} (already exists as global)")
                        skipped_files += 1
                        continue
                    
                    # Download and process
                    doc_info = download_and_process_file(category, filename)
                    if doc_info:
                        chunks_stored = store_global_document(db, doc_info)
                        if chunks_stored > 0:
                            total_files += 1
                            total_chunks += chunks_stored
                            lang_str = f" ({doc_info['language'].upper()})" if doc_info['language'] else ""
                            print(f"  ✅ {filename}{lang_str}: {chunks_stored} chunks")
                        else:
                            print(f"  ❌ Failed to store: {filename}")
                    else:
                        print(f"  ❌ Failed to process: {filename}")
                        
            except Exception as e:
                print(f"  ❌ Error processing category {category}: {e}")
                continue
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(f"Files processed: {total_files}")
        print(f"Files skipped (already exist): {skipped_files}")
        print(f"Chunks stored: {total_chunks}")
        
        if total_files > 0:
            print("\n✅ Starter pack initialization complete!")
            print("\nNext steps:")
            print("1. Verify global documents: SELECT filename, ownership_type FROM documents WHERE ownership_type = 'global';")
            print("2. Start the backend: cd backend && python main.py")
            print("3. Test with a new user account")
        else:
            print("\n⚠️  No new files were processed.")
            print("Make sure files are uploaded to Supabase Storage first.")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


if __name__ == "__main__":
    main()
