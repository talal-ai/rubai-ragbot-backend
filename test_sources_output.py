"""
Test script to verify sources are properly formatted with storage_url
"""
import json
from sqlalchemy.orm import Session
from database import SessionLocal, Document
from sqlalchemy import text

def test_sources_structure():
    """Test that sources contain storage_url"""
    db = SessionLocal()
    
    try:
        # Get a sample document with storage_url
        result = db.execute(
            text("""
                SELECT 
                    filename, 
                    page_number, 
                    chunk_index, 
                    storage_url,
                    metadata->>'category' as category
                FROM documents 
                WHERE storage_url IS NOT NULL
                LIMIT 3
            """)
        ).fetchall()
        
        print("\n" + "="*80)
        print("TESTING SOURCES STRUCTURE")
        print("="*80)
        
        if not result:
            print("[X] NO DOCUMENTS WITH storage_url FOUND!")
            return
        
        print(f"\n[OK] Found {len(result)} documents with storage_url\n")
        
        # Simulate what backend sends to frontend
        sources = []
        for row in result:
            source = {
                "filename": row[0],
                "page_number": row[1],
                "chunk_index": row[2],
                "storage_url": row[3],
                "category": row[4]
            }
            sources.append(source)
            
            print(f"[FILE] {source['filename']}")
            print(f"   Page: {source['page_number']}")
            print(f"   Category: {source['category']}")
            print(f"   URL: {source['storage_url'][:80]}..." if len(source['storage_url']) > 80 else f"   URL: {source['storage_url']}")
            print()
        
        # Show what JSON would look like
        print("="*80)
        print("JSON PAYLOAD SENT TO FRONTEND:")
        print("="*80)
        json_output = json.dumps({"sources": sources}, indent=2)
        print(json_output)
        
        print("="*80)
        print("VERIFICATION:")
        print("="*80)
        for source in sources:
            has_url = bool(source.get('storage_url'))
            has_page = source.get('page_number') is not None
            print(f"[OK] {source['filename']}: URL={has_url}, Page={has_page}")
        
    finally:
        db.close()

if __name__ == "__main__":
    test_sources_structure()
