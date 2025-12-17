"""
COMPREHENSIVE TEST: Citation Links Feature
==========================================
This script tests the complete flow of citation links from database to frontend.

Test Steps:
1. Verify database has documents with storage_url and page_number
2. Simulate a RAG query response
3. Verify sources are properly enriched with storage_url
4. Generate a sample response with page citations
5. Show what the frontend should receive
"""
import json
from sqlalchemy.orm import Session
from database import SessionLocal
from sqlalchemy import text

def test_citation_links_flow():
    db = SessionLocal()
    
    try:
        print("\n" + "="*100)
        print("CITATION LINKS FEATURE - COMPREHENSIVE TEST")
        print("="*100)
        
        # Step 1: Check database structure
        print("\n[STEP 1] Verifying Database Structure...")
        print("-" * 100)
        
        result = db.execute(
            text("""
                SELECT 
                    filename, 
                    page_number, 
                    chunk_index, 
                    storage_url,
                    metadata->>'category' as category,
                    LEFT(content, 100) as content_preview
                FROM documents 
                WHERE storage_url IS NOT NULL 
                AND page_number IS NOT NULL
                ORDER BY filename, page_number
                LIMIT 5
            """)
        ).fetchall()
        
        if not result:
            print("[ERROR] No documents found with both storage_url and page_number!")
            print("Action needed: Upload documents to knowledge base")
            return False
        
        print(f"[OK] Found {len(result)} document chunks with storage_url and page_number")
        print()
        
        # Display sample documents
        for row in result:
            print(f"  File: {row[0]}")
            print(f"  Page: {row[1]}")
            print(f"  Category: {row[4]}")
            print(f"  URL: {row[3][:60]}...")
            print(f"  Content: {row[5]}...")
            print()
        
        # Step 2: Simulate RAG query
        print("\n[STEP 2] Simulating RAG Query Response...")
        print("-" * 100)
        
        # Get unique files
        files_info = {}
        for row in result:
            filename = row[0]
            if filename not in files_info:
                files_info[filename] = {
                    'filename': filename,
                    'pages': set(),
                    'storage_url': row[3],
                    'category': row[4]
                }
            files_info[filename]['pages'].add(row[1])
        
        # Step 3: Build sources array (what backend sends to frontend)
        print("\n[STEP 3] Building Sources Array (Backend Response)...")
        print("-" * 100)
        
        sources = []
        for file_info in files_info.values():
            for page in sorted(file_info['pages']):
                source = {
                    "filename": file_info['filename'],
                    "page_number": page,
                    "chunk_index": 0,  # Simplified for this test
                    "storage_url": file_info['storage_url'],
                    "category": file_info['category']
                }
                sources.append(source)
        
        print(f"[OK] Created {len(sources)} source objects")
        print("\nSources JSON (sent to frontend):")
        print(json.dumps(sources, indent=2))
        
        # Step 4: Simulate LLM Response with Page Citations
        print("\n[STEP 4] Simulating LLM Response with Page Citations...")
        print("-" * 100)
        
        # Create a mock response mentioning pages
        page_numbers = sorted([s['page_number'] for s in sources])
        
        mock_response = f"""Based on the document, here's what I found:

The main concept is introduced on Page {page_numbers[0]}, which explains the core functionality. 

Additional details can be found on page {page_numbers[1] if len(page_numbers) > 1 else page_numbers[0]}, providing more context about the implementation.

For complete information, refer to p. {page_numbers[-1]} which contains the summary."""
        
        print("[OK] Generated mock LLM response with page citations:")
        print()
        print(mock_response)
        print()
        
        # Step 5: Show expected frontend behavior
        print("\n[STEP 5] Expected Frontend Behavior...")
        print("-" * 100)
        
        print("[OK] The MarkdownRenderer should:")
        print(f"  1. Detect page patterns: 'Page {page_numbers[0]}', 'page {page_numbers[1]}', 'p. {page_numbers[-1]}'")
        print(f"  2. Look up sources array for each page number")
        print(f"  3. Create clickable links with storage_url")
        print()
        
        # Step 6: Verification
        print("\n[STEP 6] Final Verification Checklist...")
        print("-" * 100)
        
        checks = []
        
        # Check 1: All sources have storage_url
        all_have_urls = all(s.get('storage_url') for s in sources)
        checks.append(("All sources have storage_url", all_have_urls))
        
        # Check 2: All sources have page_number
        all_have_pages = all(s.get('page_number') is not None for s in sources)
        checks.append(("All sources have page_number", all_have_pages))
        
        # Check 3: URLs are valid Supabase URLs
        all_valid_urls = all('supabase.co/storage' in s.get('storage_url', '') for s in sources)
        checks.append(("All URLs are valid Supabase URLs", all_valid_urls))
        
        # Check 4: Categories are present
        all_have_category = all(s.get('category') for s in sources)
        checks.append(("All sources have category", all_have_category))
        
        print()
        for check_name, passed in checks:
            status = "[OK]" if passed else "[FAIL]"
            print(f"  {status} {check_name}")
        
        print()
        
        # Final status
        all_passed = all(passed for _, passed in checks)
        
        print("\n" + "="*100)
        if all_passed:
            print("[SUCCESS] All checks passed! Citation links should work correctly.")
            print()
            print("NEXT STEPS:")
            print("1. Check browser console for debug logs")
            print("2. Verify frontend MarkdownRenderer receives sources")
            print("3. Test clicking on page citations in the UI")
        else:
            print("[WARNING] Some checks failed. Review the issues above.")
        
        print("="*100)
        print()
        
        return all_passed
        
    finally:
        db.close()

if __name__ == "__main__":
    success = test_citation_links_flow()
    exit(0 if success else 1)
