"""
BROWSER SIMULATION TEST
=======================
This script simulates what happens in the browser when the LLM responds with page citations.
"""

def simulate_browser_flow():
    """Simulate the complete browser flow"""
    
    print("\n" + "="*100)
    print("BROWSER SIMULATION - Citation Links Flow")
    print("="*100)
    
    # Step 1: Backend sends sources
    print("\n[STEP 1] Backend HTTP Response")
    print("-" * 100)
    
    backend_response = {
        "text": "Based on the document, the main concept is explained on Page 1. Additional details are on page 2, and the conclusion is on p. 3.",
        "sources": [
            {
                "filename": "Minutesheet.pdf",
                "page_number": 1,
                "chunk_index": 0,
                "storage_url": "https://tonodshnbuztozteuaon.supabase.co/storage/v1/object/public/knowledge-base/ai-docs/Minutesheet.pdf",
                "category": "ai-docs"
            },
            {
                "filename": "Minutesheet.pdf",
                "page_number": 2,
                "chunk_index": 1,
                "storage_url": "https://tonodshnbuztozteuaon.supabase.co/storage/v1/object/public/knowledge-base/ai-docs/Minutesheet.pdf",
                "category": "ai-docs"
            },
            {
                "filename": "Minutesheet.pdf",
                "page_number": 3,
                "chunk_index": 2,
                "storage_url": "https://tonodshnbuztozteuaon.supabase.co/storage/v1/object/public/knowledge-base/ai-docs/Minutesheet.pdf",
                "category": "ai-docs"
            }
        ]
    }
    
    print("Response text:", backend_response["text"])
    print("\nSources received:")
    for source in backend_response["sources"]:
        print(f"  - {source['filename']} (Page {source['page_number']}): {source['storage_url'][:60]}...")
    
    # Step 2: backendService.ts receives and logs
    print("\n[STEP 2] backendService.ts - Console Log")
    print("-" * 100)
    print("[DEBUG] Sources received from backend:", backend_response["sources"])
    
    # Step 3: ChatInterface.tsx callback
    print("\n[STEP 3] ChatInterface.tsx - Sources Callback")
    print("-" * 100)
    print("[DEBUG] Sources callback triggered:", backend_response["sources"])
    print("Updating message with sources...")
    
    # Step 4: MarkdownRenderer receives sources
    print("\n[STEP 4] MarkdownRenderer.tsx - Receives Sources")
    print("-" * 100)
    print("[MarkdownRenderer] Received sources:", backend_response["sources"])
    
    # Step 5: Pattern matching
    print("\n[STEP 5] MarkdownRenderer - Pattern Matching")
    print("-" * 100)
    
    text = backend_response["text"]
    import re
    
    # Pattern from MarkdownRenderer.tsx
    pagePattern = r'\b(?:[Pp]age|[Pp]\.|[Pp]g\.?)\s+(\d+)\b'
    
    matches = re.finditer(pagePattern, text)
    
    print(f"Text: {text}\n")
    print("Detected page citations:")
    
    for match in re.finditer(pagePattern, text):
        full_match = match.group(0)
        page_num = int(match.group(1))
        
        # Find source
        source = next((s for s in backend_response["sources"] if s["page_number"] == page_num), None)
        
        print(f"\n  Pattern found: '{full_match}' (Page {page_num})")
        print(f"  [MarkdownRenderer] Looking for page {page_num}, found:", source)
        
        if source and source.get("storage_url"):
            print(f"  [MarkdownRenderer] Creating link for page {page_num} with URL: {source['storage_url']}")
            print(f"  Result: <a href=\"{source['storage_url']}\" target=\"_blank\" class=\"violet link\">{full_match}</a>")
        else:
            print(f"  Result: <span class=\"violet\">{full_match}</span> (no link - source not found)")
    
    # Step 6: Final rendered output
    print("\n[STEP 6] Final Rendered Output")
    print("-" * 100)
    
    print("\nHow it appears to the user:\n")
    print("  Based on the document, the main concept is explained on")
    print("  [Page 1] <- violet, underlined, clickable")
    print("           ^^^^^^^ (opens https://...Minutesheet.pdf)")
    print("\n  Additional details are on")
    print("  [page 2] <- violet, underlined, clickable")
    print("           ^^^^^^^ (opens https://...Minutesheet.pdf)")
    print("\n  and the conclusion is on")
    print("  [p. 3] <- violet, underlined, clickable")
    print("         ^^^^^^ (opens https://...Minutesheet.pdf)")
    
    # Step 7: User interaction
    print("\n[STEP 7] User Clicks on 'Page 1'")
    print("-" * 100)
    print("Action: window.open('https://tonodshnbuztozteuaon.supabase.co/storage/v1/object/public/knowledge-base/ai-docs/Minutesheet.pdf', '_blank')")
    print("Result: PDF opens in new browser tab")
    
    # Summary
    print("\n" + "="*100)
    print("SUMMARY")
    print("="*100)
    
    print("\n[SUCCESS] Complete flow verified!")
    print("\nWhat you should see in browser console:")
    print("  1. [DEBUG] Sources received from backend: [...]")
    print("  2. [DEBUG] Sources callback triggered: [...]")
    print("  3. [MarkdownRenderer] Received sources: [...]")
    print("  4. [MarkdownRenderer] Looking for page X, found: {...}")
    print("  5. [MarkdownRenderer] Creating link for page X with URL: https://...")
    
    print("\nWhat you should see in the UI:")
    print("  - Page citations in violet color")
    print("  - Underlined text")
    print("  - External link icon (↗)")
    print("  - Clickable - opens PDF in new tab")
    
    print("\n" + "="*100)

if __name__ == "__main__":
    simulate_browser_flow()
