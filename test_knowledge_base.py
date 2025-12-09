"""
Test script for category-based Knowledge Base functionality

This script tests:
1. Upload a file to privacy-policies category
2. Query: "How many files are in the knowledge base?"
3. Query: "What files are in privacy policies folder?"
4. Query: "What content is in [filename] in privacy policies?"
"""

import requests
import json
import os
from pathlib import Path

API_BASE = "http://localhost:8000"

def test_upload_to_category():
    """Test uploading a file to privacy-policies category"""
    print("\n=== TEST 1: Upload file to privacy-policies ===")
    
    # Find test-doc folder
    test_doc_path = Path(__file__).parent / "test-doc"
    
    if not test_doc_path.exists():
        print("❌ test-doc folder not found")
        return False
    
    # Find first PDF or TXT file in test-doc
    test_files = list(test_doc_path.glob("*.pdf")) + list(test_doc_path.glob("*.txt"))
    
    if not test_files:
        print("❌ No PDF or TXT files found in test-doc")
        return False
    
    test_file = test_files[0]
    print(f"📄 Uploading: {test_file.name}")
    
    with open(test_file, 'rb') as f:
        files = {'file': (test_file.name, f, 'application/pdf' if test_file.suffix == '.pdf' else 'text/plain')}
        response = requests.post(
            f"{API_BASE}/documents/upload?category=privacy-policies",
            files=files
        )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ Upload successful!")
        print(f"   Filename: {result['filename']}")
        print(f"   Category: {result.get('category', 'N/A')}")
        print(f"   Chunks: {result.get('chunks_stored', 'N/A')}")
        return True, test_file.name
    else:
        print(f"❌ Upload failed: {response.status_code}")
        print(response.text)
        return False, None


def test_query_all_files():
    """Test query: How many files in knowledge base?"""
    print("\n=== TEST 2: Query all files ===")
    
    response = requests.get(f"{API_BASE}/documents")
    
    if response.status_code == 200:
        result = response.json()
        docs = result['documents']
        print(f"✅ Found {len(docs)} files in knowledge base")
        
        # Group by category
        by_category = {}
        for doc in docs:
            cat = doc.get('category', 'unknown')
            if cat not in by_category:
                by_category[cat] = []
            by_category[cat].append(doc['filename'])
        
        for cat, files in by_category.items():
            print(f"   {cat}: {len(files)} files")
            for f in files:
                print(f"     - {f}")
        
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        return False


def test_query_category_files():
    """Test query: What files are in privacy-policies?"""
    print("\n=== TEST 3: Query privacy-policies category ===")
    
    response = requests.get(f"{API_BASE}/documents?category=privacy-policies")
    
    if response.status_code == 200:
        result = response.json()
        docs = result['documents']
        print(f"✅ Found {len(docs)} files in privacy-policies")
        for doc in docs:
            print(f"   - {doc['filename']} ({doc['chunk_count']} chunks)")
        return True
    else:
        print(f"❌ Query failed: {response.status_code}")
        return False


def test_rag_query_all(filename):
    """Test RAG: Ask about all knowledge base"""
    print("\n=== TEST 4: RAG Query - All files ===")
    
    query = "How many files are in the knowledge base? List them all."
    
    response = requests.post(
        f"{API_BASE}/chat/rag",
        json={
            "message": query,
            "chat_id": "test-session-all",
            "tone": "Professional"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ RAG Response:")
        print(f"   {result['response'][:300]}...")
        if result.get('sources'):
            print(f"\n   Sources: {len(result['sources'])} documents")
        return True
    else:
        print(f"❌ RAG query failed: {response.status_code}")
        print(response.text)
        return False


def test_rag_query_category(filename):
    """Test RAG: Ask about privacy-policies folder specifically"""
    print("\n=== TEST 5: RAG Query - Privacy Policies only ===")
    
    query = f"What content is in the privacy policies folder? Specifically tell me about {filename}"
    
    response = requests.post(
        f"{API_BASE}/chat/rag",
        json={
            "message": query,
            "category": "privacy-policies",
            "chat_id": "test-session-category",
            "tone": "Professional"
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"✅ RAG Response:")
        print(f"   {result['response'][:300]}...")
        if result.get('sources'):
            print(f"\n   Sources: {len(result['sources'])} documents")
            for source in result['sources']:
                print(f"     - {source.get('filename', 'N/A')} (category: {source.get('category', 'N/A')})")
        return True
    else:
        print(f"❌ RAG query failed: {response.status_code}")
        print(response.text)
        return False


def main():
    print("🧪 Testing Category-Based Knowledge Base\n")
    print("=" * 60)
    
    # Test 1: Upload
    success, filename = test_upload_to_category()
    if not success:
        print("\n❌ Upload test failed. Stopping.")
        return
    
    # Test 2: Query all files
    if not test_query_all_files():
        print("\n⚠️ Query all files failed")
    
    # Test 3: Query category files
    if not test_query_category_files():
        print("\n⚠️ Query category files failed")
    
    # Test 4: RAG query all
    if not test_rag_query_all(filename):
        print("\n⚠️ RAG query all failed")
    
    # Test 5: RAG query category
    if not test_rag_query_category(filename):
        print("\n⚠️ RAG query category failed")
    
    print("\n" + "=" * 60)
    print("🎉 Testing complete!")


if __name__ == "__main__":
    main()
