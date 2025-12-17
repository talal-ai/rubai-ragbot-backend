#!/usr/bin/env python3
"""
Script to list and delete all files from Supabase storage AND clean up database records
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from supabase_storage import list_files_in_category, delete_file_from_storage, VALID_CATEGORIES
from database import SessionLocal
from sqlalchemy import text

def list_all_files():
    """List all files in all categories"""
    total_files = 0
    all_files = {}
    
    for category in VALID_CATEGORIES:
        files = list_files_in_category(category)
        all_files[category] = files
        total_files += len(files)
        print(f"{category}: {len(files)} files")
        for file in files:
            print(f"  - {file['name']}")
    
    print(f"\nTotal files: {total_files}")
    return all_files

def list_database_documents():
    """List all documents in the database"""
    db = SessionLocal()
    try:
        result = db.execute(text('SELECT DISTINCT filename FROM documents ORDER BY filename'))
        files = [row[0] for row in result]
        print(f"\nDatabase documents: {len(files)}")
        for file in files:
            print(f"  - {file}")
        return files
    finally:
        db.close()

def delete_all_files():
    """Delete all files from storage AND database"""
    print("=== DELETING FROM SUPABASE STORAGE ===")
    total_deleted_storage = 0
    
    for category in VALID_CATEGORIES:
        files = list_files_in_category(category)
        print(f"Deleting {len(files)} files from {category}...")
        
        for file in files:
            filename = file['name']
            success, message = delete_file_from_storage(filename, category)
            if success:
                print(f"  ✓ Deleted from storage: {filename}")
                total_deleted_storage += 1
            else:
                print(f"  ✗ Failed to delete from storage {filename}: {message}")
    
    print(f"\nStorage cleanup: {total_deleted_storage} files deleted")
    
    print("\n=== CLEANING UP DATABASE ===")
    db = SessionLocal()
    try:
        # Count chunks before deletion
        result = db.execute(text('SELECT COUNT(*) FROM documents'))
        total_chunks_before = result.scalar()
        print(f'Total document chunks in database: {total_chunks_before}')
        
        # Delete all documents
        result = db.execute(text('DELETE FROM documents'))
        deleted_chunks = result.rowcount if hasattr(result, 'rowcount') else 0  # type: ignore
        db.commit()
        
        print(f'Document chunks deleted from database: {deleted_chunks}')
        print('Database cleanup complete!')
        
        return total_deleted_storage, deleted_chunks
    finally:
        db.close()

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Manage Supabase storage and database files')
    parser.add_argument('action', choices=['list', 'delete'], help='Action to perform')
    
    args = parser.parse_args()
    
    if args.action == 'list':
        list_all_files()
        list_database_documents()
    elif args.action == 'delete':
        confirm = input("Are you sure you want to delete ALL files from storage AND database? (yes/no): ")
        if confirm.lower() == 'yes':
            storage_deleted, db_deleted = delete_all_files()
            print(f"\n=== SUMMARY ===")
            print(f"Files deleted from storage: {storage_deleted}")
            print(f"Document chunks deleted from database: {db_deleted}")
            print("Cleanup complete!")
        else:
            print("Operation cancelled.")