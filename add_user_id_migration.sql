-- Migration to add user_id column to documents table for user isolation
-- Run this script to update your database schema

BEGIN;

-- Add user_id column (nullable for existing documents)
ALTER TABLE documents ADD COLUMN IF NOT EXISTS user_id VARCHAR(255);

-- Create index on user_id for performance
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);

-- Create composite index for user_id and filename
CREATE INDEX IF NOT EXISTS idx_documents_user_filename ON documents(user_id, filename);

-- Drop old unique constraint
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_unique_chunk;

-- Add new unique constraint that includes user_id
ALTER TABLE documents ADD CONSTRAINT documents_unique_user_chunk 
    UNIQUE (user_id, filename, chunk_index);

-- Drop old index
DROP INDEX IF EXISTS idx_documents_filename_created;

COMMIT;

-- Note: Existing documents will have NULL user_id and will be accessible to all users
-- To assign existing documents to a specific user, run:
-- UPDATE documents SET user_id = 'USER_ID_HERE' WHERE user_id IS NULL;
