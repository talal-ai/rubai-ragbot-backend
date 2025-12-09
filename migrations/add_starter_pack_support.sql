-- ============================================
-- Starter Pack Migration for RubAI
-- Adds support for global documents (starter pack) with user visibility control
-- ============================================
-- Run this in Supabase SQL Editor
-- ============================================

BEGIN;

-- ============================================
-- 1. Add ownership_type column to documents table
-- ============================================

-- Add ownership_type column
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS ownership_type VARCHAR(20) DEFAULT 'personal' NOT NULL;

-- Create index for performance
CREATE INDEX IF NOT EXISTS idx_documents_ownership_type 
ON documents(ownership_type);

COMMENT ON COLUMN documents.ownership_type IS 'Document ownership: global (starter pack) or personal (user upload)';

-- ============================================
-- 2. Create user_document_visibility table
-- ============================================

CREATE TABLE IF NOT EXISTS user_document_visibility (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    is_hidden BOOLEAN DEFAULT FALSE NOT NULL,
    hidden_at TIMESTAMPTZ,
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    UNIQUE(user_id, filename)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_visibility_user_id 
ON user_document_visibility(user_id);

CREATE INDEX IF NOT EXISTS idx_visibility_user_hidden 
ON user_document_visibility(user_id, is_hidden);

COMMENT ON TABLE user_document_visibility IS 'Tracks which users have hidden which global documents';
COMMENT ON COLUMN user_document_visibility.is_hidden IS 'TRUE if user has hidden this global document from their view';

-- ============================================
-- 3. Enable RLS on user_document_visibility
-- ============================================

ALTER TABLE user_document_visibility ENABLE ROW LEVEL SECURITY;

-- Users can only view/modify their own visibility preferences
CREATE POLICY "Users can manage their own visibility preferences" 
ON user_document_visibility
FOR ALL USING (auth.uid() = user_id);

-- ============================================
-- 4. Update RLS policies on documents table
-- ============================================

-- Drop existing policies
DROP POLICY IF EXISTS "Users can view their own documents" ON documents;
DROP POLICY IF EXISTS "Users can delete their own documents" ON documents;

-- New policy: Users can view personal docs + global docs (not hidden)
CREATE POLICY "Users can view their own and global documents" ON documents
    FOR SELECT USING (
        auth.uid() = user_id OR 
        (ownership_type = 'global' AND user_id IS NULL)
    );

-- Keep existing insert policy
-- Users can only insert personal documents (global docs inserted via backend script)

-- New policy: Users can only delete personal documents (not global)
CREATE POLICY "Users can delete only personal documents" ON documents
    FOR DELETE USING (
        auth.uid() = user_id AND 
        ownership_type = 'personal'
    );

-- ============================================
-- 5. Migrate existing data
-- ============================================

-- Mark all existing documents as personal (default behavior)
UPDATE documents 
SET ownership_type = 'personal' 
WHERE ownership_type IS NULL OR ownership_type = '';

-- ============================================
-- 6. Verification
-- ============================================

DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Starter Pack Migration Complete';
    RAISE NOTICE '============================================';
END $$;

-- Check ownership_type column
SELECT 
    COUNT(*) as total_documents,
    COUNT(*) FILTER (WHERE ownership_type = 'personal') as personal_docs,
    COUNT(*) FILTER (WHERE ownership_type = 'global') as global_docs
FROM documents;

-- Check visibility table
SELECT 
    COUNT(*) as total_visibility_records
FROM user_document_visibility;

-- Check indexes
SELECT 
    schemaname,
    tablename,
    indexname
FROM pg_indexes 
WHERE tablename IN ('documents', 'user_document_visibility')
    AND (indexname LIKE '%ownership%' OR indexname LIKE '%visibility%')
ORDER BY tablename, indexname;

COMMIT;

-- ============================================
-- Usage Instructions:
-- ============================================
-- 1. Run this migration in Supabase SQL Editor
-- 2. Upload starter pack files to Supabase Storage (knowledge-base bucket)
-- 3. Run: python backend/init_starter_pack.py
-- 4. Verify global documents exist: SELECT * FROM documents WHERE ownership_type = 'global';
