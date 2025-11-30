-- ============================================
-- LlamaIndex Migration for RubAI
-- Adds support for BGE-large embeddings and conversation memory
-- ============================================
-- Run this in Supabase SQL Editor AFTER backing up your database
-- ============================================

BEGIN;

-- ============================================
-- 1. Add LlamaIndex compatibility columns to documents table
-- ============================================

-- Add text_id column for LlamaIndex node tracking
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS text_id VARCHAR(255);

-- Create index on text_id for faster lookups
CREATE INDEX IF NOT EXISTS idx_documents_text_id 
ON documents(text_id);

-- Add new embedding column for BGE-large (1024 dimensions)
-- keeping old 768-dim embedding for backward compatibility
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS embedding_new VECTOR(1024);

COMMENT ON COLUMN documents.embedding_new IS 'BGE-large embeddings (1024 dimensions) - FREE local model';
COMMENT ON COLUMN documents.text_id IS 'LlamaIndex node ID for document chunk tracking';

-- ============================================
-- 2. Create HNSW index for new BGE-large embeddings
-- ============================================

-- Drop old embedding index if using ivfflat (we'll use HNSW which is faster)
DROP INDEX IF EXISTS idx_documents_embedding;

-- Create new HNSW index for 1024-dim BGE-large embeddings
-- m=16: connections per element (good balance of speed/accuracy)
-- ef_construction=64: build quality (higher = better quality, slower build)
CREATE INDEX IF NOT EXISTS idx_documents_embedding_new 
ON documents 
USING hnsw (embedding_new vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

COMMENT ON INDEX idx_documents_embedding_new IS 'HNSW index for fast similarity search with BGE-large embeddings';

-- ============================================
-- 3. LlamaIndex Chat Store Table
-- ============================================
-- Note: Your existing chat_sessions table is perfect!
-- LlamaIndex can use it directly with a simple adapter
-- No changes needed here.

-- Optionally add an index for chat history lookups if not exists
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_updated 
ON chat_sessions(user_id, updated_at DESC);

COMMENT ON TABLE chat_sessions IS 'Stores chat history - compatible with LlamaIndex PostgresChatStore';

-- ============================================
-- 4. Verification Queries
-- ============================================

-- Check new columns exist
DO $$
BEGIN
    RAISE NOTICE '============================================';
    RAISE NOTICE 'Migration Verification';
    RAISE NOTICE '============================================';
END $$;

SELECT 
    column_name,
    data_type,
    CASE 
        WHEN column_name = 'embedding' THEN '768-dim (old Gemini)'
        WHEN column_name = 'embedding_new' THEN '1024-dim (BGE-large FREE)'
        WHEN column_name = 'text_id' THEN 'LlamaIndex node ID'
        WHEN column_name = 'user_id' THEN 'Supabase auth UUID'
        ELSE ''
    END as description
FROM information_schema.columns 
WHERE table_name = 'documents' 
    AND column_name IN ('user_id', 'embedding', 'embedding_new', 'text_id')
ORDER BY column_name;

-- Check indexes
SELECT 
    indexname,
    CASE 
        WHEN indexname LIKE '%embedding_new%' THEN '✅ New HNSW index for BGE-large'
        WHEN indexname LIKE '%text_id%' THEN '✅ LlamaIndex node ID index'
        WHEN indexname LIKE '%user%' THEN '✅ User isolation index'
        ELSE ''
    END as description
FROM pg_indexes 
WHERE tablename = 'documents' 
    AND (indexname LIKE '%text_id%' OR indexname LIKE '%embedding%' OR indexname LIKE '%user%')
ORDER BY indexname;

-- Check table sizes
SELECT 
    'documents' as table_name,
    COUNT(*) as total_chunks,
    COUNT(embedding_new) as chunks_with_bge_embeddings,
    COUNT(text_id) as chunks_with_llamaindex_id,
    pg_size_pretty(pg_total_relation_size('documents')) as table_size
FROM documents;

COMMIT;

-- ============================================
-- Post-Migration Notes
-- ============================================

/*
✅ MIGRATION COMPLETE!

What changed:
1. Added 'text_id' column for LlamaIndex node tracking
2. Added 'embedding_new' column for BGE-large embeddings (1024-dim)
3. Created HNSW index for fast vector search
4. Added helpful indexes for chat history

What's next:
1. New document uploads will automatically use BGE-large embeddings
2. Old documents still work with their 768-dim embeddings
3. You can optionally re-generate embeddings for old documents later

BGE-large Benefits:
- ✅ 100% FREE (runs locally)
- ✅ FAST (no API calls)
- ✅ PRIVATE (data stays on your server)
- ✅ NO RATE LIMITS
- ✅ Better accuracy than Gemini embeddings

Your system is now ready for production with conversation memory!
*/
