-- Migration to upgrade embedding dimension and add LlamaIndex compatibility
-- Run this after installing LlamaIndex dependencies
-- This prepares the database for BGE-large embeddings (1024-dim)

BEGIN;

-- 1. Add text_id column for LlamaIndex node compatibility
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS text_id VARCHAR(255);

CREATE INDEX IF NOT EXISTS idx_documents_text_id 
ON documents(text_id);

-- 2. Add new embedding column with 1024 dimensions (for BGE-large)
ALTER TABLE documents 
ADD COLUMN IF NOT EXISTS embedding_new VECTOR(1024);

-- 3. Drop old HNSW index
DROP INDEX IF EXISTS documents_embedding_idx;

-- 4. Create new HNSW index for 1024-dim embeddings
-- m=16: number of connections per element (higher = more accurate but slower)
-- ef_construction=64: size of dynamic candidate list (higher = better index quality)
CREATE INDEX IF NOT EXISTS documents_embedding_new_idx 
ON documents 
USING hnsw (embedding_new vector_cosine_ops)
WITH (m = 16, ef_construction = 64);

-- 5. For chat store integration (LlamaIndex PostgresChatStore)
-- Create chat_sessions table if it doesn't exist with proper schema
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'chat_sessions') THEN
        CREATE TABLE chat_sessions (
            id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            key TEXT NOT NULL UNIQUE,  -- LlamaIndex uses 'key' for chat_store_key
            messages JSONB NOT NULL DEFAULT '[]'::jsonb,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            updated_at TIMESTAMPTZ DEFAULT NOW()
        );
        
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_key ON chat_sessions(key);
        CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at);
    END IF;
END $$;

COMMIT;

-- ============================================================
-- POST-MIGRATION STEPS (Run separately after this migration)
-- ============================================================

-- After this migration, you MUST run Python script to:
-- 1. Fetch all existing documents
-- 2. Re-generate embeddings using BGE-large (1024-dim)
-- 3. Update embedding_new column
--
-- Run: python scripts/migrate_embeddings.py

-- ============================================================
-- VERIFICATION QUERIES
-- ============================================================

-- Check if new columns exist
SELECT 
    column_name, 
    data_type,
    CASE 
        WHEN column_name = 'embedding' THEN '768-dim (old)'
        WHEN column_name = 'embedding_new' THEN '1024-dim (BGE-large)'
        ELSE ''
    END as description
FROM information_schema.columns 
WHERE table_name = 'documents' 
    AND column_name IN ('embedding', 'embedding_new', 'text_id')
ORDER BY column_name;

-- Check indexes
SELECT 
    indexname,
    indexdef
FROM pg_indexes 
WHERE tablename = 'documents' 
    AND indexname LIKE '%embedding%'
ORDER BY indexname;

-- Check chat_sessions table
SELECT 
    'chat_sessions' as table_name,
    COUNT(*) as record_count
FROM chat_sessions;
