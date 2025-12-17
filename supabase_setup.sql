-- ============================================
-- RubAI Database Setup for Supabase
-- ============================================
-- Run this in Supabase SQL Editor:
-- https://supabase.com/dashboard/project/YOUR_PROJECT/sql
-- ============================================

-- 1. Enable required extensions
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- ============================================
-- 2. Documents Table (RAG Knowledge Base)
-- ============================================
CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    user_id UUID REFERENCES auth.users(id) ON DELETE CASCADE,
    filename VARCHAR(500) NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    page_number INTEGER,
    embedding VECTOR(768) NOT NULL,  -- Gemini text-embedding-004 dimension
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb NOT NULL,
    
    -- Constraints
    CONSTRAINT documents_chunk_index_positive CHECK (chunk_index >= 0),
    CONSTRAINT documents_filename_not_empty CHECK (length(TRIM(filename)) > 0),
    CONSTRAINT documents_content_not_empty CHECK (length(TRIM(content)) > 0),
    CONSTRAINT documents_unique_user_chunk UNIQUE (user_id, filename, chunk_index)
);

-- Document indexes
CREATE INDEX IF NOT EXISTS idx_documents_user_id ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_documents_filename ON documents(filename);
CREATE INDEX IF NOT EXISTS idx_documents_created_at ON documents(created_at);
CREATE INDEX IF NOT EXISTS idx_documents_user_filename ON documents(user_id, filename);
CREATE INDEX IF NOT EXISTS idx_documents_embedding ON documents 
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);

-- Document comments
COMMENT ON TABLE documents IS 'RAG document chunks with embeddings for semantic search';
COMMENT ON COLUMN documents.embedding IS 'Vector embedding from Gemini text-embedding-004 (768 dimensions)';
COMMENT ON COLUMN documents.chunk_index IS 'Position of chunk in original document (0-based)';
COMMENT ON COLUMN documents.metadata IS 'Additional metadata as JSON (file type, size, etc)';

-- ============================================
-- 3. Chat Sessions Table
-- ============================================
CREATE TABLE IF NOT EXISTS chat_sessions (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    user_id UUID NOT NULL REFERENCES auth.users(id) ON DELETE CASCADE,
    title TEXT DEFAULT 'New Chat' NOT NULL,
    messages JSONB DEFAULT '[]'::jsonb NOT NULL,
    selected_documents TEXT[] DEFAULT '{}',
    created_at TIMESTAMPTZ DEFAULT NOW() NOT NULL,
    updated_at TIMESTAMPTZ DEFAULT NOW() NOT NULL
);

-- Chat session indexes
CREATE INDEX IF NOT EXISTS idx_chat_sessions_user_id ON chat_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated_at ON chat_sessions(updated_at);

-- Chat session comments
COMMENT ON TABLE chat_sessions IS 'Stores user chat history and sessions';
COMMENT ON COLUMN chat_sessions.messages IS 'Array of chat messages in JSONB format';
COMMENT ON COLUMN chat_sessions.selected_documents IS 'List of document filenames selected for this chat context';

-- ============================================
-- 4. Profiles Table (User metadata)
-- ============================================
CREATE TABLE IF NOT EXISTS profiles (
    id UUID PRIMARY KEY REFERENCES auth.users(id) ON DELETE CASCADE,
    email TEXT,
    full_name TEXT,
    avatar_url TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_profiles_email ON profiles(email);

-- ============================================
-- 5. Row Level Security (RLS)
-- ============================================

-- Enable RLS on all tables
ALTER TABLE documents ENABLE ROW LEVEL SECURITY;
ALTER TABLE chat_sessions ENABLE ROW LEVEL SECURITY;
ALTER TABLE profiles ENABLE ROW LEVEL SECURITY;

-- Documents RLS Policies
CREATE POLICY "Users can view their own documents" ON documents
    FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can insert their own documents" ON documents
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can delete their own documents" ON documents
    FOR DELETE USING (auth.uid() = user_id);

-- Chat Sessions RLS Policies
CREATE POLICY "Users can view their own chat sessions" ON chat_sessions
    FOR SELECT USING (auth.uid() = user_id);

CREATE POLICY "Users can insert their own chat sessions" ON chat_sessions
    FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can update their own chat sessions" ON chat_sessions
    FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can delete their own chat sessions" ON chat_sessions
    FOR DELETE USING (auth.uid() = user_id);

-- Profiles RLS Policies
CREATE POLICY "Users can view their own profile" ON profiles
    FOR SELECT USING (auth.uid() = id);

CREATE POLICY "Users can update their own profile" ON profiles
    FOR UPDATE USING (auth.uid() = id);

-- ============================================
-- 6. Auto-create profile on user signup
-- ============================================
CREATE OR REPLACE FUNCTION public.handle_new_user()
RETURNS trigger AS $$
BEGIN
    INSERT INTO public.profiles (id, email, full_name, avatar_url)
    VALUES (
        NEW.id,
        NEW.email,
        NEW.raw_user_meta_data->>'full_name',
        NEW.raw_user_meta_data->>'avatar_url'
    );
    RETURN NEW;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Trigger to auto-create profile
DROP TRIGGER IF EXISTS on_auth_user_created ON auth.users;
CREATE TRIGGER on_auth_user_created
    AFTER INSERT ON auth.users
    FOR EACH ROW EXECUTE FUNCTION public.handle_new_user();

-- ============================================
-- 7. Verify Setup
-- ============================================
SELECT 
    'pgvector extension' as check_name,
    CASE WHEN EXISTS (
        SELECT 1 FROM pg_extension WHERE extname = 'vector'
    ) THEN '✅ Enabled' ELSE '❌ Not enabled' END as status
UNION ALL
SELECT 
    'documents table',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_name = 'documents'
    ) THEN '✅ Created' ELSE '❌ Not created' END
UNION ALL
SELECT 
    'chat_sessions table',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_name = 'chat_sessions'
    ) THEN '✅ Created' ELSE '❌ Not created' END
UNION ALL
SELECT 
    'profiles table',
    CASE WHEN EXISTS (
        SELECT 1 FROM information_schema.tables WHERE table_name = 'profiles'
    ) THEN '✅ Created' ELSE '❌ Not created' END;
