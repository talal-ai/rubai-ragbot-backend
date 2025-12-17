# 🎉 RAG System Complete!

## What We Built

A **complete RAG (Retrieval-Augmented Generation) system** using:
- ✅ **PostgreSQL + pgvector** for vector storage
- ✅ **Gemini embeddings** for semantic search
- ✅ **Document processing** (PDF & TXT support)
- ✅ **Intelligent chunking** with LangChain
- ✅ **Streaming RAG responses**

---

## 🏗️ Architecture

```
User uploads PDF → Extract text → Chunk text → Generate embeddings → Store in PostgreSQL

User asks question → Generate query embedding → Search similar chunks → Send to Gemini with context → Stream response
```

---

## 📁 New Files Created

### Backend Core
- `database.py` - PostgreSQL + pgvector setup
- `embeddings.py` - Gemini embedding generation
- `document_processor.py` - PDF extraction & text chunking
- `vector_store.py` - Vector search & document management
- `main.py` - Updated with RAG endpoints

### Documentation
- `POSTGRES_SETUP.md` - Database setup guide
- `RAG_GUIDE.md` - This file

---

## 🚀 Setup Instructions

### 1. Install PostgreSQL with pgvector

**Easiest way (Docker):**
```bash
docker run -d \
  --name rubai-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rubai_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

See `POSTGRES_SETUP.md` for other options.

### 2. Update Environment Variables

Edit `backend/.env`:
```
GEMINI_API_KEY=your_actual_api_key
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/rubai_db
```

### 3. Install Python Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 4. Start the Backend

```bash
python main.py
```

The database will auto-initialize on first run!

---

## 📡 API Endpoints

### Document Management

**Upload Document**
```bash
POST /documents/upload
Content-Type: multipart/form-data

file: <PDF or TXT file>
```

**List Documents**
```bash
GET /documents
```

**Delete Document**
```bash
DELETE /documents/{filename}
```

**Search Documents**
```bash
POST /search?query=your question&top_k=5
```

### RAG Chat

**Chat with RAG (Non-streaming)**
```bash
POST /chat/rag
{
  "message": "What is this document about?",
  "tone": "Professional"
}
```

**Chat with RAG (Streaming)**
```bash
POST /chat/rag/stream
{
  "message": "Explain the main points",
  "tone": "Brief"
}
```

---

## 🧪 Testing the RAG System

### 1. Upload a Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@your_document.pdf"
```

### 2. Search for Content

```bash
curl -X POST "http://localhost:8000/search?query=machine learning&top_k=3"
```

### 3. Ask Questions

```bash
curl -X POST "http://localhost:8000/chat/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What are the key concepts?",
    "tone": "Professional"
  }'
```

---

## 🎯 How It Works

### Document Upload Flow
1. User uploads PDF/TXT
2. Extract text from document
3. Split into chunks (1000 chars, 200 overlap)
4. Generate embeddings for each chunk (Gemini)
5. Store chunks + embeddings in PostgreSQL

### RAG Query Flow
1. User asks a question
2. Generate embedding for question
3. Search PostgreSQL for similar chunks (cosine similarity)
4. Retrieve top 3 most relevant chunks
5. Build context with chunks + question
6. Send to Gemini for answer
7. Stream response back to user

---

## 🔧 Configuration

### Chunking Parameters
Edit `document_processor.py`:
```python
chunk_size=1000,      # Max characters per chunk
chunk_overlap=200     # Overlap between chunks
```

### Search Parameters
Edit `vector_store.py`:
```python
top_k=5,                    # Number of results
similarity_threshold=0.5    # Minimum similarity (0-1)
```

### Embedding Model
Edit `embeddings.py`:
```python
model="models/text-embedding-004"  # Gemini embedding model
```

---

## 📊 Database Schema

```sql
CREATE TABLE documents (
    id SERIAL PRIMARY KEY,
    filename VARCHAR NOT NULL,
    content TEXT NOT NULL,
    chunk_index INTEGER NOT NULL,
    embedding VECTOR(768),  -- Gemini embedding dimension
    created_at TIMESTAMP DEFAULT NOW(),
    metadata TEXT
);

CREATE INDEX ON documents USING ivfflat (embedding vector_cosine_ops);
```

---

## 🎨 Next Steps

### Frontend Integration
1. Add document upload UI
2. Show document list
3. Display sources with answers
4. Add delete document button

### Enhancements
- [ ] Support more file types (DOCX, CSV)
- [ ] Add metadata filtering
- [ ] Implement hybrid search (keyword + semantic)
- [ ] Add document preview
- [ ] Cache embeddings
- [ ] Add user authentication
- [ ] Multi-user document isolation

---

## 🐛 Troubleshooting

**Database connection error?**
- Check if PostgreSQL is running
- Verify DATABASE_URL in `.env`
- Run: `docker ps` to see if container is up

**Embedding generation slow?**
- Normal for first time (cold start)
- Consider caching embeddings
- Use batch processing for multiple docs

**Out of memory?**
- Reduce chunk_size
- Process documents in batches
- Use smaller embedding model

**Search returns no results?**
- Lower similarity_threshold
- Check if documents are uploaded
- Verify embeddings are generated

---

## 📚 Tech Stack

- **FastAPI** - Web framework
- **PostgreSQL** - Database
- **pgvector** - Vector similarity search
- **SQLAlchemy** - ORM
- **Gemini** - Embeddings + LLM
- **pypdf** - PDF processing
- **LangChain** - Text splitting

---

**Status:** ✅ **RAG System Ready!**

Upload documents and start asking questions! 🚀
