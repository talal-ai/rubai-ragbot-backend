# RubAI Backend - Professional RAG Chatbot System

A production-ready FastAPI backend with RAG (Retrieval-Augmented Generation) capabilities powered by Google Gemini and PostgreSQL with pgvector.

## 🚀 Features

- ✅ **RAG System**: Upload documents (PDF/TXT) and chat with intelligent context
- ✅ **Vector Search**: Fast semantic search using pgvector with HNSW indexing
- ✅ **Streaming Responses**: Real-time AI responses with Server-Sent Events
- ✅ **File Validation**: Secure file uploads with type, size, and content validation
- ✅ **Error Handling**: Professional error handling with detailed logging
- ✅ **Input Validation**: Pydantic models with comprehensive validation
- ✅ **CORS Security**: Configurable allowed origins
- ✅ **Connection Pooling**: Optimized database connection management
- ✅ **Duplicate Prevention**: Unique constraints prevent duplicate document chunks

## 📋 Prerequisites

- Python 3.8 or higher
- PostgreSQL with pgvector extension (or Supabase account)
- Google Gemini API key

## 🔧 Setup

### 1. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy the example environment file:

```bash
cp .env.example .env
```

Edit `.env` and add your credentials:

```env
# Required: Your Gemini API key
GEMINI_API_KEY=your_gemini_api_key_here

# Required: Database connection string
# For Supabase:
DATABASE_URL=postgresql://postgres.YOUR_PROJECT:YOUR_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres

# Optional: CORS configuration (comma-separated)
ALLOWED_ORIGINS=http://localhost:3000,http://localhost:5173
```

### 3. Database Setup

#### Option A: Using Supabase (Recommended)

The database schema is automatically managed. Just ensure:
1. You have a Supabase account
2. pgvector extension is enabled (should be automatic)
3. Your DATABASE_URL is correctly configured

#### Option B: Local PostgreSQL

```bash
# Install PostgreSQL with pgvector
docker run -d \
  --name rubai-postgres \
  -e POSTGRES_PASSWORD=postgres \
  -e POSTGRES_DB=rubai_db \
  -p 5432:5432 \
  pgvector/pgvector:pg16
```

## 🎯 Running the Backend

```bash
python main.py
```

The server will start at `http://localhost:8000`

- **API Documentation**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/health

## 📡 API Endpoints

### Health & Status

- `GET /` - Root endpoint
- `GET /health` - Health check with database status

### Chat Endpoints

- `POST /chat` - Non-streaming chat
- `POST /chat/stream` - Streaming chat with SSE
- `POST /chat/rag` - RAG chat (with document context)
- `POST /chat/rag/stream` - Streaming RAG chat

### Document Management

- `POST /documents/upload` - Upload PDF or TXT file
- `GET /documents` - List all uploaded documents
- `DELETE /documents/{filename}` - Delete a document
- `POST /search` - Search document chunks by semantic similarity

## 📝 Usage Examples

### Upload a Document

```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@document.pdf"
```

### Chat with RAG

```bash
curl -X POST "http://localhost:8000/chat/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "tone": "Professional"
  }'
```

### Search Documents

```bash
curl -X POST "http://localhost:8000/search?query=artificial intelligence&top_k=5"
```

## 🏗️ Project Structure

```
backend/
├── main.py                 # FastAPI application & endpoints
├── config.py              # Configuration settings
├── database.py            # Database models & connection
├── embeddings.py          # Gemini embedding generation
├── document_processor.py  # PDF/TXT processing & chunking
├── vector_store.py        # Vector search & storage
├── requirements.txt       # Python dependencies
├── .env                   # Environment variables (not in git)
└── .env.example          # Example environment file
```

## 🔒 Security Features

- **File Validation**: Maximum 10MB, only PDF/TXT allowed
- **MIME Type Checking**: Validates actual file content
- **Filename Sanitization**: Prevents path traversal attacks
- **Input Validation**: Pydantic models validate all inputs
- **CORS Protection**: Configurable allowed origins
- **SQL Injection Prevention**: Parameterized queries
- **Error Message Sanitization**: No sensitive data in error responses

## 📊 Database Schema

### `documents` Table

| Column | Type | Description |
|--------|------|-------------|
| id | BIGSERIAL | Primary key |
| filename | VARCHAR(500) | Document filename |
| content | TEXT | Chunk text content |
| chunk_index | INTEGER | Position in document |
| embedding | VECTOR(768) | Gemini embedding |
| created_at | TIMESTAMPTZ | Upload timestamp |
| metadata | JSONB | Additional metadata |

**Indexes:**
- HNSW index on embedding for fast similarity search
- B-tree indexes on filename, created_at
- Unique constraint on (filename, chunk_index)

## ⚙️ Configuration

Edit `config.py` to adjust:

- `MAX_FILE_SIZE`: Maximum upload size (default: 10MB)
- `DEFAULT_CHUNK_SIZE`: Text chunk size (default: 1000 chars)
- `DEFAULT_CHUNK_OVERLAP`: Chunk overlap (default: 200 chars)
- `SIMILARITY_THRESHOLD`: Minimum similarity for search (default: 0.5)
- `DB_POOL_SIZE`: Database connection pool size (default: 10)

## 🐛 Troubleshooting

### Database Connection Error

```
❌ Database connection error: could not connect to server
```

**Solution**: Check your DATABASE_URL in `.env` file and ensure database is running.

### Gemini API Error

```
Failed to generate response
```

**Solution**: Verify your GEMINI_API_KEY in `.env` is valid and has sufficient quota.

### File Upload Error

```
File type not allowed
```

**Solution**: Only PDF and TXT files are supported. Check file extension and MIME type.

### Import Error

```
ModuleNotFoundError: No module named 'config'
```

**Solution**: Ensure you're running from the backend directory and all dependencies are installed.

## 📈 Performance

- **Connection Pooling**: 10 persistent connections + 20 overflow
- **HNSW Indexing**: Fast approximate nearest neighbor search
- **Streaming Responses**: Low latency with Server-Sent Events
- **Efficient Chunking**: LangChain's RecursiveCharacterTextSplitter

## 🔄 Production Deployment

### Environment Variables for Production

```env
GEMINI_API_KEY=your_production_key
DATABASE_URL=your_production_database_url
ALLOWED_ORIGINS=https://your-frontend-domain.com
```

### Recommended Hosting

- **Backend**: Railway, Render, Fly.io, AWS EC2
- **Database**: Supabase, AWS RDS, Railway PostgreSQL
- **Monitoring**: Add logging service (Sentry, LogRocket)

### Docker Deployment (Optional)

```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
CMD ["python", "main.py"]
```

## 📚 API Documentation

Full interactive API documentation available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## 🤝 Support

For issues or questions:
1. Check the troubleshooting section
2. Review API documentation
3. Check application logs

## 📄 License

Proprietary - RubAI System

---

**Version**: 1.0.0  
**Last Updated**: 2024
