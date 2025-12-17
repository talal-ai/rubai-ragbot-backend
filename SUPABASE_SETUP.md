# 🎉 Supabase Setup Complete!

## ✅ Your Supabase Configuration

**Project URL:** https://tonodshnbuztozteuaon.supabase.co

**Status:** Ready to use!

---

## 📝 Next Steps

### 1. Add Your Gemini API Key

Edit `backend/.env` and replace:
```
GEMINI_API_KEY=your_gemini_api_key_here
```

With your actual Gemini API key.

### 2. Add Your Supabase Database Password

Edit `backend/.env` and replace `YOUR_SUPABASE_PASSWORD` in:
```
DATABASE_URL=postgresql://postgres.tonodshnbuztozteuaon:YOUR_SUPABASE_PASSWORD@aws-0-ap-southeast-1.pooler.supabase.com:6543/postgres
```

**Where to find your password:**
- Go to: https://supabase.com/dashboard/project/tonodshnbuztozteuaon/settings/database
- Look for "Database password" (the one you set when creating the project)

### 3. Run Database Setup SQL

**Option A: Using Supabase Dashboard (Recommended)**
1. Go to: https://supabase.com/dashboard/project/tonodshnbuztozteuaon/sql/new
2. Copy the contents of `backend/supabase_setup.sql`
3. Paste and click "Run"
4. You should see ✅ for both checks

**Option B: Using Python**
```bash
cd backend
python -c "from database import init_db; init_db()"
```

### 4. Install Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 5. Start the Backend

```bash
python main.py
```

You should see:
```
✅ Database initialized successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## 🧪 Test Your Setup

### Test 1: Health Check
```bash
curl http://localhost:8000/health
```

Expected: `{"status":"healthy","service":"RubAI Backend"}`

### Test 2: Upload a Document
Create a test file `test.txt`:
```
This is a test document about artificial intelligence and machine learning.
```

Upload it:
```bash
curl -X POST "http://localhost:8000/documents/upload" \
  -F "file=@test.txt"
```

### Test 3: Search
```bash
curl -X POST "http://localhost:8000/search?query=artificial intelligence&top_k=3"
```

### Test 4: RAG Chat
```bash
curl -X POST "http://localhost:8000/chat/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "message": "What is this document about?",
    "tone": "Professional"
  }'
```

---

## 📊 View Your Data in Supabase

**Table Editor:**
https://supabase.com/dashboard/project/tonodshnbuztozteuaon/editor

You can see all uploaded documents and their embeddings here!

---

## 🔧 Configuration Details

### Connection Pooler
We're using Supabase's connection pooler (port 6543) for better performance:
- **Direct connection:** Port 5432 (limited connections)
- **Pooler connection:** Port 6543 (recommended for apps) ✅

### Database Details
- **Host:** aws-0-ap-southeast-1.pooler.supabase.com
- **Port:** 6543
- **Database:** postgres
- **User:** postgres.tonodshnbuztozteuaon

---

## 🐛 Troubleshooting

**"Could not connect to database"**
- Check your database password in `.env`
- Verify you're connected to the internet
- Check Supabase project is active

**"pgvector extension not found"**
- Run the SQL in `supabase_setup.sql`
- Or run: `CREATE EXTENSION IF NOT EXISTS vector;` in SQL Editor

**"Table does not exist"**
- Run `supabase_setup.sql` in SQL Editor
- Or start the backend (it will auto-create tables)

**"Authentication failed"**
- Double-check your database password
- Get it from: Settings → Database → "Reset database password" if needed

---

## 📚 Useful Links

- **Supabase Dashboard:** https://supabase.com/dashboard/project/tonodshnbuztozteuaon
- **SQL Editor:** https://supabase.com/dashboard/project/tonodshnbuztozteuaon/sql
- **Table Editor:** https://supabase.com/dashboard/project/tonodshnbuztozteuaon/editor
- **Database Settings:** https://supabase.com/dashboard/project/tonodshnbuztozteuaon/settings/database
- **API Docs:** https://supabase.com/dashboard/project/tonodshnbuztozteuaon/api

---

## 🎯 What's Configured

✅ Supabase project connected  
✅ pgvector extension ready  
✅ Connection pooler configured  
✅ API keys stored in `.env`  
✅ Database URL configured  

**You just need to:**
1. Add your Gemini API key
2. Add your database password
3. Run the SQL setup
4. Start the backend!

---

**Ready to go! 🚀**

Once you complete steps 1-3 above, your RAG system will be fully operational!
