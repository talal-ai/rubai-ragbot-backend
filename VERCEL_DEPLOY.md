# Deploying to Vercel - FastAPI Backend

## ✅ Setup Complete!

Your FastAPI backend is now configured for Vercel deployment. Here's what was set up:

### Files Created:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless handler using Mangum
- `.vercelignore` - Files to exclude from deployment
- Updated `requirements.txt` - Added `mangum` for serverless support

## 🚀 Deployment Steps

### Option 1: Using Vercel CLI (Recommended)

1. **Install Vercel CLI** (if not already installed):
```bash
npm i -g vercel
```

2. **Login to Vercel**:
```bash
vercel login
```

3. **Deploy**:
```bash
vercel
```

4. **For production**:
```bash
vercel --prod
```

### Option 2: Using GitHub Integration

1. **Push your code to GitHub** (if not already):
```bash
git add .
git commit -m "Configure Vercel deployment"
git push origin main
```

2. **Connect to Vercel**:
   - Go to [vercel.com](https://vercel.com)
   - Click "New Project"
   - Import your GitHub repository
   - Vercel will auto-detect the configuration

3. **Add Environment Variables**:
   - In Vercel dashboard, go to Project Settings → Environment Variables
   - Add all variables from your `.env` file:
     - `GEMINI_API_KEY`
     - `DATABASE_URL`
     - `SUPABASE_URL`
     - `SUPABASE_ANON_KEY`
     - `OLLAMA_API_KEY`
     - `OLLAMA_MODEL`
     - `LLM_PROVIDER`
     - `ALLOWED_ORIGINS`
     - And any other variables you need

4. **Deploy** - Vercel will automatically deploy on every push!

## ⚠️ Important Notes

### Vercel Limitations:
- **Function Timeout**: 
  - Free tier: 10 seconds
  - Pro tier: 60 seconds
  - Enterprise: Up to 300 seconds
- **Cold Starts**: First request may be slower (~1-2 seconds)
- **File Size Limits**: 50MB for serverless functions
- **Memory**: 1GB default (can be increased on Pro)

### For Your RAG Backend:
- ✅ **Works well for**: API endpoints, chat responses, document queries
- ⚠️ **May need optimization for**: 
  - Large file uploads (consider using Supabase Storage directly)
  - Long-running RAG processing (may hit timeout limits)
  - Streaming responses (should work but test thoroughly)

### Database Connections:
- Connection pooling is handled automatically
- Supabase connection pooling works great with Vercel
- Each serverless function invocation reuses connections when possible

## 🔧 Troubleshooting

### If deployment fails:
1. Check that all dependencies are in `requirements.txt`
2. Ensure Python version is compatible (3.11 recommended)
3. Check Vercel build logs for errors

### If API doesn't work:
1. Verify environment variables are set in Vercel dashboard
2. Check function logs in Vercel dashboard
3. Test endpoints using Vercel's function logs

### Performance Tips:
1. Use Supabase connection pooling (already configured)
2. Consider caching for frequently accessed data
3. Optimize database queries
4. Use Vercel's Edge Functions for static responses if needed

## 📝 Environment Variables to Set in Vercel

Make sure to add these in Vercel Dashboard → Settings → Environment Variables:

```
GEMINI_API_KEY=your_key
DATABASE_URL=your_database_url
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_anon_key
OLLAMA_API_KEY=your_ollama_key
OLLAMA_MODEL=qwen3-coder:480b-cloud
LLM_PROVIDER=ollama
ALLOWED_ORIGINS=https://your-frontend.vercel.app,http://localhost:3000
```

## 🎉 You're Ready!

Your backend is configured for Vercel. Just deploy and you're live!

