"""
LlamaIndex configuration for RubAI RAG system
Professional RAG with conversation memory using Gemini embeddings and Gemini LLM
"""
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from dotenv import load_dotenv
import logging

load_dotenv()
logger = logging.getLogger(__name__)

# ==================== EMBEDDING MODEL ====================
# Using Gemini text-embedding-004 (768-dim)
# - Consistent with document upload embeddings
# - Fast API-based embedding generation
# - Produces 768-dimensional embeddings

def get_embed_model():
    """
    Get embedding model for document and query encoding
    
    Returns:
        GeminiEmbedding: Gemini model for embeddings (768-dim)
    """
    logger.info("Initializing Gemini embedding model (768-dim)...")
    
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    embed_model = GeminiEmbedding(
        model_name="models/text-embedding-004",
        api_key=gemini_api_key,
        embed_batch_size=100,  # Batch processing for efficiency
    )
    
    logger.info("✅ Gemini embedding model initialized (768-dim)")
    return embed_model

# ==================== LLM MODEL ====================

def get_llm():
    """
    Get LLM for response generation using Gemini
    
    Returns:
        Gemini: Configured Gemini model for response generation
    """
    from llama_index.llms.gemini import Gemini
    logger.info("Initializing Gemini LLM...")
    
    # Get Gemini API key from environment
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_api_key:
        raise ValueError("GEMINI_API_KEY not found in environment variables")
    
    llm = Gemini(
        model="models/gemini-2.5-flash",
        api_key=gemini_api_key,
        temperature=0.7,
    )
    
    logger.info(f"✅ Gemini LLM initialized: models/gemini-2.5-flash")
    return llm

# ==================== GLOBAL SETTINGS ====================

def init_llamaindex():
    """
    Initialize LlamaIndex global settings
    
    This configures:
    - Embedding model (Gemini text-embedding-004 768-dim)
    - LLM (Gemini 2.5 Flash)
    - Chunking parameters
    - Context window size
    """
    logger.info("Initializing LlamaIndex global settings...")
    
    # Set embedding model
    Settings.embed_model = get_embed_model()
    
    # Set LLM
    Settings.llm = get_llm()
    
    # Set chunking parameters (matching current system)
    Settings.chunk_size = 1500
    Settings.chunk_overlap = 250
    
    # Set context window for Gemini
    Settings.context_window = 32000  # Gemini context window
    Settings.num_output = 2048       # Max output tokens
    
    logger.info("✅ LlamaIndex initialized successfully")
    logger.info("  - Embedding: Gemini text-embedding-004 (768-dim)")
    logger.info("  - LLM: Gemini 2.5 Flash")
    logger.info("  - Chunk size: 1500, Overlap: 250")
    logger.info("  - Context window: 32000 tokens")

# ==================== CONFIGURATION CONSTANTS ====================

# Embedding dimension (768 for Gemini text-embedding-004)
EMBEDDING_DIM = 768

# Chat memory token limit
CHAT_MEMORY_TOKEN_LIMIT = 3000

# Retrieval parameters
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3
