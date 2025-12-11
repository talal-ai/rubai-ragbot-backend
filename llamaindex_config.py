"""
LlamaIndex configuration for RubAI RAG system
Professional RAG with conversation memory using Gemini or Ollama embeddings and LLM
"""
from llama_index.core import Settings
from llama_index.embeddings.gemini import GeminiEmbedding
import os
from pathlib import Path
from dotenv import load_dotenv
import logging

# Load .env from backend directory
backend_dir = Path(__file__).parent
env_path = backend_dir / ".env"
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

# ==================== EMBEDDING MODEL ====================
# Using Gemini text-embedding-004 (768-dim) for embeddings
# Ollama embeddings can be added when llama_index.embeddings.ollama is available

def get_embed_model():
    """
    Get embedding model for document and query encoding
    
    Returns:
        Embedding model (currently only Gemini supported)
    """
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if llm_provider == "ollama":
        # For now, use Gemini embeddings even with Ollama LLM
        logger.info("Note: Using Gemini embeddings with Ollama LLM")
        logger.info("(llama_index.embeddings.ollama not yet available)")
    
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
    Get LLM for response generation
    
    Returns:
        LLM based on LLM_PROVIDER setting (Gemini or Ollama)
    """
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    if llm_provider == "ollama":
        logger.info("Initializing Ollama LLM (using custom wrapper)...")
        
        ollama_api_key = os.getenv("OLLAMA_API_KEY")
        if not ollama_api_key:
            raise ValueError("OLLAMA_API_KEY not found in environment variables")
        
        ollama_model = os.getenv("OLLAMA_MODEL", "ministral-3:8b-cloud")
        ollama_base_url = os.getenv("OLLAMA_API_BASE_URL", "https://ollama.com/api")
        
        # Create Ollama provider first (outside Pydantic model)
        from llm_provider import create_llm_provider
        ollama_provider = create_llm_provider("ollama", api_key=ollama_api_key, model=ollama_model, base_url=ollama_base_url)
        
        # Create custom LlamaIndex LLM wrapper for Ollama Cloud
        from llama_index.core.llms import CustomLLM, CompletionResponse, CompletionResponseGen, LLMMetadata
        
        # Use closure to capture provider and model_name
        def create_ollama_llm(provider, model_name):
            class OllamaCloudLLM(CustomLLM):
                """Custom LlamaIndex LLM wrapper for Ollama Cloud API"""
                
                def __init__(self, **kwargs):
                    super().__init__(**kwargs)
                    # Store values directly in __dict__ to bypass Pydantic validation
                    object.__setattr__(self, '_provider', provider)
                    object.__setattr__(self, '_model_name', model_name)
                
                @property
                def metadata(self) -> LLMMetadata:
                    # Access via object.__getattribute__ to bypass Pydantic's __getattr__
                    try:
                        model_name = object.__getattribute__(self, '_model_name')
                    except AttributeError:
                        model_name = 'ollama'
                    return LLMMetadata(
                        context_window=8000,
                        num_output=1024,
                        model_name=model_name,
                    )
                
                def complete(self, prompt: str, **kwargs) -> CompletionResponse:
                    """Non-streaming completion"""
                    provider = object.__getattribute__(self, '_provider')
                    response_text = provider.generate_content(prompt, **kwargs)
                    return CompletionResponse(text=response_text)
                
                def stream_complete(self, prompt: str, **kwargs) -> CompletionResponseGen:
                    """Streaming completion"""
                    provider = object.__getattribute__(self, '_provider')
                    def gen() -> CompletionResponseGen:
                        for chunk in provider.generate_content_stream(prompt, **kwargs):
                            yield CompletionResponse(text=chunk, delta=chunk)
                    return gen()
            
            return OllamaCloudLLM
        
        OllamaCloudLLMClass = create_ollama_llm(ollama_provider, ollama_model)
        
        llm = OllamaCloudLLMClass(
            temperature=0.7,
        )
        
        logger.info(f"✅ Ollama LLM initialized: {ollama_model}")
        return llm
    
    # Gemini (default or fallback)
    from llama_index.llms.gemini import Gemini
    logger.info("Initializing Gemini LLM...")
    
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
    - Embedding model (Gemini text-embedding-004)
    - LLM (based on LLM_PROVIDER)
    - Chunking parameters
    - Context window size
    """
    logger.info("Initializing LlamaIndex global settings...")
    
    llm_provider = os.getenv("LLM_PROVIDER", "gemini").lower()
    
    # Set embedding model
    Settings.embed_model = get_embed_model()
    
    # Set LLM
    Settings.llm = get_llm()
    
    # Set chunking parameters (matching current system)
    Settings.chunk_size = 1500
    Settings.chunk_overlap = 250
    
    # Set context window based on provider
    if llm_provider == "ollama":
        Settings.context_window = 8000   # Ollama context window
        Settings.num_output = 1024       # Max output tokens for Ollama
        logger.info("✅ LlamaIndex initialized successfully")
        logger.info("  - Embedding: Gemini text-embedding-004 (768-dim)")
        logger.info(f"  - LLM: Ollama {os.getenv('OLLAMA_MODEL', 'qwen3-coder:480b-cloud')}")
    else:
        Settings.context_window = 32000  # Gemini context window
        Settings.num_output = 2048       # Max output tokens
        logger.info("✅ LlamaIndex initialized successfully")
        logger.info("  - Embedding: Gemini text-embedding-004 (768-dim)")
        logger.info("  - LLM: Gemini 2.5 Flash")
    
    logger.info("  - Chunk size: 1500, Overlap: 250")
    logger.info(f"  - Context window: {Settings.context_window} tokens")

# ==================== CONFIGURATION CONSTANTS ====================

# Embedding dimension (768 for Gemini text-embedding-004)
EMBEDDING_DIM = 768

# Chat memory token limit
CHAT_MEMORY_TOKEN_LIMIT = 3000

# Retrieval parameters
DEFAULT_TOP_K = 5
DEFAULT_SIMILARITY_THRESHOLD = 0.3
