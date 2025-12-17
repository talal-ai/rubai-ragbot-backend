import google.generativeai as genai
from typing import List
import logging
from config import GEMINI_API_KEY, EMBEDDING_MODEL

logger = logging.getLogger(__name__)

genai.configure(api_key=GEMINI_API_KEY)

# Gemini API batch limits
BATCH_SIZE = 100  # Maximum texts per batch request

def generate_embedding(text: str) -> List[float]:
    """
    Generate embedding for document text using Gemini's embedding model
    
    Args:
        text: Text to generate embedding for
        
    Returns:
        List of 768 float values representing the embedding
    """
    try:
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")
            
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=text.strip(),
            task_type="retrieval_document"
        )
        return result['embedding']
    except ValueError as e:
        logger.error(f"Validation error in embedding: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise RuntimeError(f"Failed to generate embedding: {str(e)}")

def generate_query_embedding(query: str) -> List[float]:
    """
    Generate embedding for a search query
    
    Args:
        query: Search query text
        
    Returns:
        List of 768 float values representing the embedding
    """
    try:
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
            
        result = genai.embed_content(
            model=EMBEDDING_MODEL,
            content=query.strip(),
            task_type="retrieval_query"
        )
        return result['embedding']
    except ValueError as e:
        logger.error(f"Validation error in query embedding: {e}")
        raise
    except Exception as e:
        logger.error(f"Error generating query embedding: {e}")
        raise RuntimeError(f"Failed to generate query embedding: {str(e)}")


def generate_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings for multiple texts in batch (much faster than sequential)
    
    Args:
        texts: List of texts to generate embeddings for
        
    Returns:
        List of embeddings (each embedding is a list of 768 floats)
        
    Note:
        Gemini API supports batching up to 100 texts per request.
        This function automatically handles batching for larger inputs.
    """
    try:
        if not texts:
            return []
        
        # Filter out empty texts
        valid_texts = [text.strip() for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("No valid texts provided")
        
        all_embeddings = []
        total_batches = (len(valid_texts) + BATCH_SIZE - 1) // BATCH_SIZE
        
        logger.info(f"Generating embeddings for {len(valid_texts)} texts in {total_batches} batch(es)")
        
        # Process in batches of BATCH_SIZE
        for i in range(0, len(valid_texts), BATCH_SIZE):
            batch = valid_texts[i:i + BATCH_SIZE]
            batch_num = i // BATCH_SIZE + 1
            
            try:
                logger.debug(f"Processing batch {batch_num}/{total_batches} ({len(batch)} texts)")
                
                result = genai.embed_content(
                    model=EMBEDDING_MODEL,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Handle both single and batch responses
                if isinstance(result['embedding'][0], list):
                    # Batch response: list of embeddings
                    all_embeddings.extend(result['embedding'])
                else:
                    # Single response: one embedding
                    all_embeddings.append(result['embedding'])
                    
            except Exception as e:
                logger.error(f"Error processing batch {batch_num}: {e}")
                raise
        
        logger.info(f"Successfully generated {len(all_embeddings)} embeddings")
        return all_embeddings
        
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Error generating batch embeddings: {e}")
        raise RuntimeError(f"Failed to generate batch embeddings: {str(e)}")
