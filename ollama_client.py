"""
Ollama Cloud API Client
Handles HTTP requests to Ollama Cloud API for text generation
"""
import requests
import json
import logging
from typing import Optional, Iterator
import time

logger = logging.getLogger(__name__)


class OllamaClient:
    """Client for interacting with Ollama Cloud API"""
    
    def __init__(self, api_key: str, base_url: str = "https://ollama.com/api"):
        """
        Initialize Ollama Cloud client
        
        Args:
            api_key: Ollama Cloud API key
            base_url: Base URL for Ollama Cloud API (default: https://ollama.com/api)
        """
        self.api_key = api_key.strip()
        self.base_url = base_url.strip().rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        logger.info(f"Ollama Cloud client initialized with base URL: {base_url}")
    
    def generate(
        self,
        prompt: str,
        model: str,
        system_instruction: Optional[str] = None,
        stream: bool = False,
        max_retries: int = 3
    ) -> str:
        """
        Generate text using Ollama Cloud API (non-streaming)
        
        Args:
            prompt: User prompt
            model: Model name (e.g., 'qwen3-coder:480b-cloud')
            system_instruction: Optional system instruction
            stream: Whether to stream (False for this method)
            max_retries: Maximum number of retry attempts
            
        Returns:
            Generated text response
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": False
        }
        
        # Add system instruction if provided
        if system_instruction:
            payload["system"] = system_instruction
        
        for attempt in range(max_retries):
            try:
                logger.debug(f"Ollama API request (attempt {attempt + 1}): model={model}, prompt_length={len(prompt)}")
                
                response = requests.post(
                    url,
                    headers=self.headers,
                    json=payload,
                    timeout=120  # 2 minute timeout for large models
                )
                
                # Handle different status codes
                if response.status_code == 200:
                    data = response.json()
                    if "response" in data:
                        generated_text = data["response"]
                        logger.debug(f"Ollama API response received: {len(generated_text)} characters")
                        return generated_text
                    else:
                        raise ValueError(f"Unexpected response format: {data}")
                
                elif response.status_code == 401:
                    error_msg = "Ollama Cloud authentication failed. Please check your API key."
                    logger.error(error_msg)
                    raise AuthenticationError(error_msg)
                
                elif response.status_code == 429:
                    retry_after = response.headers.get("Retry-After", "60")
                    error_msg = f"Rate limit exceeded. Retry after {retry_after} seconds."
                    logger.warning(error_msg)
                    if attempt < max_retries - 1:
                        wait_time = int(retry_after) if retry_after.isdigit() else 60
                        time.sleep(wait_time)
                        continue
                    raise RateLimitError(error_msg)
                
                elif response.status_code == 400:
                    error_data = response.json() if response.content else {}
                    error_msg = error_data.get("error", f"Bad request: {response.text}")
                    logger.error(f"Ollama API error: {error_msg}")
                    raise ValueError(f"Invalid request: {error_msg}")
                
                else:
                    error_msg = f"Ollama API error: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    if attempt < max_retries - 1:
                        wait_time = 2 ** attempt  # Exponential backoff
                        time.sleep(wait_time)
                        continue
                    raise RuntimeError(error_msg)
            
            except requests.exceptions.Timeout:
                error_msg = "Ollama API request timed out"
                logger.error(error_msg)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise TimeoutError(error_msg)
            
            except requests.exceptions.ConnectionError as e:
                error_msg = f"Connection error to Ollama API: {str(e)}"
                logger.error(error_msg)
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue
                raise ConnectionError(error_msg)
            
            except (AuthenticationError, RateLimitError, ValueError):
                # Don't retry these errors
                raise
        
        raise RuntimeError(f"Failed to generate content after {max_retries} attempts")
    
    def generate_stream(
        self,
        prompt: str,
        model: str,
        system_instruction: Optional[str] = None
    ) -> Iterator[str]:
        """
        Generate text with streaming using Ollama Cloud API
        
        Args:
            prompt: User prompt
            model: Model name (e.g., 'qwen3-coder:480b-cloud')
            system_instruction: Optional system instruction
            
        Yields:
            Text chunks as they are generated
        """
        url = f"{self.base_url}/generate"
        
        payload = {
            "model": model,
            "prompt": prompt,
            "stream": True
        }
        
        # Add system instruction if provided
        if system_instruction:
            payload["system"] = system_instruction
        
        try:
            logger.debug(f"Ollama API streaming request: model={model}, prompt_length={len(prompt)}")
            
            response = requests.post(
                url,
                headers=self.headers,
                json=payload,
                stream=True,
                timeout=120
            )
            
            if response.status_code == 200:
                for line in response.iter_lines():
                    if line:
                        try:
                            data = json.loads(line.decode('utf-8'))
                            if "response" in data:
                                chunk = data["response"]
                                if chunk:  # Only yield non-empty chunks
                                    yield chunk
                            
                            # Check if done
                            if data.get("done", False):
                                break
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse JSON line: {line}")
                            continue
            
            elif response.status_code == 401:
                error_msg = "Ollama Cloud authentication failed. Please check your API key."
                logger.error(error_msg)
                raise AuthenticationError(error_msg)
            
            elif response.status_code == 429:
                retry_after = response.headers.get("Retry-After", "60")
                error_msg = f"Rate limit exceeded. Retry after {retry_after} seconds."
                logger.warning(error_msg)
                raise RateLimitError(error_msg)
            
            else:
                error_text = response.text if hasattr(response, 'text') else "Unknown error"
                error_msg = f"Ollama API error: {response.status_code} - {error_text}"
                logger.error(error_msg)
                raise RuntimeError(error_msg)
        
        except requests.exceptions.Timeout:
            error_msg = "Ollama API streaming request timed out"
            logger.error(error_msg)
            raise TimeoutError(error_msg)
        
        except requests.exceptions.ConnectionError as e:
            error_msg = f"Connection error to Ollama API: {str(e)}"
            logger.error(error_msg)
            raise ConnectionError(error_msg)


class AuthenticationError(Exception):
    """Raised when authentication fails"""
    pass


class RateLimitError(Exception):
    """Raised when rate limit is exceeded"""
    pass

