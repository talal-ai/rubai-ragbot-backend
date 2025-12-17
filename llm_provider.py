"""
LLM Provider Abstraction Layer
Supports multiple LLM providers (Gemini, Ollama Cloud) with unified interface
"""
from abc import ABC, abstractmethod
from typing import AsyncIterator, Iterator, Optional
import logging

logger = logging.getLogger(__name__)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate_content(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        """
        Generate content synchronously
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            **kwargs: Additional provider-specific parameters
            
        Returns:
            Generated text response
        """
        pass
    
    @abstractmethod
    def generate_content_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> Iterator[str]:
        """
        Generate content with streaming
        
        Args:
            prompt: The user prompt
            system_instruction: Optional system instruction
            **kwargs: Additional provider-specific parameters
            
        Yields:
            Text chunks as they are generated
        """
        pass


class GeminiProvider(LLMProvider):
    """Gemini LLM Provider implementation"""
    
    def __init__(self, model, api_key: str):
        """
        Initialize Gemini provider
        
        Args:
            model: Gemini model instance (GenerativeModel)
            api_key: Gemini API key
        """
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        self.model = model
        logger.info("Gemini provider initialized")
    
    def generate_content(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        """Generate content using Gemini"""
        try:
            # Gemini handles system instruction via model initialization
            # If system_instruction is provided in kwargs, it's already set in model
            response = self.model.generate_content(prompt, **kwargs)
            return response.text
        except Exception as e:
            logger.error(f"Gemini generation error: {e}")
            raise RuntimeError(f"Failed to generate content with Gemini: {str(e)}")
    
    def generate_content_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Generate streaming content using Gemini"""
        try:
            response = self.model.generate_content(prompt, stream=True, **kwargs)
            for chunk in response:
                if chunk.text:
                    yield chunk.text
        except Exception as e:
            logger.error(f"Gemini streaming error: {e}")
            raise RuntimeError(f"Failed to stream content with Gemini: {str(e)}")


class OllamaCloudProvider(LLMProvider):
    """Ollama Cloud LLM Provider implementation"""
    
    def __init__(self, api_key: str, model: str, base_url: str = "https://ollama.com/api"):
        """
        Initialize Ollama Cloud provider
        
        Args:
            api_key: Ollama Cloud API key
            model: Model name (e.g., 'qwen3-coder:480b-cloud')
            base_url: Ollama Cloud API base URL
        """
        from ollama_client import OllamaClient
        self.client = OllamaClient(api_key=api_key, base_url=base_url)
        self.model = model
        self.system_instruction = None
        logger.info(f"Ollama Cloud provider initialized with model: {model}")
    
    def set_system_instruction(self, instruction: str):
        """Set system instruction for the provider"""
        self.system_instruction = instruction
    
    def generate_content(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> str:
        """Generate content using Ollama Cloud"""
        try:
            # Use provided system_instruction or fallback to instance-level one
            sys_instruction = system_instruction or self.system_instruction
            response = self.client.generate(
                prompt=prompt,
                model=self.model,
                system_instruction=sys_instruction,
                stream=False
            )
            return response
        except Exception as e:
            logger.error(f"Ollama Cloud generation error: {e}")
            raise RuntimeError(f"Failed to generate content with Ollama Cloud: {str(e)}")
    
    def generate_content_stream(self, prompt: str, system_instruction: Optional[str] = None, **kwargs) -> Iterator[str]:
        """Generate streaming content using Ollama Cloud"""
        try:
            # Use provided system_instruction or fallback to instance-level one
            sys_instruction = system_instruction or self.system_instruction
            for chunk in self.client.generate_stream(
                prompt=prompt,
                model=self.model,
                system_instruction=sys_instruction
            ):
                yield chunk
        except Exception as e:
            logger.error(f"Ollama Cloud streaming error: {e}")
            raise RuntimeError(f"Failed to stream content with Ollama Cloud: {str(e)}")


def create_llm_provider(provider_type: str, **kwargs) -> LLMProvider:
    """
    Factory function to create LLM provider instances
    
    Args:
        provider_type: "gemini" or "ollama"
        **kwargs: Provider-specific initialization parameters
        
    Returns:
        LLMProvider instance
    """
    provider_type = provider_type.lower()
    
    if provider_type == "gemini":
        from config import GEMINI_API_KEY, GEMINI_MODEL
        import google.generativeai as genai
        
        api_key = kwargs.get("api_key") or GEMINI_API_KEY
        model_name = kwargs.get("model") or GEMINI_MODEL
        system_instruction = kwargs.get("system_instruction")
        
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(
            model_name=model_name,
            system_instruction=system_instruction
        )
        return GeminiProvider(model=model, api_key=api_key)
    
    elif provider_type == "ollama":
        from config import OLLAMA_API_KEY, OLLAMA_MODEL, OLLAMA_API_BASE_URL
        
        api_key = kwargs.get("api_key") or OLLAMA_API_KEY
        model = kwargs.get("model") or OLLAMA_MODEL
        base_url = kwargs.get("base_url") or OLLAMA_API_BASE_URL
        system_instruction = kwargs.get("system_instruction")
        
        if not api_key:
            raise ValueError("OLLAMA_API_KEY is required for Ollama provider")
        
        provider = OllamaCloudProvider(
            api_key=api_key,
            model=model,
            base_url=base_url
        )
        
        if system_instruction:
            provider.set_system_instruction(system_instruction)
        
        return provider
    
    else:
        raise ValueError(f"Unknown provider type: {provider_type}. Supported: 'gemini', 'ollama'")

