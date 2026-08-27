from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class GenerationResult(BaseModel):
    """Result of a model generation call."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str = "stop"

class ModelEngine(ABC):
    """Abstract base class for all model backends."""
    
    @abstractmethod
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        """Generate response given a list of messages."""
        pass
        
    @abstractmethod
    def get_token_count(self, text: str) -> int:
        """Count the number of tokens in the given text."""
        pass

class TransformersEngine(ModelEngine):
    """Local model via HuggingFace transformers."""
    
    def __init__(self, model_path: str, device: str = 'auto', dtype: str = 'bfloat16'):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        return GenerationResult(text="", prompt_tokens=0, completion_tokens=0)
        
    def get_token_count(self, text: str) -> int:
        return len(text.split())

class VLLMEngine(ModelEngine):
    """Local model via vLLM for high-throughput inference."""
    
    def __init__(self, model_path: str, tensor_parallel: int = 1):
        self.model_path = model_path
        self.tensor_parallel = tensor_parallel
        
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        return GenerationResult(text="", prompt_tokens=0, completion_tokens=0)
        
    def get_token_count(self, text: str) -> int:
        return len(text.split())

class APIEngine(ModelEngine):
    """Remote model via OpenAI-compatible API."""
    
    def __init__(self, base_url: str, api_key: str, model: str):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        
    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        return GenerationResult(text="", prompt_tokens=0, completion_tokens=0)
        
    def get_token_count(self, text: str) -> int:
        return len(text.split())
