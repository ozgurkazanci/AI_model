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
    """Remote model via an OpenAI-compatible API.

    Real implementation over urllib, so it adds no dependency. Works against any
    /v1/chat/completions endpoint, including llama.cpp's own server -- see
    asic_ai.inference.llama_server.LlamaServerEngine for the local iGPU variant,
    which adds process lifecycle and exact tokenisation.

    Messages are passed through unchanged. That matters: the system message must
    stay byte-identical to build_system_message(), and any client-side
    re-templating is how training/serving prompt drift gets reintroduced.
    """

    def __init__(self, base_url: str, api_key: str = "", model: str = "local",
                 timeout: int = 300):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.timeout = timeout

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        import json
        import urllib.request

        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024)),
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            data = json.loads(resp.read())

        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return GenerationResult(
            text=(choice.get("message") or {}).get("content", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def get_token_count(self, text: str) -> int:
        """Rough estimate -- a generic endpoint exposes no tokenizer.

        Roughly 4 characters per token. Do NOT use this for context budgeting;
        use a backend that can tokenise exactly (LlamaServerEngine does).
        """
        return max(1, len(text) // 4)
