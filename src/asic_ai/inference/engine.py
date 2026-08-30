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
    """Local model via HuggingFace transformers.

    Real implementation. This previously returned an empty string from
    generate() and a word count from get_token_count(), so a caller wired to it
    saw a model that produced nothing and reported no error -- the same shape as
    the adapter returning zeros.

    Loading is lazy: constructing the engine is cheap and the weights are read
    on the first generate(). On this project's hardware the Vulkan path in
    asic_ai.inference.llama_server is considerably faster; this exists for
    running an unquantised HF checkpoint directly, before any GGUF conversion.
    """

    def __init__(self, model_path: str, device: str = 'auto', dtype: str = 'bfloat16'):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._model = None
        self._tokenizer = None

    def _load(self):
        if self._model is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        dtype_map = {"bfloat16": torch.bfloat16, "float16": torch.float16,
                     "float32": torch.float32, "auto": "auto"}
        resolved = dtype_map.get(self.dtype, "auto")
        # CPU torch has no fast bfloat16 matmul; asking for it is slower than
        # float32, not faster.
        if resolved is torch.bfloat16 and not torch.cuda.is_available():
            resolved = torch.float32

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, trust_remote_code=True)
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token
        # transformers 5.x: the argument is `dtype`, not `torch_dtype`.
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_path, trust_remote_code=True, dtype=resolved)
        self._model.eval()

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        import torch
        self._load()

        # The tokenizer's own chat template, so the system message reaches the
        # model exactly as it was trained. Never re-template by hand.
        text = self._tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        inputs = self._tokenizer(text, return_tensors="pt")
        prompt_tokens = int(inputs["input_ids"].shape[1])

        temperature = float(kwargs.get("temperature", 0.7))
        max_new = int(kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024)))

        gen_kwargs = {
            "max_new_tokens": max_new,
            "pad_token_id": self._tokenizer.pad_token_id,
        }
        if temperature > 0.0:
            gen_kwargs.update(do_sample=True, temperature=temperature,
                              top_p=float(kwargs.get("top_p", 0.95)))
        else:
            gen_kwargs.update(do_sample=False)

        with torch.no_grad():
            out = self._model.generate(**inputs, **gen_kwargs)

        new_tokens = out[0][prompt_tokens:]
        completion = self._tokenizer.decode(new_tokens, skip_special_tokens=True)
        finish = "length" if len(new_tokens) >= max_new else "stop"
        return GenerationResult(
            text=completion,
            prompt_tokens=prompt_tokens,
            completion_tokens=int(len(new_tokens)),
            finish_reason=finish,
        )

    def get_token_count(self, text: str) -> int:
        """Exact, from the model's own tokenizer.

        The previous len(text.split()) undercounts a netlist by roughly 3x, which
        would silently overflow the context window.
        """
        self._load()
        return len(self._tokenizer(text)["input_ids"])

class VLLMEngine(ModelEngine):
    """Local model via vLLM. Requires vLLM, which is not installed here.

    Raises rather than returning an empty string. An engine that silently
    produces nothing looks, to every caller in this repo, exactly like a model
    that declined to answer.

    vLLM has no Windows build. On this machine use
    asic_ai.inference.llama_server (Vulkan on the iGPU) or TransformersEngine.
    """

    def __init__(self, model_path: str, tensor_parallel: int = 1):
        self.model_path = model_path
        self.tensor_parallel = tensor_parallel
        self._llm = None

    def _load(self):
        if self._llm is not None:
            return
        try:
            from vllm import LLM
        except ImportError as exc:
            raise ImportError(
                "VLLMEngine needs vLLM, which is not installed (and has no "
                "Windows build). Use asic_ai.inference.llama_server for the "
                "local iGPU, or TransformersEngine for an HF checkpoint."
            ) from exc
        self._llm = LLM(model=self.model_path,
                        tensor_parallel_size=self.tensor_parallel)

    def generate(self, messages: List[Dict[str, Any]], **kwargs) -> GenerationResult:
        self._load()
        raise NotImplementedError(  # pragma: no cover - no vLLM on this machine
            "vLLM is installed but this engine's generate() is not implemented. "
            "Implement it rather than returning an empty result.")

    def get_token_count(self, text: str) -> int:
        self._load()
        raise NotImplementedError  # pragma: no cover

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
