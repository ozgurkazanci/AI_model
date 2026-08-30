"""Run the fine-tuned model on the local iGPU through llama.cpp's HTTP server.

Path chosen and why:

  torch-directml is installed but broken -- it is built against the torch 2.4.1
  ABI and raises "DLL load failed while importing torch_directml_native" under
  this project's torch 2.5.1+cpu. Downgrading torch would break transformers
  5.16.1 and checkpoint resume, so DirectML is a dead end here.

  llama.cpp's Vulkan backend needs no Python bindings. PyPI ships
  llama-cpp-python only as a source tarball, and building it with Vulkan
  requires MSVC + CMake + the Vulkan SDK, none of which are installed. The
  prebuilt llama.cpp binaries expose an OpenAI-compatible HTTP API instead,
  which is what this module drives.

Everything is probed at runtime. With no binaries and no server present, import
still succeeds and `available()` returns False, so the test suite and every
other code path are unaffected.
"""
from __future__ import annotations

import json
import logging
import os
import subprocess
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

from asic_ai.inference.engine import GenerationResult, ModelEngine

log = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "configs" / "local_inference.yaml"

ENV_DIR = "ASIC_AI_LLAMA_CPP_DIR"
ENV_URL = "ASIC_AI_LLAMA_SERVER_URL"

_DEFAULT_DIRS = [
    Path("C:/Users/ozgur/tools/llamacpp-vulkan"),
    Path.home() / "tools" / "llamacpp-vulkan",
    Path("/opt/llama.cpp/bin"),
]


# ----------------------------------------------------------------- config ---

def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load configs/local_inference.yaml, or {} when it is absent."""
    p = Path(path) if path else CONFIG_PATH
    if not p.exists():
        return {}
    with open(p, encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


def _dir_with_server(d: Path) -> Optional[Path]:
    for exe in ("llama-server.exe", "llama-server"):
        if (d / exe).exists():
            return d
    return None


def find_llama_cpp_dir(config: dict[str, Any] | None = None) -> Optional[Path]:
    """Locate the llama.cpp binaries: env var, then config, then defaults.

    ASIC_AI_LLAMA_CPP_DIR is AUTHORITATIVE: when it is set, only that directory
    is considered, and a directory with no server binary yields None rather than
    falling through. Falling back would make the variable useless for the one
    thing it is most needed for -- reproducing, on a machine that does have the
    binaries, what happens on a machine that does not.
    """
    env = os.environ.get(ENV_DIR)
    if env:
        return _dir_with_server(Path(env))

    cfg = config if config is not None else load_config()
    cand = (cfg.get("llama_cpp") or {}).get("dir")
    for d in ([Path(cand)] if cand else []) + _DEFAULT_DIRS:
        found = _dir_with_server(d)
        if found is not None:
            return found
    return None


def server_binary(config: dict[str, Any] | None = None) -> Optional[Path]:
    d = find_llama_cpp_dir(config)
    if d is None:
        return None
    for exe in ("llama-server.exe", "llama-server"):
        if (d / exe).exists():
            return d / exe
    return None


def available(config: dict[str, Any] | None = None) -> bool:
    """True when a llama.cpp server binary can be found on this machine."""
    return server_binary(config) is not None


def list_devices(config: dict[str, Any] | None = None) -> list[str]:
    """Backends llama.cpp can see, e.g. ['Vulkan0: AMD Radeon 780M ...'].

    Empty when the binaries are missing or the probe fails; never raises.
    """
    d = find_llama_cpp_dir(config)
    if d is None:
        return []
    for exe in ("llama-cli.exe", "llama-cli"):
        cli = d / exe
        if not cli.exists():
            continue
        try:
            r = subprocess.run([str(cli), "--list-devices"], capture_output=True,
                               text=True, timeout=60)
        except (OSError, subprocess.SubprocessError):
            return []
        out = []
        for line in (r.stdout or "").splitlines():
            line = line.strip()
            if line and ":" in line and not line.lower().startswith("available"):
                out.append(line)
        return out
    return []


# ----------------------------------------------------------------- server ---

@dataclass
class ServerConfig:
    """How to launch llama-server. Defaults come from local_inference.yaml."""
    model: str
    host: str = "127.0.0.1"
    port: int = 8231
    context_size: int = 8192
    n_gpu_layers: int = 99
    startup_timeout: int = 90
    env: dict[str, str] = field(default_factory=dict)
    extra_args: list[str] = field(default_factory=list)

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    @classmethod
    def from_config(cls, config: dict[str, Any] | None = None,
                    model: str | None = None) -> "ServerConfig":
        cfg = config if config is not None else load_config()
        srv = cfg.get("server") or {}
        models = cfg.get("models") or {}
        chosen = model or models.get("default") or ""
        if chosen and not Path(chosen).is_absolute():
            chosen = str(REPO_ROOT / chosen)
        return cls(
            model=chosen,
            host=srv.get("host", "127.0.0.1"),
            port=int(srv.get("port", 8231)),
            context_size=int(srv.get("context_size", 8192)),
            n_gpu_layers=int(srv.get("n_gpu_layers", 99)),
            startup_timeout=int(srv.get("startup_timeout", 90)),
            env={str(k): str(v) for k, v in (srv.get("env") or {}).items()},
        )


class LlamaServer:
    """Launch and stop a llama-server process.

    Use as a context manager; the process is always terminated on exit.

        with LlamaServer(ServerConfig.from_config()) as srv:
            engine = LlamaServerEngine(srv.base_url)
    """

    def __init__(self, config: ServerConfig, binary: Path | None = None):
        self.config = config
        self.binary = binary or server_binary()
        self.proc: Optional[subprocess.Popen] = None
        self.log_path: Optional[Path] = None

    @property
    def base_url(self) -> str:
        return self.config.base_url

    def start(self, log_path: str | Path | None = None) -> "LlamaServer":
        if self.binary is None:
            raise FileNotFoundError(
                "llama-server not found. Set "
                f"{ENV_DIR} or llama_cpp.dir in configs/local_inference.yaml."
            )
        if not self.config.model or not Path(self.config.model).exists():
            raise FileNotFoundError(f"GGUF model not found: {self.config.model!r}")

        cmd = [
            str(self.binary),
            "-m", self.config.model,
            "--host", self.config.host,
            "--port", str(self.config.port),
            "-c", str(self.config.context_size),
            "-ngl", str(self.config.n_gpu_layers),
            "--no-webui",
        ] + self.config.extra_args

        env = os.environ.copy()
        env.update(self.config.env)

        self.log_path = Path(log_path) if log_path else None
        stdout = open(self.log_path, "w", encoding="utf-8") if self.log_path else subprocess.DEVNULL

        log.info("starting llama-server on %s (ngl=%d)", self.base_url,
                 self.config.n_gpu_layers)
        self.proc = subprocess.Popen(cmd, stdout=stdout, stderr=subprocess.STDOUT, env=env)

        if not self.wait_until_healthy(self.config.startup_timeout):
            self.stop()
            raise RuntimeError(
                f"llama-server did not become healthy within "
                f"{self.config.startup_timeout}s"
                + (f"; see {self.log_path}" if self.log_path else "")
            )
        return self

    def wait_until_healthy(self, timeout: int) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                return False  # died during startup
            if health(self.base_url, timeout=2):
                return True
            time.sleep(0.5)
        return False

    def stop(self) -> None:
        if self.proc is None:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def __enter__(self) -> "LlamaServer":
        return self.start()

    def __exit__(self, *exc: Any) -> None:
        self.stop()


def health(base_url: str, timeout: int = 5) -> bool:
    """True when a llama.cpp server is answering at base_url."""
    try:
        with urllib.request.urlopen(f"{base_url}/health", timeout=timeout) as r:
            return json.loads(r.read()).get("status") == "ok"
    except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
        return False


# ----------------------------------------------------------------- engine ---

class LlamaServerEngine(ModelEngine):
    """ModelEngine backed by llama.cpp's OpenAI-compatible endpoint.

    Uses urllib so nothing new is added to requirements. The server does the
    chat templating, so `messages` is passed through unchanged -- which matters,
    because the system message must stay byte-identical to the training-time
    output of build_system_message().
    """

    def __init__(self, base_url: str | None = None, model: str = "local",
                 timeout: int = 300):
        self.base_url = (base_url or os.environ.get(ENV_URL)
                         or ServerConfig.from_config().base_url).rstrip("/")
        self.model = model
        self.timeout = timeout

    def _post(self, path: str, payload: dict[str, Any]) -> dict[str, Any]:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())

    def generate(self, messages: list[dict[str, Any]], **kwargs: Any) -> GenerationResult:
        payload: dict[str, Any] = {
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "top_p": kwargs.get("top_p", 0.95),
            "max_tokens": kwargs.get("max_new_tokens", kwargs.get("max_tokens", 1024)),
        }
        if "stop" in kwargs:
            payload["stop"] = kwargs["stop"]

        data = self._post("/v1/chat/completions", payload)
        choice = (data.get("choices") or [{}])[0]
        usage = data.get("usage") or {}
        return GenerationResult(
            text=(choice.get("message") or {}).get("content", ""),
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=choice.get("finish_reason") or "stop",
        )

    def get_token_count(self, text: str) -> int:
        """Exact count from the server's tokenizer, with a crude fallback.

        The fallback is a rough 4-chars-per-token estimate and is only used when
        the endpoint is unreachable; do not use it for context budgeting.
        """
        try:
            data = self._post("/tokenize", {"content": text})
            return len(data.get("tokens") or [])
        except (urllib.error.URLError, OSError, ValueError, json.JSONDecodeError):
            return max(1, len(text) // 4)

    def healthy(self) -> bool:
        return health(self.base_url, timeout=3)
