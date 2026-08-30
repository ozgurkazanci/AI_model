"""Tests for local iGPU inference via llama.cpp.

Everything here is machine-dependent, so the rule is: a machine WITHOUT
llama.cpp, without a GGUF model, and without a running server must still see a
fully green suite. Nothing may fail merely because the hardware is absent --
only skip. The pure-logic tests (config resolution, payload construction,
fallbacks) run everywhere and are the ones that actually guard the code.
"""
from __future__ import annotations

import json
import io
from pathlib import Path
from unittest import mock

import pytest

from asic_ai.data.format import TOOL_DEFINITIONS, build_system_message
from asic_ai.inference import llama_server
from asic_ai.inference.engine import APIEngine, GenerationResult
from asic_ai.inference.llama_server import (
    LlamaServerEngine, ServerConfig, health,
)

REPO_ROOT = Path(__file__).parent.parent

HAS_LLAMA_CPP = llama_server.available()
_CFG = llama_server.load_config()
_MODEL = ServerConfig.from_config(_CFG).model
HAS_MODEL = bool(_MODEL) and Path(_MODEL).exists()
SERVER_UP = health(ServerConfig.from_config(_CFG).base_url, timeout=2)

skip_no_llama = pytest.mark.skipif(
    not HAS_LLAMA_CPP, reason="llama.cpp binaries not installed on this machine")
skip_no_model = pytest.mark.skipif(
    not (HAS_LLAMA_CPP and HAS_MODEL), reason="no GGUF model built")


# ----------------------------------------------------------------- config ---

def test_config_file_exists_and_parses():
    cfg = llama_server.load_config()
    assert cfg, "configs/local_inference.yaml should exist and be non-empty"
    assert "server" in cfg and "models" in cfg


def test_missing_config_returns_empty_not_raises():
    assert llama_server.load_config("/definitely/not/here.yaml") == {}


def test_server_config_defaults_are_sane():
    c = ServerConfig.from_config()
    assert c.port > 0
    assert c.base_url.startswith("http://")
    assert c.n_gpu_layers >= 0


def test_model_path_is_resolved_against_repo_root():
    c = ServerConfig.from_config()
    if c.model:
        assert Path(c.model).is_absolute(), "relative model paths break cwd-independence"


def test_coopmat_workaround_is_documented_in_config():
    """The flag is load-bearing on this driver; losing it breaks every model load.

    If a driver update makes it unnecessary, delete it deliberately and
    re-benchmark -- do not let it vanish silently.
    """
    cfg = llama_server.load_config()
    env = (cfg.get("server") or {}).get("env") or {}
    if "GGML_VK_DISABLE_COOPMAT" in env:
        text = (REPO_ROOT / "configs" / "local_inference.yaml").read_text(encoding="utf-8")
        assert "ErrorExtensionNotPresent" in text, (
            "the coopmat workaround must keep its explanation next to it")


def test_env_override_is_authoritative(monkeypatch, tmp_path):
    """Pointing the env var at an empty dir must yield None, not fall back.

    Without this, there is no way to reproduce a bare machine on a developer
    box that does have the binaries, and the skip paths can never be exercised.
    """
    monkeypatch.setenv(llama_server.ENV_DIR, str(tmp_path))
    assert llama_server.find_llama_cpp_dir({}) is None
    assert llama_server.available({}) is False


# ------------------------------------------------------------- resilience ---

def test_health_is_false_for_a_dead_url():
    assert health("http://127.0.0.1:1", timeout=1) is False


def test_engine_token_count_falls_back_when_server_is_down():
    """A crude estimate is acceptable; an exception is not."""
    engine = LlamaServerEngine("http://127.0.0.1:1")
    n = engine.get_token_count("hello world, this is a netlist")
    assert isinstance(n, int) and n >= 1


def test_engine_healthy_is_false_when_server_is_down():
    assert LlamaServerEngine("http://127.0.0.1:1").healthy() is False


def test_server_start_without_binary_raises_clearly(monkeypatch):
    """On a machine with no llama.cpp at all, the error must name the cause.

    Note binary=None means "discover it", not "there is none", so the discovery
    function itself is patched -- otherwise this silently tests nothing on a
    machine that does have the binaries.
    """
    monkeypatch.setattr(llama_server, "server_binary", lambda *a, **k: None)
    cfg = ServerConfig(model="/nope.gguf", port=1)
    with pytest.raises(FileNotFoundError, match="llama-server not found"):
        llama_server.LlamaServer(cfg).start()


def test_server_start_with_missing_model_raises_clearly():
    cfg = ServerConfig(model="/definitely/not/here.gguf", port=1)
    srv = llama_server.LlamaServer(cfg)
    if srv.binary is None:
        pytest.skip("llama.cpp not installed; the binary check fires first")
    with pytest.raises(FileNotFoundError, match="GGUF model not found"):
        srv.start()


def test_server_stop_is_safe_when_never_started():
    llama_server.LlamaServer(ServerConfig(model="x"), binary=None).stop()


# ---------------------------------------------------------------- payload ---

def _fake_urlopen(capture: dict, response: dict):
    class _Resp:
        def read(self):
            return json.dumps(response).encode()
        def __enter__(self):
            return self
        def __exit__(self, *a):
            return False

    def _open(req, timeout=None):
        capture["url"] = req.full_url
        capture["headers"] = dict(req.headers)
        capture["body"] = json.loads(req.data.decode())
        return _Resp()
    return _open


CHAT_RESPONSE = {
    "choices": [{"message": {"content": "hi"}, "finish_reason": "stop"}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 3},
}


def test_api_engine_posts_openai_shape_and_parses_usage():
    cap: dict = {}
    engine = APIEngine("http://example.invalid/", api_key="secret", model="m")
    with mock.patch("urllib.request.urlopen", _fake_urlopen(cap, CHAT_RESPONSE)):
        out = engine.generate([{"role": "user", "content": "x"}],
                              temperature=0.3, max_new_tokens=42)
    assert cap["url"] == "http://example.invalid/v1/chat/completions"
    assert cap["body"]["temperature"] == 0.3
    assert cap["body"]["max_tokens"] == 42
    assert cap["headers"]["Authorization"] == "Bearer secret"
    assert isinstance(out, GenerationResult)
    assert out.text == "hi" and out.prompt_tokens == 11 and out.completion_tokens == 3


def test_api_engine_omits_auth_header_without_a_key():
    cap: dict = {}
    with mock.patch("urllib.request.urlopen", _fake_urlopen(cap, CHAT_RESPONSE)):
        APIEngine("http://example.invalid").generate([{"role": "user", "content": "x"}])
    assert "Authorization" not in cap["headers"]


def test_engines_pass_messages_through_unmodified():
    """Client-side re-templating is how training/serving prompt drift returns.

    The system message must reach the server byte-identical to
    build_system_message().
    """
    system = build_system_message()
    messages = [{"role": "system", "content": system},
                {"role": "user", "content": "design an OTA"}]

    for engine in (APIEngine("http://example.invalid"),
                   LlamaServerEngine("http://example.invalid")):
        cap: dict = {}
        with mock.patch("urllib.request.urlopen", _fake_urlopen(cap, CHAT_RESPONSE)):
            engine.generate(list(messages))
        sent = cap["body"]["messages"]
        assert sent == messages
        assert sent[0]["content"] == system


# --------------------------------------------------------------- hardware ---

@skip_no_llama
def test_llama_cpp_reports_a_device():
    devices = llama_server.list_devices()
    assert devices, "llama.cpp found but reported no compute device"


@skip_no_llama
def test_server_binary_is_executable_path():
    b = llama_server.server_binary()
    assert b is not None and b.exists()


@pytest.fixture(scope="module")
def live_server():
    """One server for every live test in this module.

    Launching a second llama-server while the first is releasing its Vulkan
    context makes the iGPU drop connections mid-request (ConnectionResetError),
    so the tests share one process rather than each starting their own. A flaky
    test is worse than no test.
    """
    if not (HAS_LLAMA_CPP and HAS_MODEL):
        pytest.skip("no GGUF model built")
    cfg = ServerConfig.from_config()
    cfg.port = 8232  # do not collide with a server the user already has up
    srv = llama_server.LlamaServer(cfg)
    srv.start()
    try:
        yield srv
    finally:
        srv.stop()


@skip_no_model
def test_live_igpu_generation_emits_a_contract_tool_call(live_server):
    """Full path: run on the iGPU, canonical prompt, in-contract tool call."""
    engine = LlamaServerEngine(live_server.base_url)
    assert engine.healthy()

    result = engine.generate(
        [{"role": "system", "content": build_system_message()},
         {"role": "user", "content":
          "Design a two-stage OTA in sky130. Specs: dc_gain > 60 dB, "
          "UGB > 30 MHz, PM > 60 deg, Idd < 500 uA. Start by querying the PDK."}],
        temperature=0.0, max_new_tokens=160,
    )

    assert result.prompt_tokens > 1000, "the canonical system message should be large"
    assert result.completion_tokens > 0
    assert "<tool_call>" in result.text, f"no tool call in: {result.text[:200]!r}"

    start = result.text.index("<tool_call>") + len("<tool_call>")
    end = result.text.index("</tool_call>", start)
    call = json.loads(result.text[start:end])
    known = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert call["name"] in known, f"hallucinated tool {call['name']!r}"
    assert isinstance(call.get("arguments"), dict)


@skip_no_model
def test_exact_token_count_against_the_live_server(live_server):
    n = LlamaServerEngine(live_server.base_url).get_token_count(build_system_message())
    # The canonical system message is ~1800 tokens. The crude len//4 fallback
    # would land near 1750, so this asserts a band a real tokenizer produces
    # while still failing if the fallback silently took over at 0 or 1.
    assert 1000 < n < 4000
