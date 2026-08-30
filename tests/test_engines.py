"""Engines must generate, or say they cannot. Never return an empty success.

TransformersEngine.generate and VLLMEngine.generate both returned

    GenerationResult(text="", prompt_tokens=0, completion_tokens=0)

which every caller in this repo reads as "the model declined to answer". Both
get_token_count implementations returned len(text.split()), which undercounts
the canonical system message by 1.8x (969 words against 1733 real tokens) and
would silently overflow a context window.

This is the same shape as the adapter returning zeros, the optimizer never
evaluating, and the eval runner reporting a constant pass: a placeholder that
looks like success.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from asic_ai.data.format import TOOL_DEFINITIONS, build_system_message
from asic_ai.inference.engine import (
    APIEngine, GenerationResult, ModelEngine, TransformersEngine, VLLMEngine,
)

REPO_ROOT = Path(__file__).parent.parent
MERGED = REPO_ROOT / "models" / "asic-ai-0.5b-merged"
HAS_MERGED = (MERGED / "config.json").exists()

skip_no_model = pytest.mark.skipif(
    not HAS_MERGED, reason="merged HF checkpoint not built on this machine")


@pytest.fixture(scope="module")
def engine():
    """One loaded model for the whole module.

    Loading the checkpoint takes most of a minute on CPU. Constructing an
    engine per test turned a 17-second suite into a two-minute one for no extra
    coverage -- the tests exercise different behaviours of the same weights.
    """
    if not HAS_MERGED:
        pytest.skip("merged HF checkpoint not built on this machine")
    e = TransformersEngine(str(MERGED), dtype="float32")
    e._load()
    return e


# ------------------------------------------------------------- contracts ----

@pytest.mark.parametrize("cls", [TransformersEngine, VLLMEngine, APIEngine])
def test_engines_implement_the_interface(cls):
    assert issubclass(cls, ModelEngine)


def test_construction_is_cheap_and_does_not_load_weights():
    """A constructor that loads a checkpoint makes every import expensive."""
    e = TransformersEngine("/definitely/not/here")
    assert e._model is None and e._tokenizer is None


def test_transformers_engine_reports_a_real_error_for_a_missing_model():
    """Not an empty string -- an error naming the problem."""
    e = TransformersEngine("/definitely/not/here")
    with pytest.raises(Exception) as exc:
        e.get_token_count("hello")
    assert "not here" in str(exc.value).lower() or "not a" in str(exc.value).lower() \
        or "does not" in str(exc.value).lower() or "no such" in str(exc.value).lower() \
        or "repo" in str(exc.value).lower()


def test_vllm_engine_refuses_instead_of_returning_empty():
    e = VLLMEngine("some/model")
    with pytest.raises(ImportError, match="vLLM"):
        e.generate([{"role": "user", "content": "hi"}])
    with pytest.raises(ImportError):
        e.get_token_count("hi")


def test_vllm_error_points_at_a_working_alternative():
    """An unusable backend should say what to use instead."""
    with pytest.raises(ImportError) as exc:
        VLLMEngine("m").generate([])
    msg = str(exc.value)
    assert "llama_server" in msg and "TransformersEngine" in msg


# -------------------------------------------------------------- real run ----

@skip_no_model
def test_token_count_is_exact_not_a_word_count(engine):
    """The old word count undercounts the system message by nearly half."""
    e = engine
    system = build_system_message()
    exact = e.get_token_count(system)
    words = len(system.split())
    assert exact > 1000, "the canonical system message is ~1700 tokens"
    assert exact > words * 1.5, (
        f"{exact} tokens vs {words} words -- a word count would silently "
        "overflow the context window")


@skip_no_model
def test_generates_real_text_and_real_counts(engine):
    e = engine
    result = e.generate(
        [{"role": "system", "content": build_system_message()},
         {"role": "user", "content": "Design a two-stage OTA in sky130. "
                                     "dc_gain > 60 dB. Start by querying the PDK."}],
        temperature=0.0, max_new_tokens=80)

    assert isinstance(result, GenerationResult)
    assert result.text.strip(), "generate() returned nothing -- the original defect"
    assert result.prompt_tokens > 1000, "the system message alone is ~1700 tokens"
    assert 0 < result.completion_tokens <= 80
    assert result.finish_reason in ("stop", "length")


@skip_no_model
def test_the_hf_path_emits_the_same_contract_tool_call_as_the_gguf_path(engine):
    """Both engines serve the same weights; a divergence means a prompt bug."""
    import json

    e = engine
    result = e.generate(
        [{"role": "system", "content": build_system_message()},
         {"role": "user", "content": "Design a two-stage OTA in sky130. "
                                     "dc_gain > 60 dB. Start by querying the PDK."}],
        temperature=0.0, max_new_tokens=100)

    assert "<tool_call>" in result.text, f"no tool call in {result.text[:200]!r}"
    start = result.text.index("<tool_call>") + len("<tool_call>")
    end = result.text.index("</tool_call>", start)
    call = json.loads(result.text[start:end])
    known = {t["function"]["name"] for t in TOOL_DEFINITIONS}
    assert call["name"] in known, f"hallucinated tool {call['name']!r}"


@skip_no_model
def test_greedy_decoding_is_deterministic(engine):
    """temperature=0 must not sample; a flaky engine makes eval meaningless."""
    e = engine
    messages = [{"role": "user", "content": "List two CMOS amplifier topologies."}]
    a = e.generate(messages, temperature=0.0, max_new_tokens=24)
    b = e.generate(messages, temperature=0.0, max_new_tokens=24)
    assert a.text == b.text
